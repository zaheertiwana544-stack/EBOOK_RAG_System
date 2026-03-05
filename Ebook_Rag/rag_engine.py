# rag_engine.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import fitz  # PyMuPDF
import os
import re
import json
from typing import List, Tuple, Optional, Dict, Any

class EbookRAG:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.chat_history = []
        self.current_book = None
        self.toc_structure = []  # Enhanced hierarchical TOC
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
            groq_api_key=groq_api_key
        )
        
        self.prompt = ChatPromptTemplate.from_template("""You're a friendly book expert chatting with a reader.

Context from book:
{context}

Question: {question}

Respond naturally:
- Be conversational, not robotic
- Answer directly without over-explaining
- For greetings ("hey", "hi"), just say hello back
- If unsure, say "I'm not sure about that" briefly
- Only cite pages when giving specific facts
- Keep answers under 3 sentences unless asked for detail

Your response:""")
    
    def extract_structure_from_pdf(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Extract document hierarchy using PyMuPDF's TOC and font analysis
        """
        doc = fitz.open(pdf_path)
        structure_map = {}
        
        # 1. Get PDF's built-in Table of Contents
        toc = doc.get_toc()
        # Format: [[level, title, page_number], ...]
        
        for level, title, page_num in toc:
            if page_num not in structure_map:
                structure_map[page_num] = []
            
            # Classify by level and content
            item_type = self._classify_toc_entry(level, title)
            
            structure_map[page_num].append({
                "level": level,
                "title": title.strip(),
                "type": item_type
            })
        
        # 2. Detect examples/sections on pages not in TOC by font analysis
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1
            
            # Skip if already has TOC entries
            if page_number in structure_map:
                continue
            
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        flags = span["flags"]
                        
                        # Detect examples by pattern + prominent font
                        if self._is_example_header(text, font_size):
                            if page_number not in structure_map:
                                structure_map[page_number] = []
                            
                            structure_map[page_number].append({
                                "level": 3,
                                "title": text,
                                "type": "example"
                            })
                            break  # Only first match per page
        
        doc.close()
        return structure_map
    
    def _classify_toc_entry(self, level: int, title: str) -> str:
        """Classify TOC entry by level and title content"""
        title_lower = title.lower()
        
        if "example" in title_lower or "ex." in title_lower:
            return "example"
        elif "chapter" in title_lower:
            return "chapter"
        elif level == 1:
            return "chapter"
        elif level == 2:
            return "section"
        elif level == 3:
            return "subsection"
        else:
            return "subsubsection"
    
    def _is_example_header(self, text: str, font_size: float, min_size: float = 11) -> bool:
        """Detect if text is an example header"""
        if font_size < min_size:
            return False
        
        patterns = [
            r'^Example\s+\d+\.?\d*',
            r'^Ex\.\s*\d+\.?\d*',
            r'^\d+\.\d+\s+Example',
            r'^Example\s+[A-Z]',
        ]
        
        return any(re.match(p, text, re.IGNORECASE) for p in patterns)
    
    def process_pdf(self, pdf_path: str, save_name: str = None) -> dict:
        try:
            # Step 1: Extract document structure
            structure_map = self.extract_structure_from_pdf(pdf_path)
            
            # Step 2: Load documents with PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_path)
            raw_documents = loader.load()
            
            # Step 3: Enrich documents with hierarchical metadata
            documents = self._enrich_with_structure(raw_documents, structure_map)
            
            # Step 4: Smart splitting that preserves examples
            chunks = self._smart_chunk(documents)
            
            # Step 5: Create vector store
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.current_book = save_name or os.path.basename(pdf_path)
            
            # Setup retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            self._build_chain()
            
            # Save index and TOC
            if save_name:
                save_path = f"faiss_index_{save_name}"
                self.vectorstore.save_local(save_path)
                
                # Save TOC structure
                with open(f"{save_path}_toc.json", "w", encoding="utf-8") as f:
                    json.dump(self.toc_structure, f, indent=2)
            
            return {
                "success": True,
                "chunks": len(chunks),
                "pages": len(documents),
                "chapters": len([t for t in self.toc_structure if t["type"] == "chapter"]),
                "examples": len([t for t in self.toc_structure if t["type"] == "example"]),
                "book": self.current_book
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _enrich_with_structure(self, documents: List[Document], structure_map: Dict) -> List[Document]:
        """Add hierarchical metadata to documents based on structure map"""
        enriched_docs = []
        
        current_hierarchy = {
            "chapter": "Introduction",
            "section": None,
            "example": None,
            "path": "Introduction"
        }
        
        for doc in documents:
            page_num = doc.metadata.get("page", 0)
            
            # Update hierarchy if this page has structure markers
            if page_num in structure_map:
                for item in structure_map[page_num]:
                    title = item["title"]
                    item_type = item["type"]
                    
                    if item_type == "chapter":
                        current_hierarchy["chapter"] = title
                        current_hierarchy["section"] = None
                        current_hierarchy["example"] = None
                    elif item_type == "section":
                        current_hierarchy["section"] = title
                        current_hierarchy["example"] = None
                    elif item_type == "example":
                        current_hierarchy["example"] = title
                    
                    # Build full path
                    parts = [current_hierarchy["chapter"]]
                    if current_hierarchy["section"]:
                        parts.append(current_hierarchy["section"])
                    if current_hierarchy["example"]:
                        parts.append(current_hierarchy["example"])
                    current_hierarchy["path"] = " > ".join(parts)
                    
                    # Store in TOC structure
                    self.toc_structure.append({
                        **item,
                        "page": page_num,
                        "hierarchy_path": current_hierarchy["path"]
                    })
            
            # Create enriched document
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "chapter": current_hierarchy["chapter"],
                    "section": current_hierarchy["section"],
                    "example": current_hierarchy["example"],
                    "hierarchy_path": current_hierarchy["path"],
                    "book": self.current_book
                }
            )
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    def _smart_chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents while preserving example boundaries"""
        chunks = []
        
        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata
            
            # If document contains an example, use special handling
            if metadata.get("example") or self._contains_example(text):
                example_chunks = self._split_by_examples(text, metadata)
                chunks.extend(example_chunks)
            else:
                # Regular splitting for non-example content
                if len(text) > 1000:
                    paragraphs = text.split('\n\n')
                    current_text = ""
                    
                    for para in paragraphs:
                        if len(current_text) + len(para) < 800:
                            current_text += para + "\n\n"
                        else:
                            if current_text:
                                chunks.append(Document(
                                    page_content=current_text.strip(),
                                    metadata=metadata
                                ))
                            current_text = para + "\n\n"
                    
                    if current_text:
                        chunks.append(Document(
                            page_content=current_text.strip(),
                            metadata=metadata
                        ))
                else:
                    chunks.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
        
        return chunks
    
    def _contains_example(self, text: str) -> bool:
        """Check if text contains example headers"""
        patterns = [
            r'Example\s+\d+\.?\d*',
            r'Ex\.\s*\d+\.?\d*',
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)
    
    def _split_by_examples(self, text: str, metadata: Dict) -> List[Document]:
        """Split text by example headers to keep examples intact"""
        # Pattern to match example headers
        pattern = r'(Example\s+\d+\.?\d*[^\n]*)'
        
        parts = re.split(f'({pattern})', text, flags=re.IGNORECASE)
        
        chunks = []
        current_chunk = ""
        current_meta = metadata.copy()
        
        for part in parts:
            if not part.strip():
                continue
            
            # Check if this is an example header
            if re.match(pattern, part, re.IGNORECASE):
                # Save previous chunk if exists
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=current_meta
                    ))
                
                # Start new chunk with this example
                current_chunk = part + "\n"
                
                # Update metadata for this example
                example_name = part.strip()
                current_meta = metadata.copy()
                current_meta["example"] = example_name
                current_meta["hierarchy_path"] = f"{metadata['hierarchy_path']} > {example_name}"
            else:
                current_chunk += part
            
            # Split if too long
            if len(current_chunk) > 1200:
                chunks.append(Document(
                    page_content=current_chunk[:1200].strip(),
                    metadata=current_meta
                ))
                current_chunk = current_chunk[1200:]
        
        # Don't forget last chunk
        if current_chunk.strip():
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=current_meta
            ))
        
        return chunks
    
    def _build_chain(self):
        def format_docs(docs):
            formatted = []
            for doc in docs:
                path = doc.metadata.get("hierarchy_path", "Unknown")
                page = doc.metadata.get("page", "N/A")
                content = doc.page_content[:400]
                formatted.append(f"[{path} | Page {page}]\n{content}")
            return "\n\n---\n\n".join(formatted)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str, use_memory: bool = True) -> Tuple[str, List[int]]:
        if not self.rag_chain:
            return "Please upload a book first!", []
        
        # Detect structural references in question
        struct_filter = self._parse_structural_query(question)
        
        # Retrieve documents
        try:
            if struct_filter:
                # Use metadata filtering for precise retrieval
                docs = self.vectorstore.similarity_search(
                    question,
                    k=5,
                    filter=struct_filter
                )
            else:
                docs = self.retriever.invoke(question)
        except Exception as e:
            # Fallback to regular retrieval if filter fails
            docs = self.retriever.invoke(question)
        
        pages = list(set([d.metadata.get("page") for d in docs if d.metadata.get("page")]))
        
        # Build context
        context_parts = []
        for doc in docs:
            path = doc.metadata.get("hierarchy_path", "Unknown")
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content[:500]
            context_parts.append(f"[{path} | Page {page}]\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        messages = self.prompt.format_messages(context=context, question=question)
        answer = self.llm.invoke(messages).content
        
        # Enhance answer if vague but structural query was detected
        if ("don't know" in answer.lower() or "not sure" in answer.lower()) and struct_filter:
            structure_info = self._get_structure_help(struct_filter)
            if structure_info:
                answer = f"{structure_info}\n\nHowever, I couldn't retrieve the specific content. {answer}"
        
        self.chat_history.append((question, answer))
        return answer, pages
    
    def _parse_structural_query(self, question: str) -> Optional[Dict]:
        """Extract chapter/example references for metadata filtering"""
        question_lower = question.lower()
        filters = {}
        
        # Chapter detection
        chapter_match = re.search(r'chapter\s+(\d+)', question_lower)
        if chapter_match:
            ch_num = chapter_match.group(1)
            # Find exact chapter title from TOC
            for item in self.toc_structure:
                if item["type"] == "chapter" and ch_num in item["title"].lower():
                    filters["chapter"] = item["title"]
                    break
        
        # Example detection
        example_match = re.search(r'example\s+(\d+\.?\d*)', question_lower)
        if example_match:
            ex_num = example_match.group(1)
            # Find example in TOC
            for item in self.toc_structure:
                if item["type"] == "example" and ex_num in item["title"].lower():
                    filters["example"] = item["title"]
                    # Also infer chapter if not set
                    if "chapter" not in filters and " > " in item["hierarchy_path"]:
                        parts = item["hierarchy_path"].split(" > ")
                        filters["chapter"] = parts[0]
                    break
        
        # Section detection
        section_match = re.search(r'section\s+(\d+\.?\d*)', question_lower)
        if section_match and "section" not in filters:
            sec_num = section_match.group(1)
            for item in self.toc_structure:
                if item["type"] == "section" and sec_num in item["title"].lower():
                    filters["section"] = item["title"]
                    break
        
        return filters if filters else None
    
    def _get_structure_help(self, filters: Dict) -> str:
        """Provide structure info when content retrieval fails"""
        parts = []
        
        if "example" in filters:
            ex_title = filters["example"]
            for item in self.toc_structure:
                if item["type"] == "example" and item["title"] == ex_title:
                    parts.append(f"📍 {item['hierarchy_path']} is on page {item['page']}")
                    break
        
        if "chapter" in filters and not parts:
            ch_title = filters["chapter"]
            for item in self.toc_structure:
                if item["type"] == "chapter" and item["title"] == ch_title:
                    # Find sections in this chapter
                    sections = [
                        s["title"] for s in self.toc_structure
                        if s["type"] == "section" and s["hierarchy_path"].startswith(ch_title)
                    ]
                    if sections:
                        parts.append(f"📖 {ch_title} contains: {', '.join(sections[:3])}")
                    break
        
        return "\n".join(parts) if parts else ""
    
    def get_structure_info(self, chapter_num: str = None, example_num: str = None):
        """Get hierarchical structure information"""
        if not self.toc_structure:
            return {"error": "No structure information available. Process a PDF first."}
        
        if example_num:
            for item in self.toc_structure:
                if item["type"] == "example" and example_num in item["title"].lower():
                    return {
                        "found": True,
                        "path": item["hierarchy_path"],
                        "page": item["page"],
                        "title": item["title"]
                    }
            return {"found": False, "message": f"Example {example_num} not found"}
        
        if chapter_num:
            matches = [
                item for item in self.toc_structure
                if item["type"] == "chapter" and chapter_num in item["title"].lower()
            ]
            if matches:
                ch = matches[0]
                # Get contents of this chapter
                contents = [
                    {"type": c["type"], "title": c["title"], "page": c["page"]}
                    for c in self.toc_structure
                    if c["hierarchy_path"].startswith(ch["title"]) and c["type"] != "chapter"
                ]
                return {
                    "found": True,
                    "chapter": ch["title"],
                    "page": ch["page"],
                    "contents": contents[:10]  # First 10 items
                }
            return {"found": False, "message": f"Chapter {chapter_num} not found"}
        
        # Return summary
        return {
            "total_chapters": len([t for t in self.toc_structure if t["type"] == "chapter"]),
            "total_sections": len([t for t in self.toc_structure if t["type"] == "section"]),
            "total_examples": len([t for t in self.toc_structure if t["type"] == "example"]),
            "full_structure": self.toc_structure[:30]  # First 30 items
        }
    
    def clear_memory(self):
        self.chat_history = []
    
    def get_stats(self):
        if not self.vectorstore:
            return {"status": "No book loaded"}
        
        return {
            "status": "Ready",
            "book": self.current_book,
            "vectors": self.vectorstore.index.ntotal,
            "chapters": len([t for t in self.toc_structure if t["type"] == "chapter"]),
            "examples": len([t for t in self.toc_structure if t["type"] == "example"]),
            "history": len(self.chat_history)
        }
