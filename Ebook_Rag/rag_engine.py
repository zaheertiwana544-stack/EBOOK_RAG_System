# =========================================
# RAG CORE ENGINE (MAIN MODULE)
# =========================================

import os
import pickle
import json
import hashlib
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ======================
# LOAD ENV
# ======================
load_dotenv()

# ======================
# CONFIG
# ======================
PKL_FILE = "faiss_index.pkl"
HASH_FILE = "indexed_books.json"
PDF_PATH = "Python.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ======================
# HELPERS
# ======================
def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_hashes():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}


def save_hashes(data):
    with open(HASH_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_vectorstore():
    if os.path.exists(PKL_FILE):
        with open(PKL_FILE, "rb") as f:
            return pickle.load(f)
    return None


def save_vectorstore(vs):
    with open(PKL_FILE, "wb") as f:
        pickle.dump(vs, f)


# ======================
# BUILD RAG SYSTEM
# ======================
def build_rag():
    print("\n🚀 Initializing RAG System...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing
    vectorstore = load_vectorstore()
    hashes = load_hashes()

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"{PDF_PATH} not found")

    file_hash = get_file_hash(PDF_PATH)

    # ======================
    # CHECK DUPLICATION
    # ======================
    if file_hash in hashes and vectorstore:
        print("✅ Using existing vector DB")
    else:
        print("📄 Processing PDF...")

        loader = PyMuPDFLoader(PDF_PATH)
        docs = loader.load()

        chunks = splitter.split_documents(docs)

        if vectorstore:
            vectorstore.add_documents(chunks)
        else:
            vectorstore = FAISS.from_documents(chunks, embeddings)

        hashes[file_hash] = PDF_PATH
        save_hashes(hashes)
        save_vectorstore(vectorstore)

        print("✅ Vector DB updated")

    # ======================
    # RETRIEVER
    # ======================
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ======================
    # LLM
    # ======================
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # ======================
    # PROMPT
    # ======================
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ======================
    # RAG CHAIN
    # ======================
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG System Ready!\n")
    return rag_chain


# ======================
# CLI TEST MODE
# ======================
if __name__ == "__main__":
    rag_chain = build_rag()

    print("=" * 50)
    print("📚 RAG CHAT SYSTEM (CLI MODE)")
    print("=" * 50)

    while True:
        q = input("\nAsk question (or 'exit'): ")

        if q.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(q)
        print("\n💡 Answer:\n", response)
        print("-" * 50)
