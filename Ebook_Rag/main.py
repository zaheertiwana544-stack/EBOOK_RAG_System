# =========================================
# RAG ENGINE (CORE LOGIC ONLY)
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
def get_hash(file_path):
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
# BUILD RAG
# ======================
def build_rag():
    print("🚀 Initializing RAG...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = load_vectorstore()
    hashes = load_hashes()

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("PDF not found")

    file_hash = get_hash(PDF_PATH)

    # ======================
    # CHECK IF ALREADY INDEXED
    # ======================
    if file_hash not in hashes or vectorstore is None:
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

        print("✅ Vector DB created/updated")
    else:
        print("✅ Using cached vector DB")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG READY")
    return rag_chain
