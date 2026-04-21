# =========================================
# FASTAPI BACKEND (DEPLOY THIS FILE)
# =========================================

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

from main import build_rag

# ======================
# INIT APP
# ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# LOAD RAG ON START
# ======================
rag_chain = build_rag()


# ======================
# REQUEST MODEL
# ======================
class ChatRequest(BaseModel):
    question: str


# ======================
# ROOT
# ======================
@app.get("/")
def home():
    return {"message": "AI RAG Agent Running 🚀"}


# ======================
# CHAT ENDPOINT
# ======================
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        answer = rag_chain.invoke(req.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
