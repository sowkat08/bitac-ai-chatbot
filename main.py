import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# =========================
# APP
# =========================

app = FastAPI(title="BITAC AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV
# =========================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("Missing API keys")

# =========================
# PINECONE
# =========================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# =========================
# EMBEDDINGS
# =========================

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# =========================
# VECTOR STORE
# =========================

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# =========================
# LLM (FIXED MODEL)
# =========================

llm = ChatCohere(
    model="command-r-plus",
    cohere_api_key=COHERE_API_KEY,
    temperature=0.3
)

# =========================
# PROMPT
# =========================

system_prompt = """
You are an AI assistant for BITAC.

Use ONLY the given context to answer.
If answer not found, say politely you don't know.

If user asks in Bangla, respond in Bangla.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# =========================
# RAG CHAIN
# =========================

question_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_chain)

# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    message: str

# =========================
# CHAT API
# =========================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:
        response = rag_chain.invoke({
            "input": request.message
        })

        answer = response.get("answer") or response.get("output") or "No answer found"

        return {
            "chatbot_response": answer,
            "status": "success"
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="Chat error")

# =========================
# UI
# =========================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BITAC AI Chatbot</title>
</head>
<body>
    <h2>BITAC AI Chatbot is Running 🚀</h2>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE
