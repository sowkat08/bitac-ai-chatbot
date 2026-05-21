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
# PINECONE + VECTORSTORE
# =========================

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# =========================
# LLM
# =========================

llm = ChatCohere(
    model="command-r-plus",
    cohere_api_key=COHERE_API_KEY,
    temperature=0.3
)

# =========================
# PROMPT (CHAT STYLE)
# =========================

system_prompt = """
You are BITAC AI Assistant.

Rules:
- Answer ONLY using given context
- If not found, say you don't know
- Always reply clearly
- If user writes Bangla, reply in Bangla
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# =========================
# RAG CHAIN
# =========================

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

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
        result = rag_chain.invoke({
            "input": request.message
        })

        answer = result.get("answer") or result.get("output") or "No answer found"

        return {
            "user_question": request.message,
            "chatbot_answer": answer,
            "status": "success"
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="Chat error")

# =========================
# REAL CHAT UI
# =========================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BITAC AI Chatbot</title>

    <style>
        body {
            font-family: Arial;
            background: #f4f4f4;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .chat-box {
            width: 420px;
            height: 600px;
            background: white;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }

        .header {
            background: #2563eb;
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
        }

        .messages {
            flex: 1;
            padding: 10px;
            overflow-y: auto;
        }

        .msg {
            margin: 8px;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
        }

        .user {
            background: #2563eb;
            color: white;
            margin-left: auto;
        }

        .bot {
            background: #e5e7eb;
        }

        .input-box {
            display: flex;
            border-top: 1px solid #ddd;
        }

        input {
            flex: 1;
            padding: 10px;
            border: none;
            outline: none;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 15px;
            cursor: pointer;
        }
    </style>
</head>

<body>

<div class="chat-box">

    <div class="header">
        BITAC AI Chatbot 🤖
    </div>

    <div class="messages" id="messages">
        <div class="msg bot">Hello! Ask me anything 👋</div>
    </div>

    <div class="input-box">
        <input id="input" placeholder="Type your question...">
        <button onclick="send()">Send</button>
    </div>

</div>

<script>

async function send(){

    let input = document.getElementById("input");
    let text = input.value;

    if(!text) return;

    addMessage(text, "user");

    input.value = "";

    let res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: text})
    });

    let data = await res.json();

    addMessage(data.chatbot_answer, "bot");
}

function addMessage(text, type){

    let div = document.createElement("div");

    div.className = "msg " + type;
    div.innerText = text;

    document.getElementById("messages").appendChild(div);
}

</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE
