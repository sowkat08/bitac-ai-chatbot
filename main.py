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

# ================= APP =================
app = FastAPI(title="BITAC AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ENV =================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("Missing API keys")

# ================= PINECONE =================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ================= EMBEDDINGS =================
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ================= LLM =================

llm = ChatCohere(
    model="command-r-plus-08-2024",
    cohere_api_key=COHERE_API_KEY,
    temperature=0.3
)

# ================= PROMPT =================
system_prompt = """
You are BITAC AI Assistant.

Use ONLY the context below:

{context}

Rules:
- If answer not found, say "I don't know"
- Reply clearly
- If Bangla, reply in Bangla
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_chain)

# ================= REQUEST =================
class ChatRequest(BaseModel):
    message: str

# ================= CHAT API =================
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = rag_chain.invoke({"input": request.message})

        answer = result.get("answer") or result.get("output") or str(result)

        return {
            "question": request.message,
            "answer": answer,
            "status": "success"
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ================= UI =================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BITAC AI Chatbot</title>
    <style>
        body {
            font-family: Arial;
            background: #f3f4f6;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .box {
            width: 420px;
            height: 600px;
            background: white;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }

        .header {
            background: #1e3a8a;
            color: white;
            padding: 15px;
            text-align: center;
        }

        .chat {
            flex: 1;
            padding: 10px;
            overflow-y: auto;
        }

        .msg {
            margin: 8px 0;
            padding: 10px;
            border-radius: 10px;
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

        .input {
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
            padding: 10px 15px;
            background: #2563eb;
            color: white;
            border: none;
            cursor: pointer;
        }
    </style>
</head>

<body>

<div class="box">
    <div class="header">
        BITAC AI Chatbot 🚀
    </div>

    <div class="chat" id="chat"></div>

    <div class="input">
        <input id="msg" placeholder="Type your question..." />
        <button onclick="send()">Send</button>
    </div>
</div>

<script>
async function send() {
    let input = document.getElementById("msg");
    let chat = document.getElementById("chat");

    let text = input.value;
    if (!text) return;

    chat.innerHTML += `<div class='msg user'>${text}</div>`;
    input.value = "";

    let res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: text})
    });

    let data = await res.json();

    chat.innerHTML += `<div class='msg bot'>${data.answer}</div>`;
    chat.scrollTop = chat.scrollHeight;
}
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_TEMPLATE
