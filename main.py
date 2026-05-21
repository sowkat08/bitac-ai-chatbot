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
app = FastAPI(title="BITAC AI Smart Chatbot")

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

Use only the context below.

If answer not found, say "I don't know".

Always reply clearly.
If user writes Bangla, reply in Bangla.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

# ================= REQUEST =================
class ChatRequest(BaseModel):
    message: str

# ================= CHAT API =================
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = rag_chain.invoke({"input": req.message})
        answer = result.get("answer", "No answer found")

        return {
            "question": req.message,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= UI (SMART CHAT) =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>BITAC AI Chatbot</title>
    <style>
        body {
            margin: 0;
            font-family: Arial;
            background: #0f172a;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .chatbox {
            width: 420px;
            height: 650px;
            background: white;
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: #1e3a8a;
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
        }

        .messages {
            flex: 1;
            padding: 10px;
            overflow-y: auto;
            background: #f1f5f9;
        }

        .msg {
            margin: 8px 0;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            white-space: pre-wrap;
        }

        .user {
            background: #2563eb;
            color: white;
            margin-left: auto;
        }

        .bot {
            background: #e5e7eb;
            margin-right: auto;
        }

        .input-box {
            display: flex;
            border-top: 1px solid #ddd;
        }

        input {
            flex: 1;
            padding: 12px;
            border: none;
            outline: none;
        }

        button {
            padding: 12px 15px;
            border: none;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .typing {
            font-size: 12px;
            color: gray;
            margin-left: 10px;
        }
    </style>
</head>

<body>

<div class="chatbox">
    <div class="header">BITAC AI Smart Chatbot 🚀</div>

    <div class="messages" id="messages"></div>

    <div class="input-box">
        <input id="input" placeholder="Ask something..." />
        <button onclick="send()">Send</button>
    </div>
</div>

<script>
const messages = document.getElementById("messages");

function addMessage(text, type) {
    let div = document.createElement("div");
    div.className = "msg " + type;
    div.innerText = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

async function send() {
    let input = document.getElementById("input");
    let text = input.value.trim();

    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    addMessage("Typing...", "bot");

    let res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: text})
    });

    let data = await res.json();

    // remove typing
    messages.removeChild(messages.lastChild);

    addMessage(data.answer, "bot");
}
</script>

</body>
</html>
    """
