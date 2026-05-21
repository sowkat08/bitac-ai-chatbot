import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import ChatPromptTemplate

# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="BITAC Smart Cohere Chatbot")

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV VARIABLES
# =========================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing!")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY missing!")

# =========================
# PINECONE
# =========================

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# =========================
# EMBEDDING MODEL
# =========================

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# =========================
# VECTOR STORE
# =========================

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# =========================
# RETRIEVER
# =========================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)

# =========================
# LLM
# =========================

llm = ChatCohere(
    model="command-r",
    cohere_api_key=COHERE_API_KEY,
    temperature=0.3
)

# =========================
# PROMPT
# =========================

system_prompt = """
You are an advanced AI assistant for BITAC
(Bangladesh Industrial Technical Assistance Center).

Use the following retrieved context to answer
the user's question accurately.

If the user asks in Bengali,
reply beautifully in Bengali.

If the answer is not available in the context,
say politely that you do not know.

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

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    message: str

# =========================
# HTML UI
# =========================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>BITAC AI Tech-Bot</title>

    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"
          rel="stylesheet">

    <style>

        *{
            margin:0;
            padding:0;
            box-sizing:border-box;
            font-family:'Inter',sans-serif;
        }

        body{
            background:#f3f4f6;
            display:flex;
            justify-content:center;
            align-items:center;
            height:100vh;
        }

        .chat-container{
            width:100%;
            max-width:520px;
            height:85vh;
            background:#fff;
            border-radius:18px;
            overflow:hidden;
            display:flex;
            flex-direction:column;
            box-shadow:0 10px 25px rgba(0,0,0,0.08);
        }

        .chat-header{
            background:#1e3a8a;
            color:white;
            padding:20px;
        }

        .chat-header h2{
            font-size:20px;
        }

        .chat-box{
            flex:1;
            padding:20px;
            overflow-y:auto;
            background:#fafafa;
            display:flex;
            flex-direction:column;
            gap:14px;
        }

        .message{
            max-width:80%;
            padding:12px 16px;
            border-radius:14px;
            line-height:1.6;
            font-size:14px;
            word-wrap:break-word;
        }

        .user{
            align-self:flex-end;
            background:#2563eb;
            color:white;
            border-bottom-right-radius:4px;
        }

        .bot{
            align-self:flex-start;
            background:#e5e7eb;
            color:#111827;
            border-bottom-left-radius:4px;
        }

        .input-area{
            padding:15px;
            display:flex;
            gap:10px;
            border-top:1px solid #e5e7eb;
            background:white;
        }

        .input-area input{
            flex:1;
            padding:12px;
            border:1px solid #d1d5db;
            border-radius:10px;
            outline:none;
            font-size:14px;
        }

        .input-area button{
            border:none;
            background:#2563eb;
            color:white;
            padding:0 20px;
            border-radius:10px;
            cursor:pointer;
            font-weight:600;
        }

        .input-area button:hover{
            opacity:0.9;
        }

    </style>
</head>

<body>

<div class="chat-container">

    <div class="chat-header">
        <h2>BITAC AI Tech-Bot 🚀</h2>
    </div>

    <div class="chat-box" id="chat-box">

        <div class="message bot">
            হ্যালো 👋 <br>
            আমি BITAC AI Assistant।<br>
            কীভাবে সাহায্য করতে পারি?
        </div>

    </div>

    <div class="input-area">

        <input
            type="text"
            id="user-input"
            placeholder="Type your message..."
        >

        <button onclick="sendMessage()">
            Send
        </button>

    </div>

</div>

<script>

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");

function addMessage(text, sender){

    const div = document.createElement("div");

    div.classList.add("message");
    div.classList.add(sender);

    div.innerText = text;

    chatBox.appendChild(div);

    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage(){

    const text = userInput.value.trim();

    if(!text) return;

    addMessage(text, "user");

    userInput.value = "";

    try{

        const response = await fetch("/chat", {

            method:"POST",

            headers:{
                "Content-Type":"application/json"
            },

            body:JSON.stringify({
                message:text
            })
        });

        const data = await response.json();

        addMessage(data.chatbot_response, "bot");

    }catch(error){

        addMessage(
            "সার্ভারের সাথে সংযোগ করা যাচ্ছে না।",
            "bot"
        );
    }
}

userInput.addEventListener("keypress", function(event){

    if(event.key === "Enter"){
        sendMessage();
    }
});

</script>

</body>
</html>
"""

# =========================
# CHAT ENDPOINT
# =========================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        response = rag_chain.invoke({
            "input": request.message
        })

        return {
            "chatbot_response": response["answer"],
            "status": "success"
        }

    except Exception as e:

        print("ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )

# =========================
# HOME PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE
