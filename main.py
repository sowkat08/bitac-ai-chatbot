import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere

from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
INDEX_NAME = os.getenv("INDEX_NAME", "bitac-chatbot")

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

# [🔥 MMR রিট্রিভার]: তথ্যের মূল অর্থ বা ইনটেন্ট ম্যাচ করার জন্য
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,              
        "fetch_k": 10,       
        "lambda_mult": 0.6   
    }
)

# ================= LLM =================
llm = ChatCohere(
    model="command-r-08-2024", 
    cohere_api_key=COHERE_API_KEY,
    temperature=0.0  
)

# ================= CONTEXTUALIZE QUESTION PROMPT (MEMORY RETRIEVER) =================
# এই অংশটি চ্যাট হিস্ট্রি দেখে ইউজারের আধো-আধো বা ছোট প্রশ্নকে পূর্ণাঙ্গ প্রশ্নে রূপান্তর করবে
contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# হিস্ট্রি ট্র্যাক করার স্মার্ট রিট্রিভার তৈরি
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# ================= MAIN PROMPT =================
system_prompt = """
You are the official BITAC AI Assistant.

Instructions:
1. Answer the user's question by carefully understanding the core meaning of the provided context below.
2. Even if the user words the question differently or asks in natural spoken language/Banglish, match the intent with the context and answer logically.
3. Do not assume, extrapolate, or invent any facts. If the information is completely missing from the context, strictly reply with: "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই।"
4. If the user greets you (e.g., Hi, Hello, কেমন আছেন), reply politely.
5. If the user writes in Bangla, reply clearly in Bangla. If English, reply in English.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), # মূল প্রম্পটেও মেমোরি ইনজেক্ট করা হলো
    ("human", "{input}")
])

doc_chain = create_stuff_documents_chain(llm, prompt)
# এখানে মূল রিট্রিভারের বদলে মেমোরি-অ্যাওয়ার রিট্রিভারটি দেওয়া হলো
rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

# ================= REQUEST MODEL =================
class ChatRequest(BaseModel):
    message: str
    history: list = [] # ফ্রন্টঅ্যান্ডের চ্যাট হিস্ট্রি রিসিভ করার অ্যারে

# ================= CHAT API =================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        return {"question": req.message, "answer": "অনুগ্রহ করে কিছু লিখুন।"}

    try:
        print(f"💬 Incoming Question: {req.message}")
        
        # ফ্রন্টঅ্যান্ড থেকে আসা মেমোরি লিস্টকে ল্যাংচেইনের মেসেজ ফরম্যাটে (Human/AI) রূপান্তর
        chat_history = []
        for msg in req.history:
            if msg.get("type") == "user":
                chat_history.append(HumanMessage(content=msg.get("text")))
            elif msg.get("type") == "bot":
                chat_history.append(AIMessage(content=msg.get("text")))
        
        # চেইনে ইনপুট এবং চ্যাট হিস্ট্রি একসাথে পাস করা
        result = rag_chain.invoke({
            "input": req.message,
            "chat_history": chat_history
        })
        
        answer = result.get("answer", "").strip()

        if not answer or "I don't know" in answer:
            answer = "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই।"

        return {
            "question": req.message,
            "answer": answer
        }

    except Exception as e:
        import traceback
        print("❌ CRITICAL CHAT ERROR DETECTED:")
        print(traceback.format_exc()) 
        
        return {
            "question": req.message,
            "answer": "দুঃখিত, এই মুহূর্তে উত্তর তৈরি করা যাচ্ছে না। অনুগ্রহ করে Render-এর Logs ট্যাব চেক করুন।"
        }

# ================= UI (SMART CHAT WITH MEMORY) =================
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
            font-family: Arial, sans-serif;
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
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
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
            font-size: 14px;
            line-height: 1.4;
        }

        .user {
            background: #2563eb;
            color: white;
            margin-left: auto;
        }

        .bot {
            background: #e5e7eb;
            color: #1e293b;
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
            font-weight: bold;
        }

        button:hover {
            background: #1d4ed8;
        }
    </style>
</head>
<body>

<div class="chatbox">
    <div class="header">BITAC AI Smart Chatbot 🚀</div>
    <div class="messages" id="messages"></div>
    <div class="input-box">
        <input id="input" placeholder="Ask something..." onkeypress="handleKeyPress(event)" />
        <button onclick="send()">Send</button>
    </div>
</div>

<script>
const messages = document.getElementById("messages");
let chatHistory = []; // [🔥 মেমোরি আপডেট]: ব্রাউজারে হিস্ট্রি সেভ রাখার গ্লোবাল ভেরিয়েবল

function addMessage(text, type) {
    let div = document.createElement("div");
    div.className = "msg " + type;
    div.innerText = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function handleKeyPress(e) {
    if (e.key === 'Enter') {
        send();
    }
}

async function send() {
    let input = document.getElementById("input");
    let text = input.value.trim();

    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    let typingDiv = document.createElement("div");
    typingDiv.className = "msg bot";
    typingDiv.innerHTML = "<i>...</i>";
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;

    try {
        // [🔥 মেমোরি আপডেট]: রিকোয়েস্ট বডিতে এখন চ্যাট হিস্ট্রিও পাঠানো হচ্ছে
        let res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                message: text,
                history: chatHistory 
            })
        });

        let data = await res.json();
        messages.removeChild(typingDiv); 

        if (data.answer) {
            addMessage(data.answer, "bot");
            
            // সফলভাবে উত্তর আসার পর কারেন্ট মেসেজ জোড়া মেমোরিতে পুশ করা হচ্ছে
            chatHistory.push({type: "user", text: text});
            chatHistory.push({type: "bot", text: data.answer});
            
            // মেমোরি খুব বেশি বড় হয়ে যেন ব্রাউজার স্লো না করে (সর্বোচ্চ শেষ ৫ জোড়া কথা মনে রাখবে)
            if (chatHistory.length > 10) {
                chatHistory.shift();
                chatHistory.shift();
            }
        } else {
            addMessage("দুঃখিত, কোনো উত্তর পাওয়া যায়নি।", "bot");
        }
    } catch (error) {
        messages.removeChild(typingDiv);
        addMessage("সার্ভারের সাথে যোগাযোগ করা যাচ্ছে না।", "bot");
    }
}
</script>

</body>
</html>
"""
