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

# [🔥 নতুন আপডেট]: 'similarity_score_threshold' এর পরিবর্তে 'mmr' সার্চ টাইপ ব্যবহার করা হলো
# এটি ইউজার নিজের ভাষায় ঘুরিয়ে প্রশ্ন করলেও তথ্যের মূল অর্থ বা ইনটেন্ট (Intent) ম্যাচ করতে পারে
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,              # মডেলের কাছে একসাথে ৪টি সেরা প্রাসঙ্গিক টুকরো পাঠাবে
        "fetch_k": 10,       # ডাটাবেজ থেকে প্রথমে ১০টি সম্ভাব্য খণ্ড টানবে, তারপর ফিল্টার করবে
        "lambda_mult": 0.6   # তথ্যের বৈচিত্র্য এবং মিলের মধ্যে একটি পারফেক্ট ব্যালেন্স রাখবে
    }
)

# ================= LLM =================
llm = ChatCohere(
    model="command-r-08-2024", 
    cohere_api_key=COHERE_API_KEY,
    temperature=0.0  # মডেলের নিজের থেকে তথ্য বানিয়ে বানিয়ে বাড়িয়ে কথা বলা বন্ধ রাখবে
)

# ================= PROMPT =================
# [🔥 নতুন আপডেট]: প্রম্পটকে কিছুটা বুদ্ধিমান করা হয়েছে যাতে হুবহু বাক্য না মিললেও কনটেক্সটের অর্থ বুঝে বট উত্তর দিতে পারে
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
    # ১. ইনপুট খালি কি না চেক করা
    if not req.message or not req.message.strip():
        return {"question": req.message, "answer": "অনুগ্রহ করে কিছু লিখুন।"}

    try:
        print(f"💬 Incoming Question: {req.message}")
        
        # ২. ল্যাংচেইন চেইন রান করা
        result = rag_chain.invoke({"input": req.message})
        
        # ৩. উত্তর এক্সট্রাক্ট করা
        answer = result.get("answer", "").strip()

        # উত্তর যদি খালি আসে বা মডেল না চেনে
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
    typingDiv.innerHTML = "<i>বট ভাবছে...</i>";
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;

    try {
        let res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: text})
        });

        let data = await res.json();
        messages.removeChild(typingDiv); 

        if (data.answer) {
            addMessage(data.answer, "bot");
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
