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

# [উন্নতি ১]: k কমিয়ে ৩ করা হয়েছে এবং score_threshold যোগ করা হয়েছে
# এর ফলে ডাটাবেজের তথ্যের সাথে মিল না থাকলে জোর করে ভুল ডেটা তুলে আনবে না
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.55  # ৫৫% মিল না থাকলে তথ্য রিট্রিভ করবে না
    }
)

# ================= LLM =================

# [মডেল আপডেট]: 'command-r' এর পরিবর্তে একটিভ লেটেস্ট মডেল বসানো হলো
llm = ChatCohere(
    model="command-r-08-2024", 
    cohere_api_key=COHERE_API_KEY,
    temperature=0.0  # মডেলের বানিয়ে কথা বলা বন্ধ রাখবে
)

# ================= PROMPT =================
# [উন্নতি ৩]: প্রম্পটকে আরও কঠোর ও প্রফেশনাল করা হয়েছে যেন মডেল বাউন্ডারি ক্রস না করে
system_prompt = """
You are the official BITAC AI Assistant.

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the provided context below to answer the user's question.
2. If the context does not contain the exact answer, strictly reply with: "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই।"
3. Do not assume, extrapolate, or invent any facts under any circumstances. If the information is missing, say you don't know.
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
        # [🚨 মোস্ট ক্রিশিয়াল]: এটি Render লগে আসল এররটি প্রিন্ট করে দেবে
        import traceback
        print("❌ CRITICAL CHAT ERROR DETECTED:")
        print(traceback.format_exc()) 
        
        # সার্ভার যেন ৫০০ এরর না দিয়ে ফ্রন্টএন্ডে সেফ মেসেজ পাঠায়
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

    // লোডিং এলিমেন্ট তৈরি (Typing... টেক্সটকে সুন্দর করা হয়েছে)
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
        messages.removeChild(typingDiv); // লোডিং রিমুভ

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
