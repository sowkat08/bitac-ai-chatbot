import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ================= ১. অ্যাপ সেটআপ ও CORS =================
app = FastAPI(title="BITAC AI Smart Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ২. এনভায়রনমেন্ট ভেরিয়েবল =================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "bitac-chatbot")

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("Missing API keys in Environment Variables!")

# ================= ৩. পাইনকোন ও রিট্রিভার সেটআপ =================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# ⚡ [স্পীড বুস্ট]: MMR এর ল্যাগ বাদ দিয়ে similarity সার্চ এবং k=5 করা হয়েছে সঠিক উত্তরের জন্য
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ================= ৪. লার্জ ল্যাঙ্গুয়েজ মডেল (LLM) =================
llm = ChatCohere(
    model="command-r-08-2024", 
    cohere_api_key=COHERE_API_KEY,
    temperature=0.0,
    streaming=True
)

# ================= ৫. মূল সিস্টেম প্রম্পট (বাংলায় নিখুঁত গাইডলাইন) =================
system_prompt = """
তুমি হলে BITAC (বিটাক)-এর অফিসিয়াল এআই অ্যাসিস্ট্যান্ট। 

নিয়মাবলী (কঠোরভাবে পালনীয়):
১. নিচে দেওয়া "Context"-এর ভেতরের অফিসিয়াল তথ্যের ওপর ভিত্তি করে ইউজারের প্রশ্নের সরাসরি উত্তর দাও। নিজের থেকে কোনো তথ্য অনুমান বা আবিষ্কার করবে না।
২. যদি প্রশ্নের উত্তর Context-এ সরাসরি না থাকে, কিন্তু প্রাসঙ্গিক (Related) কোনো তথ্য থাকে, তবে সেই প্রাসঙ্গিক তথ্যটি ব্যবহার করে সুন্দর করে বুঝিয়ে বলো।
৩. যদি কোনো তথ্যই বা প্রাসঙ্গিক কোনো লাইন Context-এ না থাকে, তবে বানিয়ে কিছু বলবে না। সরাসরি বলবে: "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই।"
৪. ইউজার যদি আঞ্চলিক ভাষায় বা বাংলিশে (Banglish) প্রশ্ন করে, তবে তার মূল উদ্দেশ্য বুঝে Context-এর সাথে মিলিয়ে যুক্তিসঙ্গত উত্তর দাও।
৫. উত্তর সবসময় স্পষ্ট, সহজ এবং প্রাঞ্জল বাংলায় হতে হবে।

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}")
])

doc_chain = create_stuff_documents_chain(llm, prompt)

# ================= ৬. ডাটা মডেল =================
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ================= ৭. স্মার্ট রেসপন্স জেনারেটর (Query Optimizer + Streaming) =================
async def response_generator(query: str, chat_history: list):
    try:
        optimized_query = query
        history_context = ""
        
        if chat_history:
            history_context = "\\n".join([f"{type(m).__name__}: {m.content}" for m in chat_history[-2:]])
        
        # ⚡ [ইউনিভার্সাল অপ্টিমাইজার]: ১ শব্দ বা সংক্ষিপ্ত প্রশ্নকে ডকের ভেতরের বড় বাক্যে কনভার্ট করবে (সব বিষয়ের জন্য)
        optimization_prompt = f"""
        Task: Convert the user's short, informal, or single-word question into 2-3 formal Bangla search keywords/phrases to find the exact matching documents from a BITAC vector database.
        
        Rules:
        1. If the user provides a short keyword (e.g., "asset", "ফি", "যোগ্যতা", "hostel", "sepa"), expand it to its full official meaning contextually related to BITAC (e.g., "ASSET প্রকল্পের আওতাধীন প্রশিক্ষণের ট্রেডসমূহ", "বিটাক কোর্সের ফি", "ভর্তির যোগ্যতা", "হোস্টেল আবাসন সুবিধা").
        2. Combine the current question with the recent chat history to make the query precise.
        3. Do not assume anything outside BITAC's context.
        
        Chat History:
        {history_context}
        
        Current User Question: {query}
        
        Instructions: Output ONLY the expanded Bangla search phrases. Do not write any English, explanations, or punctuation.
        """
        
        opt_response = await llm.ainvoke(optimization_prompt)
        optimized_query = opt_response.content.strip()
        print(f"🔍 Optimized Search Query: {optimized_query}")
        
        # লেভেল ১ সার্চ: অপ্টিমাইজড কুয়েরি দিয়ে খোঁজা
        docs = await asyncio.to_thread(retriever.get_relevant_documents, optimized_query)
        
        # লেভেল ২ সার্চ (Fallback): যদি ডেটা না পায়, মূল প্রশ্ন দিয়ে খোঁজা
        if not docs:
            docs = await asyncio.to_thread(retriever.get_relevant_documents, query)

        # লেভেল ৩ সার্চ (Fallback 2): ১ শব্দের ক্ষেত্রে নিরাপদ ব্যাকআপ
        if not docs and len(query.split()) == 1:
            docs = await asyncio.to_thread(retriever.get_relevant_documents, f"{query} বিটাক")

        # ক্লায়েন্ট ব্রাউজারে ডাটা স্ট্রিমিং শুরু
        async for event in doc_chain.astream({
            "input": query,
            "chat_history": chat_history,
            "context": docs
        }):
            if event:
                yield event
    except Exception as e:
        yield f"ত্রুটি ঘটেছে: {str(e)}"

# ================= ৮. চ্যাট এপিআই এন্ডপয়েন্ট =================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        return StreamingResponse((add for add in ["অনুগ্রহ করে কিছু লিখুন।"]), media_type="text/plain")

    print(f"💬 Incoming Question: {req.message}")
    
    chat_history = []
    for msg in req.history:
        if msg.get("type") == "user":
            chat_history.append(HumanMessage(content=msg.get("text")))
        elif msg.get("type") == "bot":
            chat_history.append(AIMessage(content=msg.get("text")))

    return StreamingResponse(response_generator(req.message, chat_history), media_type="text/plain")

# ================= ৯. ইউজার ইন্টারফেস (UI with Live Streaming View) =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>BITAC AI Chatbot</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; background: #0f172a; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .chatbox { width: 420px; height: 650px; background: white; border-radius: 15px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .header { background: #1e3a8a; color: white; padding: 15px; text-align: center; font-weight: bold; }
        .messages { flex: 1; padding: 10px; overflow-y: auto; background: #f1f5f9; }
        .msg { margin: 8px 0; padding: 10px; border-radius: 10px; max-width: 80%; white-space: pre-wrap; font-size: 14px; line-height: 1.4; }
        .user { background: #2563eb; color: white; margin-left: auto; }
        .bot { background: #e5e7eb; color: #1e293b; margin-right: auto; }
        .input-box { display: flex; border-top: 1px solid #ddd; }
        input { flex: 1; padding: 12px; border: none; outline: none; }
        button { padding: 12px 15px; border: none; background: #2563eb; color: white; cursor: pointer; font-weight: bold; }
        button:hover { background: #1d4ed8; }
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
let chatHistory = []; 

function addMessage(text, type) {
    let div = document.createElement("div");
    div.className = "msg " + type;
    div.innerText = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
}

function handleKeyPress(e) {
    if (e.key === 'Enter') { send(); }
}

async function send() {
    let input = document.getElementById("input");
    let text = input.value.trim();

    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    let botMessageDiv = addMessage("", "bot");
    
    try {
        let res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                message: text,
                history: chatHistory 
            })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            fullAnswer += chunk;
            botMessageDiv.innerText = fullAnswer; 
            messages.scrollTop = messages.scrollHeight;
        }

        if (fullAnswer.trim()) {
            chatHistory.push({type: "user", text: text});
            chatHistory.push({type: "bot", text: fullAnswer.trim()});
            
            if (chatHistory.length > 10) {
                chatHistory.shift();
                chatHistory.shift();
            }
        } else {
            botMessageDiv.innerText = "দুঃখিত, কোনো উত্তর পাওয়া যায়নি।";
        }
    } catch (error) {
        botMessageDiv.innerText = "সার্ভারের সাথে যোগাযোগ করা যাচ্ছে না।";
    }
}
</script>

</body>
</html>
"""
