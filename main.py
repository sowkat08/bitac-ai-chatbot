import os
import asyncio
import traceback
import requests
from fastapi import FastAPI, HTTPException, Request 
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ================= ১. অ্যাপ সেটআপ, CORS ও সিকিউরিটি হেডার্স =================
app = FastAPI(title="BITAC AI Smart Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bitac.gov.bd", "https://*.gov.bd", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://bitac.gov.bd https://*.gov.bd;"
    )
    if "X-Frame-Options" in response.headers:
        del response.headers["X-Frame-Options"]
    return response

# ================= ২. এনভায়রনমেন্ট ভেরিয়েবল =================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "bitac-chatbot")

FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
FB_VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "my_secret_bitac_token") 

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

# k=8 এবং mmr করা হয়েছে বড় ফাইল থেকে বেশি এবং বৈচিত্র্যময় ডেটা রিট্রিভ করার জন্য
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)

# ================= ৪. লার্জ ল্যাঙ্গুয়েজ মডেল (LLM) =================
llm = ChatCohere(
    model="command-r-08-2024", 
    cohere_api_key=COHERE_API_KEY,
    temperature=0.0,  # Hallucination বন্ধ করার জন্য 0 করা হয়েছে
    streaming=True
)

# ================= ৫. প্রম্পট সেটআপ =================
system_prompt = """তুমি হলে BITAC (বিটাক)-এর অফিসিয়াল এআই অ্যাসিস্ট্যান্ট। 

নিয়মাবলী (ক কঠোরভাবে পালনীয়):
১. ইউজার যদি সাধারণ সম্ভাষণ (যেমন: Hi, Hello, আসসালামু আলাইকুম, কেমন আছেন, হ্যালো) জানায়, তবে ডাটাবেজে না খুঁজে নিজেই নম্রভাবে বিটাকের পক্ষ থেকে স্বাগত জানাও।
২. ইউজার যদি জানতে চায় "বিটাক কী?" তবে সংক্ষেপে বলো: "বাংলাদেশ শিল্প কারিগরি সহায়তা কেন্দ্র (BITAC) শিল্প মন্ত্রণালয়ের অধীন একটি স্বায়ত্তশাসিত সরকারি প্রতিষ্ঠান, যা মূলত দেশের শিল্প খাতে উৎপাদনশীলতা বৃদ্ধি, দক্ষ জনশক্তি তৈরি এবং কারিগরি সহায়তা প্রদানের লক্ষ্যে কাজ করে।"
৩. তোমার প্রধান কাজ হলো শুধুমাত্র নিচে দেওয়া "Context" থেকে উত্তর খুঁজে বের করা। নিজের জ্ঞান বা অনুমান থেকে কোনো উত্তর বানাবে না (No Hallucination)।
৪. বিশেষ করে SEPA, SICIP, ASSET এবং ADVANCE COURSE সংক্রান্ত প্রশ্নের ক্ষেত্রে অত্যন্ত সতর্ক থাকবে এবং শুধুমাত্র Context-এ দেওয়া তথ্যের ওপর ভিত্তি করে সঠিক উত্তর দেবে।
৫. যদি প্রশ্নের উত্তর Context-এ সরাসরি না থাকে, তবে বানিয়ে বা নিজের মন থেকে মিথ্যা কিছু বলবে না। সরাসরি বলবে: "দুঃখিত, এই বিষয়ে আমার ডাটাবেজে সঠিক তথ্য নেই। আরও তথ্যের জন্য অনুগ্রহ করে বিটাকের অফিসিয়াল ওয়েবসাইট https://bitac.gov.bd ভিজিট করুন।"
৬. প্রয়োজনে ব্যবহারকারীকে বিস্তারিত তথ্যের জন্য বিটাকের অফিসিয়াল ওয়েবসাইট (https://bitac.gov.bd) ভিজিট করতে উৎসাহিত করবে।
৭. উত্তর সবসময় স্পষ্ট, সহজ এবং প্রাঞ্জল বাংলায় হতে হবে।

Context:
{context}"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}")
])

# ================= ৬. ডাটা মডেল =================
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ================= ७. রেসপন্স জেনারেটর =================
async def response_generator(query: str, chat_history: list):
    try:
        clean_query = query.strip().lower()
        
        if clean_query in ["hi", "hello", "hey", "সালাম", "আসসালামু আলাইকুম", "কেমন আছেন", "হ্যালো"]:
            greetings_prompt = f"ইউজার বলেছে: '{query}'। একজন পেশাদার ও বিনয়ী এআই অ্যাসিস্ট্যান্ট হিসেবে বিটাক (BITAC)-এর পক্ষ থেকে তাকে বাংলায় ছোট করে শুভেচ্ছা জানাও এবং কীভাবে সাহায্য করতে পারো জিজ্ঞেস করো।"
            async for chunk in llm.astream(greetings_prompt):
                if chunk.content:
                    yield chunk.content
            return

        optimized_query = query
        if chat_history and len(chat_history) > 0 and len(query.split()) <= 3:
            last_msg = chat_history[-1].content if hasattr(chat_history[-1], 'content') else str(chat_history[-1])
            optimization_prompt = f"Task: Combine last response and current short query to make 2-3 Bangla search keywords.\nLast AI Response: {last_msg}\nUser Short Query: {query}\nKeywords only:"
            try:
                opt_response = await llm.ainvoke(optimization_prompt)
                if opt_response and opt_response.content:
                    optimized_query = opt_response.content.strip()
            except Exception:
                optimized_query = query

        docs = await asyncio.to_thread(retriever.invoke, optimized_query)
        
        if not docs or len(docs) < 3:
            exact_docs = await asyncio.to_thread(retriever.invoke, query)
            existing_contents = {d.page_content for d in docs}
            for d in exact_docs:
                if d.page_content not in existing_contents:
                    docs.append(d)
        
        # ডাটাবেজে কোন তথ্য না পেলে সরাসরি রিটার্ন
        if not docs:
            yield "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই। আরও তথ্যের জন্য অনুগ্রহ করে বিটাকের অফিসিয়াল ওয়েবসাইট https://bitac.gov.bd ভিজিট করুন।"
            return
            
        context_str = "\n\n".join([doc.page_content for doc in docs])

        messages = prompt_template.format_messages(
            context=context_str,
            chat_history=chat_history,
            input=query
        )

        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        print("❌ CRITICAL ERROR IN GENERATOR:")
        traceback.print_exc()
        yield f"\nদুঃখিত, অভ্যন্তরীণ একটি ত্রুটি ঘটেছে। অনুগ্রহ করে আবার চেষ্টা করুন।"

# ================= ৮. চ্যাট এপিআই এন্ডপয়েন্ট (ওয়েবসাইটের জন্য) =================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        return StreamingResponse((add for add in ["অনুগ্রহ করে কিছু লিখুন।"]), media_type="text/plain")

    print(f"💬 Incoming Question From Web: {req.message}")
    
    chat_history = []
    for msg in req.history[-4:]:
        if msg.get("type") == "user":
            chat_history.append(HumanMessage(content=msg.get("text")))
        elif msg.get("type") == "bot":
            chat_history.append(AIMessage(content=msg.get("text")))

    return StreamingResponse(response_generator(req.message, chat_history), media_type="text/plain")

# ================= ⚡ ৯. ফেসবুক ওয়েবহুক ভেরিফিকেশন (GET) =================
@app.get("/webhook")
async def verify_fb_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == FB_VERIFY_TOKEN:
            print("✅ Facebook Webhook Verified Successfully!")
            return PlainTextResponse(content=challenge, status_code=200)
        else:
            raise HTTPException(status_code=403, detail="Verification token mismatch")
    return PlainTextResponse(content="Missing parameters", status_code=400)

# ================= ⚡ ১০. ফেসবুক মেসেজ রিসিভ ও রেসপন্স (POST) =================
@app.post("/webhook")
async def fb_webhook(request: Request):
    body = await request.json()
    
    if body.get("object") == "page":
        for entry in body.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                if messaging_event.get("message") and not messaging_event["message"].get("is_echo"):
                    sender_id = messaging_event["sender"]["id"]
                    user_text = messaging_event["message"].get("text")
                    
                    if user_text:
                        print(f"💬 Facebook Message from {sender_id}: {user_text}")
                        asyncio.create_task(process_and_reply_fb(sender_id, user_text))
                        
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)
    else:
        raise HTTPException(status_code=404)

# ================= ⚡ ১১. ফেসবুক মেসেজ প্রসেসিং ও সেন্ড ফাংশন =================
async def process_and_reply_fb(sender_id: str, user_text: str):
    try:
        if not FB_PAGE_ACCESS_TOKEN:
            print("❌ Error: FB_PAGE_ACCESS_TOKEN is missing!")
            return

        full_answer = ""
        async for chunk in response_generator(user_text, chat_history=[]):
            full_answer += chunk
            
        if not full_answer.strip():
            full_answer = "দুঃখিত, আমি এই মুহূর্তে আপনাকে সাহায্য করতে পারছি না।"

        fb_api_url = f"https://graph.facebook.com/v21.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
        
        payload = {
            "recipient": {"id": sender_id},
            "message": {"text": full_answer.strip()}
        }
        headers = {"Content-Type": "application/json"}
        
        response = await asyncio.to_thread(requests.post, fb_api_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ Failed to send Facebook message: {response.text}")
        else:
            print(f"✅ Reply Sent Successfully to Facebook User {sender_id}!")
            
    except Exception as e:
        print(f"❌ Error in Facebook processing: {str(e)}")

# ================= ১২. ইউজার ইন্টারফেস =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="bn">
<head>
    <title>BITAC AI Chatbot</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { width: 100%; height: 100%; overflow: hidden; background: #f1f5f9; font-family: 'Segoe UI', sans-serif; }
        body { display: flex; justify-content: center; align-items: center; }
        .chatbox { width: 100%; height: calc(var(--vh, 1vh) * 100); display: flex; flex-direction: column; overflow: hidden; background: #ffffff; }
        @media (min-width: 481px) { .chatbox { max-width: 460px; height: 90vh; max-height: 700px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); } }
        @media (max-width: 480px) { .chatbox { width: 100%; height: calc(var(--vh, 1vh) * 100); border-radius: 0; } }
        .header { background: #1e3a8a; color: white; padding: 16px; text-align: center; font-weight: bold; font-size: 16px; flex-shrink: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .messages { flex: 1; padding: 15px; overflow-y: auto; background: #f8fafc; -webkit-overflow-scrolling: touch; }
        .msg { margin: 10px 0; padding: 12px 16px; border-radius: 14px; max-width: 85%; white-space: pre-wrap; font-size: 14px; line-height: 1.5; word-wrap: break-word; }
        .user { background: #2563eb; color: white; margin-left: auto; border-bottom-right-radius: 2px; }
        .bot { background: #e2e8f0; color: #1e293b; margin-right: auto; border-bottom-left-radius: 2px; }
        .input-box { display: flex; border-top: 1px solid #e2e8f0; background: #fff; padding: 12px; flex-shrink: 0; align-items: center; padding-bottom: calc(12px + env(safe-area-inset-bottom)); }
        input { flex: 1; padding: 12px 16px; border: 1px solid #e2e8f0; border-radius: 24px; outline: none; font-size: 16px; background: #f8fafc; transition: all 0.2s; }
        input:focus { border-color: #2563eb; background: #fff; }
        button { margin-left: 8px; padding: 12px 24px; border: none; background: #2563eb; color: white; cursor: pointer; font-weight: bold; font-size: 14px; border-radius: 24px; transition: background 0.2s; flex-shrink: 0; }
        button:hover { background: #1d4ed8; }
    </style>
</head>
<body>
<div class="chatbox">
    <div class="header">BITAC AI Smart Chatbot 🚀</div>
    <div class="messages" id="messages"></div>
    <div class="input-box">
        <input id="input" placeholder="এখানে বাংলায় বা English-এ প্রশ্ন লিখুন..." onkeypress="handleKeyPress(event)" autocomplete="off" />
        <button onclick="send()">Send</button>
    </div>
</div>
<script>
const messages = document.getElementById("messages");
let chatHistory = []; 
function resetHeight() { let vh = window.innerHeight * 0.01; document.documentElement.style.setProperty('--vh', `${vh}px`); }
window.addEventListener('resize', resetHeight); window.addEventListener('orientationchange', resetHeight); resetHeight();
function addMessage(text, type) { let div = document.createElement("div"); div.className = "msg " + type; div.innerText = text; messages.appendChild(div); messages.scrollTop = messages.scrollHeight; return div; }
function handleKeyPress(e) { if (e.key === 'Enter') { send(); } }
async function send() {
    let input = document.getElementById("input"); let text = input.value.trim(); if (!text) return;
    addMessage(text, "user"); input.value = "";
    setTimeout(() => { messages.scrollTop = messages.scrollHeight; }, 50);
    let botMessageDiv = addMessage("✍️ টাইপ করছে...", "bot");
    try {
        let res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ message: text, history: chatHistory })
        });
        const reader = res.body.getReader(); const decoder = new TextDecoder(); let fullAnswer = ""; botMessageDiv.innerText = "";
        while (true) {
            const { value, done } = await reader.read(); if (done) break;
            const chunk = decoder.decode(value, { stream: true }); fullAnswer += chunk;
            botMessageDiv.innerText = fullAnswer; messages.scrollTop = messages.scrollHeight;
        }
        if (fullAnswer.trim()) {
            chatHistory.push({type: "user", text: text}); chatHistory.push({type: "bot", text: fullAnswer.trim()});
            if (chatHistory.length > 8) { chatHistory.splice(0, 2); }
        } else { botMessageDiv.innerText = "দুঃখিত, কোনো উত্তর পাওয়া যায়নি।"; }
    } catch (error) { botMessageDiv.innerText = "সার্ভারের সাথে যোগাযোগ করা যাচ্ছে না।"; }
}
</script>
</body>
</html>
"""
