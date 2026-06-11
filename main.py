import os
import asyncio
import traceback
import requests  # ⚡ ফেসবুক API-তে রিকোয়েস্ট পাঠানোর জন্য
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

# ⚡ ফেসবুকের সিক্রেট ভেরিয়েবল (এগুলো Render-এর Environment Variables-এ সেট করে দেবেন)
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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ================= ৪. লার্জ ল্যাঙ্গুয়েজ মডেল (LLM) =================
llm = ChatCohere(
    model="command-r-08-2024", 
    cohere_api_key=COHERE_API_KEY,
    temperature=0.1,
    streaming=True
)

# ================= ৫. প্রম্পট সেটআপ =================
system_prompt = """তুমি হলে BITAC (বিটাক)-এর অফিসিয়াল এআই অ্যাসিস্ট্যান্ট। 

নিয়মাবলী (কঠোরভাবে পালনীয়):
১. নিচে দেওয়া "Context"-এর ভেতরের অফিসিয়াল তথ্যের ওপর ভিত্তি করে ইউজারের প্রশ্নের সরাসরি উত্তর দাও। নিজের থেকে কোনো তথ্য অনুমান বা আবিষ্কার করবে না।
২. যদি প্রশ্নের উত্তর Context-এ সরাসরি না থাকে, কিন্তু প্রাসঙ্গিক (Related) কোনো তথ্য থাকে, তবে সেই প্রাসঙ্গিক তথ্যটি ব্যবহার করে সুন্দর করে বুঝিয়ে বলো।
৩. যদি কোনো তথ্যই বা প্রাসঙ্গিক কোনো লাইন Context-এ না থাকে, তবে বানিয়ে কিছু বলবে না। সরাসরি বলবে: "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই।"
৪. ইউজার যদি আঞ্চলিক ভাষায় বা বাংলিশে (Banglish) প্রশ্ন করে, তবে তার মূল উদ্দেশ্য বুঝে Context-এর সাথে মিলিয়ে যুক্তিসঙ্গত উত্তর দাও।
৫. উত্তর সবসময় স্পষ্ট, সহজ এবং প্রাঞ্জল বাংলায় হতে হবে।

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
        optimized_query = query
        
        # জেনারেটরে চ্যাট হিস্ট্রি খালি না থাকলে কেবল অপটিমাইজেশন রান হবে (IndexError ফিক্সড)
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
        if not docs:
            docs = await asyncio.to_thread(retriever.invoke, query)
            
        context_str = "\n\n".join([doc.page_content for doc in docs]) if docs else "No context available."

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
            # ফেসবুক প্লেইন টেক্সট বা ইনটিজার চ্যালেঞ্জ আশা করে (ফরম্যাট ফিক্সড)
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
                # কেউ মেসেজ পাঠালে এবং সেটি নিজের ইকো (Echo) না হলে
                if messaging_event.get("message") and not messaging_event["message"].get("is_echo"):
                    sender_id = messaging_event["sender"]["id"]
                    user_text = messaging_event["message"].get("text")
                    
                    if user_text:
                        print(f"💬 Facebook Message from {sender_id}: {user_text}")
                        # ফেসবুককে দ্রুত '200 OK' রেসপন্স ব্যাক করে ব্যাকগ্রাউন্ডে প্রসেস করা হচ্ছে
                        asyncio.create_task(process_and_reply_fb(sender_id, user_text))
                        
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)
    else:
        raise HTTPException(status_code=404)

# ================= ⚡ ১১. ফেসবুক মেসেজ প্রসেসিং ও সেন্ড ফাংশন =================
async def process_and_reply_fb(sender_id: str, user_text: str):
    try:
        if not FB_PAGE_ACCESS_TOKEN:
            print("❌ Error: FB_PAGE_ACCESS_TOKEN is missing in Environment Variables!")
            return

        # ১. আপনার রেসপন্স জেনারেটর থেকে উত্তর তৈরি করা
        full_answer = ""
        async for chunk in response_generator(user_text, chat_history=[]):
            full_answer += chunk
            
        if not full_answer.strip():
            full_answer = "দুঃখিত, আমি এই মুহূর্তে আপনাকে সাহায্য করতে পারছি না।"

        # ২. ফেসবুক Graph API-এর মাধ্যমে মেসেঞ্জারে উত্তর পাঠানো
        fb_api_url = f"https://graph.facebook.com/v21.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
        
        payload = {
            "recipient": {"id": sender_id},
            "message": {"text": full_answer.strip()}
        }
        headers = {"Content-Type": "application/json"}
        
        # requests.post-কে অ্যাসিনক্রোনাসলি থ্রেডে রান করা
        response = await asyncio.to_thread(requests.post, fb_api_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ Failed to send Facebook message: {response.text}")
        else:
            print(f"✅ Reply Sent Successfully to Facebook User {sender_id}!")
            
    except Exception as e:
        print(f"❌ Error in Facebook processing: {str(e)}")


# ================= ১২. ইউজার ইন্টারফেস (মোবাইল ফ্রেন্ডলি ও ১০০% রেসপন্সিভ) =================
@app.get("/", response_class=HTMLResponse)
def home():
    # আপনার এক্সিস্টিং HTML ইন্টারফেসের কোড এখানে অপরিবর্তিত থাকবে
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
