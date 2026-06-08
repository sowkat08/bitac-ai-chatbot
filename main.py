import os
import asyncio
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ================= ১. অ্যাপ সেটআপ ও CORS =================
app = FastAPI(title="BITAC AI Smart Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ২. এনভায়রনমেন্ট ভেরিয়েবল =================
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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4} # k=4 ফাস্ট এবং নিখুঁত কনটেক্সটের জন্য পারফেক্ট
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

# ================= ७. রেসপন্স জেনারেটর (Optimized Speed & Stream) =================
async def response_generator(query: str, chat_history: list):
    try:
        optimized_query = query
        
        # ১ শব্দের বা শর্ট কুয়েরির জন্য স্মার্ট অপ্টিমাইজেশন (Fast Exec)
        if len(chat_history) > 0 and len(query.split()) <= 3:
            last_msg = chat_history[-1].content if hasattr(chat_history[-1], 'content') else str(chat_history[-1])
            optimization_prompt = f"Task: Combine last response and current short query to make 2-3 Bangla search keywords.\nLast AI Response: {last_msg}\nUser Short Query: {query}\nKeywords only:"
            try:
                # খুব দ্রুত রেসপন্সের জন্য ১ সেকেন্ড টাইমআউট বা শর্ট কল
                opt_response = await llm.ainvoke(optimization_prompt)
                if opt_response and opt_response.content:
                    optimized_query = opt_response.content.strip()
            except Exception:
                optimized_query = query

        # পাইনকোন থেকে কনটেক্সট রিট্রিভ (Async Threading)
        docs = await asyncio.to_thread(retriever.invoke, optimized_query)
        
        if not docs:
            docs = await asyncio.to_thread(retriever.invoke, query)
            
        context_str = "\n\n".join([doc.page_content for doc in docs]) if docs else "No context available."

        # প্রম্পট ফরম্যাট করা
        messages = prompt_template.format_messages(
            context=context_str,
            chat_history=chat_history,
            input=query
        )

        # ⚡ [স্পীড বুস্ট ফিক্স]: সরাসরি LLM স্ট্রিম ব্যবহার করা হয়েছে যাতে কোনো চঙ্ক মিস না হয়
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        print("❌ CRITICAL ERROR IN GENERATOR:")
        traceback.print_exc()
        yield f"\nদুঃখিত, অভ্যন্তরীণ একটি ত্রুটি ঘটেছে। অনুগ্রহ করে আবার চেষ্টা করুন।"

# ================= ৮. চ্যাট এপিআই এন্ডপয়েন্ট =================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        return StreamingResponse((add for add in ["অনুগ্রহ করে কিছু লিখুন।"]), media_type="text/plain")

    print(f"💬 Incoming Question: {req.message}")
    
    chat_history = []
    # শেষ ৪টি মেসেজ হিস্ট্রি হিসেবে নেওয়া হচ্ছে (মেমোরি ঠিক রাখার জন্য যথেষ্ট)
    for msg in req.history[-4:]:
        if msg.get("type") == "user":
            chat_history.append(HumanMessage(content=msg.get("text")))
        elif msg.get("type") == "bot":
            chat_history.append(AIMessage(content=msg.get("text")))

    return StreamingResponse(response_generator(req.message, chat_history), media_type="text/plain")

# ================= ৯. ইউজার ইন্টারফেস =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>BITAC AI Chatbot</title>
    <meta charset="utf-8">
    <style>
        body { margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .chatbox { width: 450px; height: 650px; background: white; border-radius: 15px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .header { background: #1e3a8a; color: white; padding: 15px; text-align: center; font-weight: bold; font-size: 16px; }
        .messages { flex: 1; padding: 15px; overflow-y: auto; background: #f8fafc; }
        .msg { margin: 10px 0; padding: 12px; border-radius: 10px; max-width: 85%; white-space: pre-wrap; font-size: 14px; line-height: 1.5; word-wrap: break-word; }
        .user { background: #2563eb; color: white; margin-left: auto; border-bottom-right-radius: 2px; }
        .bot { background: #e2e8f0; color: #1e293b; margin-right: auto; border-bottom-left-radius: 2px; }
        .input-box { display: flex; border-top: 1px solid #e2e8f0; background: #fff; }
        input { flex: 1; padding: 15px; border: none; outline: none; font-size: 14px; }
        button { padding: 0 20px; border: none; background: #2563eb; color: white; cursor: pointer; font-weight: bold; font-size: 14px; }
        button:hover { background: #1d4ed8; }
    </style>
</head>
<body>

<div class="chatbox">
    <div class="header">BITAC AI Smart Chatbot 🚀</div>
    <div class="messages" id="messages"></div>
    <div class="input-box">
        <input id="input" placeholder="এখানে বাংলায় বা English-এ প্রশ্ন লিখুন..." onkeypress="handleKeyPress(event)" autocomplete="off" />
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

    let botMessageDiv = addMessage("✍️ টাইপ করছে...", "bot");
    
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
        botMessageDiv.innerText = ""; // Clear loader text

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
            
            if (chatHistory.length > 8) {
                chatHistory.splice(0, 2); // মেমোরি ক্লিন রাখা
            }
        } else {
            botMessageDiv.innerText = "দুঃখিত, কোনো উত্তর পাওয়া যায়নি।";
        }
    } catch (error) {
        botMessageDiv.innerText = "সার্ভারের সাথে যোগাযোগ করা যাচ্ছে না।";
    }
}
</script>

</body>
</html>
"""
