import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ১. FastAPI অ্যাপ ইনিশিয়ালাইজ করা
app = FastAPI(title="BITAC Smart Chatbot")

# ২. এনভায়রনমেন্ট ভেরিয়েবল
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("ERROR: PINECONE_API_KEY or GOOGLE_API_KEY is missing!")

# ৩. পাইনকোন ও এমবেডিং সেটআপ
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ৪. গুগল জেমিনি মডেল (LLM) সেটআপ
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)

# ৫. প্রম্পট ও চেইন সেটআপ
system_prompt = (
    "You are an advanced AI assistant for BITAC (Bangladesh Industrial Technical Assistance Center).\n"
    "Use the following pieces of retrieved context to answer the question in detail.\n"
    "If someone asks in Bengali, reply beautifully in Bengali using the context.\n"
    "If you don't know the answer or if it's not in the context, say that politely.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ডাটা মডেল
class ChatRequest(BaseModel):
    message: str

# 🎨 ৬. সুন্দর ভিজ্যুয়াল চ্যাট ইন্টারফেস (HTML + CSS + JS)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BITAC AI Tech-Bot</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body { background: #f3f4f6; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .chat-container { width: 100%; max-width: 500px; height: 80vh; background: #ffffff; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: #1e3a8a; color: white; padding: 20px; display: flex; align-items: center; gap: 12px; }
        .chat-header h3 { font-size: 18px; font-weight: 600; }
        .chat-header p { font-size: 12px; color: #93c5fd; }
        .chat-messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; background: #fafafa; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.5; word-wrap: break-word; }
        .message.user { background: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
        .message.bot { background: #e5e7eb; color: #1f2937; align-self: flex-start; border-bottom-left-radius: 2px; }
        .chat-input-area { padding: 15px; background: white; border-top: 1px solid #e5e7eb; display: flex; gap: 10px; }
        .chat-input-area input { flex: 1; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; outline: none; font-size: 14px; transition: border 0.2s; }
        .chat-input-area input:focus { border-color: #2563eb; }
        .chat-input-area button { background: #2563eb; color: white; border: none; padding: 0 20px; border-radius: 8px; font-weight: 500; cursor: pointer; transition: background 0.2s; }
        .chat-input-area button:hover { background: #1d4ed8; }
        .loading { font-style: italic; color: #6b7280; }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="chat-header">
        <div>
            <h3>BITAC Tech-Bot 🚀</h3>
            <p>Online | Powered by Gemini & Pinecone</p>
        </div>
    </div>
    
    <div class="chat-messages" id="chat-box">
        <div class="message bot">হ্যালো! আমি বিটাক এআই চ্যাটবট। আপনাকে কীভাবে সাহায্য করতে পারি?</div>
    </div>
    
    <div class="chat-input-area">
        <input type="text" id="user-input" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');

    function appendMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        msgDiv.innerText = text;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function handleKeyPress(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    }

    async class sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage(text, 'user');
        userInput.value = '';

        // লোডিং মেসেজ দেখানো
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'bot', 'loading');
        loadingDiv.innerText = 'Thinking...';
        chatBox.appendChild(loadingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await response.json();
            
            // লোডিং সরানো এবং বটের আসল উত্তর বসানো
            chatBox.removeChild(loadingDiv);
            if (data.status === 'Success') {
                appendMessage(data.chatbot_response, 'bot');
            } else {
                appendMessage('দুঃখিত, কোনো সমস্যা হয়েছে। আবার চেষ্টা করুন।', 'bot');
            }
        } catch (error) {
            chatBox.removeChild(loadingDiv);
            appendMessage('সার্ভারে কানেক্ট করা যাচ্ছে না।', 'bot');
        }
    }
</script>

</body>
</html>
"""

# ৭. রুট এন্ডপয়েন্ট যা সরাসরি ভিজ্যুয়াল স্ক্রিন লোড করবে
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_TEMPLATE

# ৮. ব্যাকএন্ড চ্যাট এপিআই এন্ডপয়েন্ট
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = rag_chain.invoke({"input": request.message})
        return {
            "chatbot_response": response["answer"],
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
