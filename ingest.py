import os
import hashlib
import pandas as pd
import requests
from bs4 import BeautifulSoup

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings

# ================= CONFIG =================
INDEX_NAME = "bitac-chatbot"

# পাইনকোন কানেকশন সেটআপ
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def uid(text, src):
    """ভেক্টরের জন্য ইউনিক আইডি তৈরি করার ফাংশন"""
    return hashlib.md5((text + src).encode()).hexdigest()

# ================= SMART CHECKING FUNCTION =================
def is_file_already_ingested(file_path_or_url):
    """পাইনকোন ডাটাবেজ থেকে লাইভ চেক করবে এই ফাইল বা ইউআরএল আগে আপলোড হয়েছে কি না"""
    try:
        # ফাইলের সোর্স/নাম দিয়ে পাইনকোনে মেটাডাটা ফিল্টার কুয়েরি করা হচ্ছে
        results = index.query(
            vector=[0.0] * 1024,  # Cohere Multilingual v3 এর ডাইমেনশন ১০২৪
            filter={"source": {"$eq": file_path_or_url}},
            top_k=1,
            include_metadata=False
        )
        return len(results.get('matches', [])) > 0
    except Exception:
        return False

# ================= LOAD LOCAL FILES =================
docs = []

# bitac_files ফোল্ডারের ভেতর থাকা ৫০০+ ফাইল রিড করার লুপ
for root, _, files in os.walk("bitac_files"):
    for f in files:
        path = os.path.join(root, f)

        # [স্মার্ট ফিল্টার]: ফাইলটি অলরেডি পাইনকোনে থাকলে এটি স্কিপ (Skip) করবে
        if is_file_already_ingested(path):
            print(f"⏭️  Skipping (Already Ingested): {path}")
            continue

        print(f"📖 Reading New File: {path}")
        try:
            if f.endswith(".txt"):
                from langchain_community.document_loaders import TextLoader
                docs += TextLoader(path).load()
            elif f.endswith(".pdf"):
                from langchain_community.document_loaders import PyPDFLoader
                docs += PyPDFLoader(path).load()
            elif f.endswith(".docx"):
                from langchain_community.document_loaders import Docx2txtLoader
                docs += Docx2txtLoader(path).load()
            elif f.endswith(".xlsx"):
                df = pd.read_excel(path)
                # এক্সেল ফাইলকে ক্লিন টেক্সট হিসেবে নেওয়া
                docs.append(Document(page_content=df.astype(str).to_string(), metadata={"source": path}))
        except Exception as e:
            print(f"❌ File error ({path}):", e)

# ================= WEB SCRAPING =================
# আপনার রিকোয়েস্টেড মূল ওয়েবসাইটসহ ৩টি লিংক এখানে দেওয়া হলো
urls = [
    "https://bitac.gov.bd/",
    "https://bitac.dhaka.gov.bd/",
    "https://bitac.gov.bd/pages/officers"
]

for url in urls:
    if is_file_already_ingested(url):
        print(f"⏭️  Skipping (Already Ingested URL): {url}")
        continue

    print(f"🌐 Scraping New URL: {url}")
    try:
        # সরকারি সার্ভারে ব্লকিং এড়াতে ব্রাউজার হেডার ব্যবহার
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        
        # [স্মার্ট ক্লিনিং]: অপ্রয়োজনীয় জাভাস্ক্রিপ্ট কোড, স্টাইল, হেডার ও ফুটার মুছে ফেলা হচ্ছে
        for script in soup(["script", "style", "noscript", "header", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator=" ", strip=True)
        docs.append(Document(page_content=text, metadata={"source": url}))
        
    except Exception as e:
        print(f"❌ Web error ({url}):", e)

# ================= SPLIT & UPLOAD =================
if not docs:
    print("✅ কোনো নতুন ফাইল বা লিংক পাওয়া যায়নি। ডাটাবেজ সম্পূর্ণ আপ-টু-ডেট!")
else:
    # বড় টেক্সটকে ছোট টুকরো করা
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    print("🧠 Generating Embeddings with Cohere...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    # পাইনকোনে পাঠানোর জন্য ডেটা ফরম্যাট করা
    upserts = []
    for i in range(len(texts)):
        src_metadata = chunks[i].metadata.get("source", "file")
        vid = uid(texts[i], src_metadata)

        upserts.append((
            vid,
            vectors[i],
            {
                "text": texts[i],
                "source": src_metadata  # এই সোর্স মেটাডাটা দেখেই কোড পরে চেক করবে
            }
        ))

    # পাইনকোনে ফাইনাল ডেটা পুশ করা
    print(f"🚀 Uploading {len(upserts)} vectors to Pinecone...")
    index.upsert(vectors=upserts)
    print(f"✅ সফলভাবে নতুন {len(docs)} টি সোর্স থেকে ডেটা আপলোড সম্পন্ন হয়েছে!")
