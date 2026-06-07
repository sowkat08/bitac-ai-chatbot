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

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def uid(text, src):
    return hashlib.md5((text + src).encode()).hexdigest()

# ================= SMART CHECKING FUNCTION =================
def is_file_already_ingested(file_path_or_url):
    try:
        results = index.query(
            vector=[0.0] * 1024,
            filter={"source": {"$eq": file_path_or_url}},
            top_k=1,
            include_metadata=False
        )
        return len(results.get('matches', [])) > 0
    except Exception as e:
        print(f"⚠️ Pinecone Query Warning: {e}")
        return False

# ================= LOAD LOCAL FILES =================
docs = []

if os.path.exists("bitac_files"):
    for root, _, files in os.walk("bitac_files"):
        for f in files:
            path = os.path.join(root, f)

            if is_file_already_ingested(path):
                print(f"⏭️  Skipping: {path}")
                continue

            print(f"📖 Reading: {path}")
            try:
                if f.endswith(".txt"):
                    from langchain_community.document_loaders import TextLoader
                    docs += TextLoader(path, encoding='utf-8').load()
                elif f.endswith(".pdf"):
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(path)
                    # মেমোরি ক্র্যাশ এড়াতে সর্বোচ্চ ৫০ পেজ পর্যন্ত রিড করার সেফটি লিমিট
                    page_count = 0
                    for page in loader.lazy_load():
                        docs.append(page)
                        page_count += 1
                        if page_count > 50: 
                            print(f"⚠️ {path} এর প্রথম ৫০ পেজ নেওয়া হয়েছে (সেফটি লিমিট)")
                            break
                elif f.endswith(".docx"):
                    from langchain_community.document_loaders import Docx2txtLoader
                    docs += Docx2txtLoader(path).load()
                elif f.endswith(".xlsx"):
                    df = pd.read_excel(path)
                    docs.append(Document(page_content=df.astype(str).to_string(), metadata={"source": path}))
            except Exception as e:
                print(f"❌ File error ({path}):", e)

# ================= WEB SCRAPING =================
urls = [
    "https://bitac.gov.bd/",
    "https://bitac.dhaka.gov.bd/",
    "https://bitac.gov.bd/pages/officers"
]

for url in urls:
    if is_file_already_ingested(url):
        print(f"⏭️  Skipping URL: {url}")
        continue

    print(f"🌐 Scraping URL: {url}")
    try:
        # রিকোয়েস্ট হেডারে ব্রাউজার ট্রিকস
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        # [🚨 ব্রেকিং সেফটি ফিক্স]: রিকোয়েস্টে রিডাইরেকশন এবং কড়া টাইমআউট (৪ সেকেন্ড) দেওয়া হলো
        r = requests.get(url, headers=headers, timeout=4, allow_redirects=True)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            
            # অপ্রয়োজনীয় এলিমেন্ট ডিকম্পোজ
            for script in soup(["script", "style", "noscript", "header", "footer", "nav", "iframe"]):
                script.decompose()
                
            text = soup.get_text(separator=" ", strip=True)
            # টেক্সট যেন খুব বেশি বড় হয়ে কন্টেইনার হ্যাং না করে (সর্বোচ্চ ৩০,০০০ ক্যারেক্টার সেফটি)
            if text:
                clean_text = text[:30000]
                docs.append(Document(page_content=clean_text, metadata={"source": url}))
                print(f"✅ Scraping Success: {url}")
        else:
            print(f"⚠️ Skipped (Status: {r.status_code})")
            
    except Exception as e:
        # সাইট ডাউন বা স্লো থাকলে স্ক্রিপ্ট না থামিয়ে সরাসরি পরের ইউআরএল-এ চলে যাবে
        print(f"⏭️ URL Skipped due to network/timeout: {url}")

# ================= SPLIT & UPLOAD =================
if not docs:
    print("✅ কোনো নতুন ডেটা নেই। ডাটাবেজ অলরেডি আপ-টু-ডেট!")
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    print(f"🧠 Generating Embeddings for {len(chunks)} chunks...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    texts = [c.page_content for c in chunks]
    
    # Cohere এর রেট লিমিট ও গিটহাব ক্র্যাশ এড়াতে ৫০টি করে ব্যাচ প্রসেস
    batch_size = 50
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        vectors += embeddings.embed_documents(batch_texts)

    upserts = []
    for i in range(len(texts)):
        src_metadata = chunks[i].metadata.get("source", "file")
        vid = uid(texts[i], src_metadata)

        upserts.append((
            vid,
            vectors[i],
            {
                "text": texts[i],
                "source": src_metadata
            }
        ))

    print(f"🚀 Uploading to Pinecone...")
    index.upsert(vectors=upserts)
    print(f"✅ সফলভাবে ইনজেস্ট সম্পন্ন হয়েছে! মোট ভেক্টর: {len(upserts)}")
