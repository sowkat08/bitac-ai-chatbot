import os
import hashlib
import time
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
                    # পিডিএফ টেবিল নিখুঁতভাবে রিড করার জন্য pdfplumber ব্যবহার
                    import pdfplumber
                    try:
                        with pdfplumber.open(path) as pdf:
                            page_count = 0
                            for page in pdf.pages:
                                text = page.extract_text() or ""
                                tables = page.extract_tables()
                                
                                # টেবিল পাওয়া গেলে সেটিকে স্ট্রাকচার্ড পাইপ (|) টেক্সটে রূপান্তর
                                if tables:
                                    text += "\n\n--- [Table Data] ---"
                                    for table in tables:
                                        for row in table:
                                            clean_row = [str(cell).strip() for cell in row if cell is not None]
                                            if clean_row:
                                                text += "\n" + " | ".join(clean_row)
                                
                                if text.strip():
                                    docs.append(Document(page_content=text, metadata={"source": path}))
                                
                                page_count += 1
                                if page_count > 50: 
                                    print(f"⚠️ {path} এর প্রথম ৫০ পেজ নেওয়া হয়েছে (সেফটি লিমিট)")
                                    break
                    except Exception as e:
                        print(f"❌ PDF table parsing error ({path}):", e)
                        
                elif f.endswith(".docx"):
                    # ওয়ার্ড ফাইলের সাধারণ প্যারাগ্রাফ ও টেবিল আলাদা করে রিড করা
                    import docx as py_docx
                    try:
                        doc_obj = py_docx.Document(path)
                        full_text = []
                        
                        # প্যারাগ্রাফ রিড করা
                        for para in doc_obj.paragraphs:
                            if para.text.strip():
                                full_text.append(para.text)
                                
                        # টেবিলের ডেটা কলাম-রো অনুযায়ী রিড করা
                        for table in doc_obj.tables:
                            full_text.append("\n--- [Table Data] ---")
                            for row in table.rows:
                                row_data = [cell.text.strip() for cell in row.cells]
                                if any(row_data): 
                                    full_text.append(" | ".join(row_data))
                                    
                        text_content = "\n".join(full_text)
                        if text_content.strip():
                            docs.append(Document(page_content=text_content, metadata={"source": path}))
                    except Exception as e:
                        print(f"❌ Word table parsing error ({path}):", e)
                        
                elif f.endswith(".xlsx"):
                    df = pd.read_excel(path)
                    docs.append(Document(page_content=df.astype(str).to_string(), metadata={"source": path}))
                    
            except Exception as e:
                print(f"❌ General File error ({path}):", e)
else:
    print("⚠️ 'bitac_files' folder not found!")

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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        r = requests.get(url, headers=headers, timeout=4, allow_redirects=True)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            
            for script in soup(["script", "style", "noscript", "header", "footer", "nav", "iframe"]):
                script.decompose()
                
            text = soup.get_text(separator=" ", strip=True)
            if text:
                clean_text = text[:30000]
                docs.append(Document(page_content=clean_text, metadata={"source": url}))
                print(f"✅ Scraping Success: {url}")
        else:
            print(f"⚠️ Skipped (Status: {r.status_code})")
            
    except Exception as e:
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
    
    # 🚨 [রেট লিমিট সমাধান]: ব্যাচ সাইজ ২০ করা হলো এবং প্রতি ব্যাচে ৪ সেকেন্ডের বিরতি দেওয়া হলো
    batch_size = 20  
    vectors = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        current_batch = (i // batch_size) + 1
        print(f"⏳ Processing batch {current_batch} of {total_batches}...")
        
        # এমবেডিং তৈরি করা
        vectors += embeddings.embed_documents(batch_texts)
        
        # ৪ সেকেন্ড বিরতি (Cohere যেন ১ মিনিটের ট্রায়াল লিমিট রিসেট করতে পারে)
        if current_batch < total_batches:
            time.sleep(4)

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
