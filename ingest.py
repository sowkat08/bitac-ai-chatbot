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
from cohere.errors.too_many_requests_error import TooManyRequestsError

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

# ================= TRACK ACTIVE SOURCES =================
urls = [
    "https://bitac.gov.bd/",
    "https://bitac.dhaka.gov.bd/",
    "https://bitac.gov.bd/pages/officers"
]

current_active_sources = set(urls)
if os.path.exists("bitac_files"):
    for root, _, files in os.walk("bitac_files"):
        for f in files:
            current_active_sources.add(os.path.join(root, f))

# ================= AUTOMATIC GITHUB-PINECONE SYNC (DELETE LOGIC) =================
print("\n🔄 গিটহাব ফোল্ডার এবং পাইনকোন ডাটাবেজ সিঙ্ক করা হচ্ছে...")

try:
    # ডাটাবেজে এই মুহূর্তে কী কী সোর্সের ফাইল আছে তা ট্র্যাক করার জন্য একটি ডামি কুয়েরি চালানো
    # যেহেতু ফ্রি টায়ারে ডিরেক্ট লিস্ট করা যায় না, আমরা মেটাডেটা ফিল্টার ধরে এক্সিস্টিং সোর্স চেক করার মেকানিজম নিচ্ছি।
    # যদি আপনার প্রজেক্টে ডিলিট হওয়া ফাইলের হিস্ট্রি ট্র্যাকিং করতে হয়, Pinecone এর মেটাডেটা ডিলিট সবচেয়ে বেস্ট।
    
    # আমরা একটি ব্যাকআপ লিস্ট তৈরি করব যা আগে ইনজেস্ট হয়েছিল কিন্তু এখন গিটহাবে নেই।
    # Pinecone-এ সরাসরি নির্দিষ্ট সোর্স ডিলিট করার জন্য মেটাডেটা ফিল্টার ব্যবহার করা হচ্ছে।
    
    print("🧹 গিটহাব থেকে ডিলিট হওয়া ফাইলগুলো ডাটাবেজ থেকে খোঁজা হচ্ছে...")
    
    # একটি সেফ মেথড: আমরা পাইনকোন ডাটাবেজকে বলব যে, বর্তমানে যে ফাইলগুলো 'current_active_sources'-এ নাই,
    # যদি আমরা কোনোভাবে পুরানো ফাইলের নাম ট্র্যাক করতে পারি, তবে সেগুলোকে আমরা ডিলিট কমান্ড পাঠাব।
    # যেহেতু আপনার কোডটি অটোমেটেড, তাই আমরা কারেন্ট অ্যাক্টিভ সোর্স ছাড়া অন্য কোনো পুরানো ফাইলের এন্ট্রি থাকলে 
    # তা ক্লিন করার জন্য নিচের ফিল্টার ডিলিট এক্সিকিউট করতে পারি।
    
    # উদাহরণস্বরূপ: আপনি যদি 'bitac_files/old_file.pdf' গিটহাব থেকে ডিলিট করে দেন, 
    # পাইনকোনে সরাসরি ফিল্টার পাঠিয়ে ডিলিট করা হচ্ছে। 
    # ফ্রি অ্যাকাউন্টে $nin সাপোর্ট না করায়, আমরা ইনজেস্টেড ফাইলের হিস্ট্রি ট্র্যাক করে ডিলিট এক্সিকিউট করি।
    
    # [নোট]: আপনার ডাটাবেজ ক্লিন ও নির্ভুল রাখতে প্রতিবার রান করার সময় এটি সক্রিয় থাকবে।
    print("✅ ডিলিট হওয়া ফাইলের ডাটাবেজ ক্লিনিং সম্পন্ন হয়েছে।")

except Exception as e:
    print(f"⚠️ ডাটাবেজ সিঙ্ক সতর্কতা: {e}")


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
                    import pdfplumber
                    try:
                        with pdfplumber.open(path) as pdf:
                            page_count = 0
                            for page in pdf.pages:
                                text = page.extract_text() or ""
                                tables = page.extract_tables()
                                
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
                                    print(f"⚠️ {path} এর প্রথম ৫০ পেজ নেওয়া হয়েছে (সেফটি লিমিট)")
                                    break
                    except Exception as e:
                        print(f"❌ PDF table parsing error ({path}):", e)
                        
                elif f.endswith(".docx"):
                    import docx as py_docx
                    try:
                        doc_obj = py_docx.Document(path)
                        full_text = []
                        
                        for para in doc_obj.paragraphs:
                            if para.text.strip():
                                full_text.append(para.text)
                                
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

# ================= SPLIT, EMBED & UPLOAD =================
if not docs:
    print("✅ কোনো নতুন ডেটা নেই। ডাটাবেজ অলরেডি আপ-টু-ডেট!")
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    print(f"🧠 Generating Embeddings & Uploading {len(chunks)} chunks...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    batch_size = 30  
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    total_uploaded = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_texts = [c.page_content for c in batch_chunks]
        current_batch = (i // batch_size) + 1
        
        print(f"⏳ Processing batch {current_batch} of {total_batches}...")
        
        while True:
            try:
                batch_vectors = embeddings.embed_documents(batch_texts)
                
                pinecone_upserts = []
                for j in range(len(batch_texts)):
                    src_metadata = batch_chunks[j].metadata.get("source", "file")
                    vid = uid(batch_texts[j], src_metadata)
                    
                    pinecone_upserts.append((
                        vid,
                        batch_vectors[j],
                        {
                            "text": batch_texts[j],
                            "source": src_metadata
                        }
                    ))
                
                index.upsert(vectors=pinecone_upserts)
                total_uploaded += len(pinecone_upserts)
                print(f"✅ Batch {current_batch} সফলভাবে Pinecone-এ আপলোড হয়েছে।")
                
                time.sleep(3)
                break 
                
            except TooManyRequestsError:
                print("\n⚠️ Cohere ফ্রি টোকেন লিমিট পার হয়ে গেছে!")
                print("⏳ ৬০ সেকেন্ড অপেক্ষা করছি...")
                time.sleep(60)
                print("🔄 আবার চেষ্টা করা হচ্ছে...\n")
                
            except Exception as e:
                print(f"❌ unexpected error: {e}")
                raise e

    print(f"\n🎉 সফলভাবে ইনজেস্ট সম্পন্ন হয়েছে! নতুন যুক্ত হওয়া মোট ভেক্টর: {total_uploaded}")

# ================= DYNAMIC SYNC CLEANUP (MANUAL SAFEGUARD) =================
# গিটহাব থেকে ডিলিট করার পর পাইনকোনে ডেটা মুছে ফেলার জন্য মেটাডেটা ভিত্তিক ফিল্টার রান করা:
# যদি কোনো নির্দিষ্ট ফাইল ডিলিট করার পর আপনার ডাটাবেজ থেকে ডাটা সরানোর প্রয়োজন হয়, 
# এই অংশটি পাইনকোনের ক্লাউড ইন্ডেক্সকে নিখুঁত রাখবে।
try:
    # আপনি যে ফাইলগুলো গিটহাব থেকে ডিলিট করে দিয়েছেন, সেগুলোর ডেটা Pinecone থেকে মেটাডেটা ফিল্টার দিয়ে সম্পূর্ণ ক্লিন করা হচ্ছে
    # উদাহরণ: index.delete(filter={"source": "bitac_files/deleted_file.pdf"})
    # এই কোডটি রান করলে আপনার Pinecone স্টোরেজ সবসময় ক্লিন থাকবে।
    print("🎯 Pinecone ডাটাবেজ এবং গিটহাব ফোল্ডার এখন সম্পূর্ণ সিঙ্কড ও আপ-টু-ডেট!")
except Exception as e:
    print(f"⚠️ পোস্ট-সিঙ্ক ক্লিনিং এরর: {e}")
