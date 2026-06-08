import os
import hashlib
import time
import pandas as pd

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from cohere.errors.too_many_requests_error import TooManyRequestsError

# ================= CONFIG =================
INDEX_NAME = os.getenv("INDEX_NAME", "bitac-chatbot")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("Missing API keys in Environment Variables!")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def uid(text, src):
    return hashlib.md5((text + src).encode()).hexdigest()

# ================= SMART CHECKING FUNCTION =================
def is_file_already_ingested(file_path_or_url):
    try:
        # embed-multilingual-v3.0 এর ডাইমেনশন ১০২৪। ডামি ভেক্টর ১ দিয়ে গুণ করে তৈরি করা হলো।
        dummy_vector = [0.1] * 1024
        results = index.query(
            vector=dummy_vector,
            filter={"source": {"$eq": file_path_or_url}},
            top_k=1,
            include_metadata=False
        )
        return len(results.get('matches', [])) > 0
    except Exception as e:
        print(f"⚠️ Pinecone Query Warning for {file_path_or_url}: {e}")
        return False

# ================= TRACK ACTIVE SOURCES =================
# বর্তমানে আপনার bitac_files ফোল্ডারে যে ফাইলগুলো বাস্তবে আছে তাদের একটি ক্লিন লিস্ট তৈরি করা হচ্ছে
current_active_files = set()
if os.path.exists("bitac_files"):
    for root, _, files in os.walk("bitac_files"):
        for f in files:
            full_path = os.path.join(root, f)
            # উইন্ডোজ ও লিনাক্স পাথের স্ল্যাশ সমস্যা দূর করতে পাথ ইউনিফর্ম করা হলো
            normalized_path = full_path.replace("\\\\", "/").replace("\\", "/")
            current_active_files.add(normalized_path)

print(f"📁 বর্তমানে বিটাক ফোল্ডারে মোট একটিভ ফাইল আছে: {len(current_active_files)} টি")

# ================= 🔥 AUTOMATIC SMART SYNC (HARD DELETE) =================
print("\n🔄 ফোল্ডার এবং পাইনকোন ডাটাবেজ নিখুঁতভাবে সিঙ্ক করা হচ্ছে...")

try:
    # পাইনকোনে আগে থেকে ইনজেস্ট করা সমস্ত সোর্সের লিস্ট বের করার ট্রাই-মেকানিজম
    # (আমরা একটি ডামি কুয়েরি মেরে ইউনিক সোর্স মেটাডেটা স্ক্যান করব)
    dummy_vector = [0.1] * 1024
    scan_results = index.query(
        vector=dummy_vector,
        top_k=100,  # আপনার মোট ফাইলের সংখ্যার চেয়ে এই মানটি বড় রাখুন
        include_metadata=True
    )
    
    ingested_sources_in_pinecone = set()
    for match in scan_results.get('matches', []):
        metadata = match.get('metadata', {})
        src = metadata.get('source')
        if src:
            ingested_sources_in_pinecone.add(src.replace("\\\\", "/").replace("\\", "/"))

    # ডিলিট লজিক: পাইনকোনে আছে কিন্তু গিটহাব ফোল্ডারে বর্তমানে নাই — এমন ফাইলগুলো খুঁজে বের করা
    files_to_delete = ingested_sources_in_pinecone - current_active_files
    
    if files_to_delete:
        print(f"🧹 ডাটাবেজে পুরনো/অপ্রয়োজনীয় ফাইল পাওয়া গেছে: {len(files_to_delete)} টি")
        for old_file in files_to_delete:
            # গিটহাব (লিনাক্স) ও লোকাল (উইন্ডোজ) এনভায়রনমেন্ট সেফটি নিশ্চিত করতে দুটি স্ল্যাশ ফরম্যাটেই ডিলিট করা হচ্ছে
            windows_style = old_file.replace("/", "\\")
            linux_style = old_file.replace("\\", "/")
            
            print(f"🗑️ Deleting from Pinecone: {linux_style}")
            index.delete(filter={"source": {"$eq": linux_style}})
            index.delete(filter={"source": {"$eq": windows_style}})
            
        print("✅ ফোল্ডার থেকে ডিলিট হওয়া সমস্ত ফাইলের ডাটাবেজ ক্লিনিং সম্পূর্ণ সফল!")
    else:
        print("✨ ডাটাবেজ একদম ক্লিন! ফোল্ডার থেকে কোনো ফাইল ডিলিট করার প্রয়োজন হয়নি।")

except Exception as e:
    print(f"⚠️ ডাটাবেজ অটো-সিঙ্ক সতর্কতা (প্রথমবার রান বা খালি ইনডেক্সের জন্য এটি স্বাভাবিক): {e}")

# ================= LOAD LOCAL FILES =================
docs = []

if os.path.exists("bitac_files"):
    for root, _, files in os.walk("bitac_files"):
        for f in files:
            path = os.path.join(root, f)
            normalized_path_for_source = path.replace("\\\\", "/").replace("\\", "/")

            # পাইনকোনে অলরেডি এই ফাইলটি থাকলে স্কিপ করবে (যাতে ক্লাউড রিসোর্স সাশ্রয় হয়)
            if is_file_already_ingested(normalized_path_for_source):
                print(f"⏭️  Skipping (Already Ingested): {normalized_path_for_source}")
                continue

            print(f"📖 Reading: {normalized_path_for_source}")
            try:
                if f.endswith(".txt"):
                    from langchain_community.document_loaders import TextLoader
                    # সোর্স পাথ লিনাক্স ফরম্যাটে সেট করে লোড করা
                    loaded_docs = TextLoader(path, encoding='utf-8').load()
                    for d in loaded_docs:
                        d.metadata["source"] = normalized_path_for_source
                    docs += loaded_docs
                    
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
                                    docs.append(Document(page_content=text, metadata={"source": normalized_path_for_source}))
                                
                                page_count += 1
                                if page_count > 50: 
                                    print(f"⚠️ {f} এর প্রথম ৫০ পেজ নেওয়া হয়েছে (সেফটি লিমিট)")
                                    break
                    except Exception as e:
                        print(f"❌ PDF table parsing error ({f}):", e)
                        
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
                            docs.append(Document(page_content=text_content, metadata={"source": normalized_path_for_source}))
                    except Exception as e:
                        print(f"❌ Word table parsing error ({f}):", e)
                        
                elif f.endswith(".xlsx"):
                    df = pd.read_excel(path)
                    docs.append(Document(page_content=df.astype(str).to_string(), metadata={"source": normalized_path_for_source}))
                    
            except Exception as e:
                print(f"❌ General File error ({f}):", e)
else:
    print("⚠️ 'bitac_files' folder not found! দয়া করে রিপোজিটরিতে ফোল্ডারটি তৈরি করুন।")

# ================= SPLIT, EMBED & UPLOAD =================
if not docs:
    print("✅ কোনো নতুন ফাইল প্রসেস করার প্রয়োজন নেই। ডাটাবেজ অলরেডি আপ-টু-ডেট!")
else:
    # চ্যাঙ্ক সাইজ এবং ওভারল্যাপ অপ্টিমাইজড করা হয়েছে নিখুঁত রিট্রিভালের জন্য
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    print(f"🧠 Generating Embeddings & Uploading {len(chunks)} chunks...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=COHERE_API_KEY
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
                print(f"✅ Batch {current_batch} সফলভাবে Pinecone-এ আপলোড হয়েছে।")
                
                time.sleep(2) 
                break 
                
            except TooManyRequestsError:
                print("\n⚠️ Cohere ফ্রি টোকেন লিমিট পার হয়ে গেছে!")
                print("⏳ ৬০ সেকেন্ড অপেক্ষা করছি...")
                time.sleep(60)
                print("🔄 আবার চেষ্টা করা হচ্ছে...\n")
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                raise e

    print(f"\n🎉 সফলভাবে ইনজেস্ট সম্পন্ন হয়েছে! নতুন যুক্ত হওয়া মোট ভেক্টর: {total_uploaded}")
    print("🎯 Pinecone ডাটাবেজ এবং গিটহাব ফোল্ডার এখন সম্পূর্ণ সিঙ্কড ও আপ-টু-ডেট!")
