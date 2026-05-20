import os
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Get API Key from GitHub Secrets
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing in GitHub Secrets!")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Safe Cleanup: ডুপ্লিকেট এড়াতে আগের ডেটা ট্রাই-ক্যাচ দিয়ে মোছা
try:
    print("🧹 Cleaning old data from Pinecone...")
    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True)
except Exception as e:
    print(f"ℹ️ Index empty or notice: {e}")

# ফাইল লোড করা
print("📁 Loading documents from bitac_files...")
try:
    loader = DirectoryLoader("bitac_files", glob="**/*")
    docs = loader.load()
except Exception as e:
    raise RuntimeError(f"❌ Cannot read bitac_files folder! Error: {e}")

if not docs:
    raise ValueError("❌ No files found inside bitac_files folder!")

# টেক্সট স্প্লিট করা (নতুন ল্যাংচেইন স্ট্যান্ডার্ড অনুযায়ী)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# এমবেডিং তৈরি
print("🧠 Generating Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# পাইনকোনে ডেটা পুশ করা
print(f"🚀 Pushing {len(chunks)} chunks to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ SUCCESS: Data Ingested into Pinecone!")
