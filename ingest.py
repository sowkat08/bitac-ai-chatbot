import os
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY or GOOGLE_API_KEY is missing!")

pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    print("🧹 Cleaning old data from Pinecone...")
    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True)
except Exception as e:
    print(f"ℹ️ Index notice: {e}")

print("📁 Loading documents from bitac_files...")
loader = DirectoryLoader("bitac_files", glob="**/*")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print("🧠 Generating Google Embeddings...")
# 💡 এখানে মডেলের নাম একদম সঠিক "text-embedding-004" করে দেওয়া হয়েছে
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)

print(f"🚀 Pushing {len(chunks)} chunks to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ SUCCESS: Data Ingested into Pinecone!")
