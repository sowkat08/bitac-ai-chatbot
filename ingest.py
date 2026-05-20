import os
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Get API Key from GitHub Secrets
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "bitac-chatbot"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ডুপ্লিকেট ডেটা এড়াতে আগের ডেটা মুছে ফেলা
print("🧹 Cleaning old data from Pinecone...")
index = pc.Index(INDEX_NAME)
index.delete(delete_all=True)

# ফাইল লোড করা
print("📁 Loading documents from bitac_files...")
loader = DirectoryLoader("bitac_files", glob="**/*")
docs = loader.load()

# টেক্সট ছোট ছোট টুকরো করা
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# এমবেডিং তৈরি
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# পাইনকোনে ডেটা পুশ করা
print(f"🚀 Pushing {len(chunks)} chunks to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ SUCCESS: Data Ingested into Pinecone!")
