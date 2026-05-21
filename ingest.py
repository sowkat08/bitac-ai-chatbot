import os
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY or COHERE_API_KEY is missing!")

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

print("🧠 Generating Cohere Multilingual Embeddings...")
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)

print(f"🚀 Pushing {len(chunks)} chunks to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ SUCCESS: Data Ingested into Pinecone using Cohere!")
