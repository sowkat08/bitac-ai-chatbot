import os
import uuid
import requests
import pandas as pd

from bs4 import BeautifulSoup
from pinecone import Pinecone

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings

# =====================
# CONFIG
# =====================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = "bitac-chatbot"

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("❌ Missing API keys")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# =====================
# CLEAN OLD DATA
# =====================

try:
    index.delete(delete_all=True, namespace="")
    print("🧹 Old data deleted")
except Exception as e:
    print("ℹ️ Delete skipped:", e)

# =====================
# TXT FILES
# =====================

txt_docs = DirectoryLoader(
    "bitac_files",
    glob="**/*.txt",
    loader_cls=TextLoader
).load()

# =====================
# PDF FILES
# =====================

pdf_docs = []
for root, _, files in os.walk("bitac_files"):
    for f in files:
        if f.endswith(".pdf"):
            pdf_docs.extend(PyPDFLoader(os.path.join(root, f)).load())

# =====================
# DOCX FILES
# =====================

docx_docs = []
for root, _, files in os.walk("bitac_files"):
    for f in files:
        if f.endswith(".docx"):
            docx_docs.extend(Docx2txtLoader(os.path.join(root, f)).load())

# =====================
# EXCEL FILES
# =====================

excel_docs = []
for root, _, files in os.walk("bitac_files"):
    for f in files:
        if f.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(root, f))
            text = df.astype(str).to_string()
            excel_docs.append(Document(page_content=text))

# =====================
# WEBSITE SCRAPER
# =====================

urls = [
    "https://example.com",
    "https://en.wikipedia.org/wiki/Artificial_intelligence"
]

web_docs = []

for url in urls:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # cleaner text (IMPORTANT FIX)
        text = " ".join(soup.stripped_strings)

        web_docs.append(
            Document(
                page_content=text,
                metadata={"source": url}
            )
        )

        print(f"🌐 Loaded: {url}")

    except Exception as e:
        print(f"❌ Failed: {url} -> {e}")

# =====================
# MERGE ALL DOCS
# =====================

docs = txt_docs + pdf_docs + docx_docs + excel_docs + web_docs

print(f"📄 Total docs: {len(docs)}")

# =====================
# SPLIT
# =====================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"✂️ Total chunks: {len(chunks)}")

# =====================
# EMBEDDINGS
# =====================

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

texts = [c.page_content for c in chunks]

vectors = embeddings.embed_documents(texts)

# =====================
# UPLOAD TO PINECONE
# =====================

print("🚀 Uploading to Pinecone...")

upserts = [
    (
        str(uuid.uuid4()),   # FIXED: safe unique ID
        vectors[i],
        {
            "text": texts[i],
            "source": chunks[i].metadata.get("source", "file")
        }
    )
    for i in range(len(texts))
]

index.upsert(vectors=upserts)

print("✅ DONE: All data (files + websites) ingested!")
print(f"📊 Total vectors: {len(upserts)}")
