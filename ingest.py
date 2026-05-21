import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

from pinecone import Pinecone
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings

# ================= CONFIG =================
INDEX_NAME = "bitac-chatbot"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# ================= CLEAN =================
index.delete(delete_all=True)

# ================= LOAD FILES =================
docs = []

# TXT
for root, _, files in os.walk("bitac_files"):
    for f in files:
        path = os.path.join(root, f)

        if f.endswith(".txt"):
            docs += TextLoader(path).load()

        elif f.endswith(".pdf"):
            docs += PyPDFLoader(path).load()

        elif f.endswith(".docx"):
            docs += Docx2txtLoader(path).load()

        elif f.endswith(".xlsx"):
            df = pd.read_excel(path)
            docs.append(Document(page_content=df.to_string()))

# ================= WEB =================
urls = [
    "https://example.com"
]

for url in urls:
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text()

        docs.append(Document(page_content=text, metadata={"source": url}))
    except:
        pass

# ================= SPLIT =================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

# ================= EMBEDDING =================
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

texts = [c.page_content for c in chunks]
vectors = embeddings.embed_documents(texts)

# ================= UPLOAD =================
upserts = [
    (str(i), vectors[i], {"text": texts[i]})
    for i in range(len(texts))
]

index.upsert(vectors=upserts)

print("✅ INGEST DONE")
