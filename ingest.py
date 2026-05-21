import os
import json
import hashlib
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

TRACK_FILE = "track.json"

# ================= TRACKING =================
if os.path.exists(TRACK_FILE):
    done = json.load(open(TRACK_FILE))
else:
    done = {}

def save():
    json.dump(done, open(TRACK_FILE, "w"))

def uid(text, src):
    return hashlib.md5((text + src).encode()).hexdigest()

# ================= LOAD =================
docs = []

for root, _, files in os.walk("bitac_files"):
    for f in files:
        path = os.path.join(root, f)

        if path in done:
            continue

        try:
            if f.endswith(".txt"):
                docs += TextLoader(path).load()

            elif f.endswith(".pdf"):
                docs += PyPDFLoader(path).load()

            elif f.endswith(".docx"):
                docs += Docx2txtLoader(path).load()

            elif f.endswith(".xlsx"):
                df = pd.read_excel(path)
                docs.append(Document(page_content=df.astype(str).to_string()))

            done[path] = True

        except Exception as e:
            print("File error:", e)

# ================= WEB =================
urls = ["https://bitac.dhaka.gov.bd/",
        https://bitac.dhaka.gov.bd/,
        https://bitac.gov.bd/pages/officers
       ]

for url in urls:
    wid = "web_" + url

    if wid in done:
        continue

    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        text = soup.get_text(separator=" ", strip=True)

        docs.append(Document(page_content=text, metadata={"source": url}))

        done[wid] = True

    except Exception as e:
        print("Web error:", e)

save()

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

# ================= UPLOAD (NO DELETE = SAFE) =================
upserts = []

for i in range(len(texts)):
    vid = uid(texts[i], chunks[i].metadata.get("source", "file"))

    upserts.append((
        vid,
        vectors[i],
        {
            "text": texts[i],
            "source": chunks[i].metadata.get("source", "file")
        }
    ))

index.upsert(vectors=upserts)

print("✅ INGEST DONE SAFE (NO DATA LOSS)")
print("📊 Vectors:", len(upserts))
