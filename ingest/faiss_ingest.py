import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ingest.loader import load_docs
from ingest.chunk import chunk_records

load_dotenv()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
OUT_DIR = "out/faiss"

def build_faiss(data_dir: str):
    recs = load_docs(data_dir)
    chunks: List[Dict] = chunk_records(recs)
    emb = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

    docs: List[Document] = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in chunks
    ]

    vs = FAISS.from_documents(docs, emb)
    os.makedirs(OUT_DIR, exist_ok=True)
    vs.save_local(OUT_DIR)
    print(f"FAISS index saved to {OUT_DIR}. Docs: {len(docs)} from {len(recs)} files.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()
    build_faiss(args.data)
