import os
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "clinical-qa")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")

emb = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def ensure_index(dimension: int = 3072):
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": os.getenv("PINECONE_ENV", "us-east-1")}}
        )
    return pc.Index(INDEX_NAME)

def upsert_chunks(index, chunks: List[Dict]):
    vectors = []
    for c in chunks:
        vec = emb.embed_query(c["text"])
        vectors.append({
            "id": c["id"],
            "values": vec,
            "metadata": {**c["metadata"], "text": c["text"]}
        })
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])
