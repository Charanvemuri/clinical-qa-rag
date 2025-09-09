import os
from typing import List, Dict
from dotenv import load_dotenv
from .prompts import PROMPT

load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX", "clinical-qa")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")

def _clients():
    from pinecone import Pinecone
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    emb = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    return index, emb, llm

def retrieve(query: str, top_k: int = 6) -> List[Dict]:
    index, emb, _ = _clients()
    qvec = emb.embed_query(query)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
    hits = []
    for m in res.matches:
        hits.append({
            "score": m.score,
            "text": m.metadata.get("text", ""),
            "source": m.metadata.get("source", "unknown"),
        })
    return hits

def _build_context(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] {h['text']}\n(Source: {h['source']})\n")
    return "\n".join(lines)

def answer(query: str, top_k: int = 6) -> Dict:
    _, _, llm = _clients()
    hits = retrieve(query, top_k=top_k)
    context = _build_context(hits)
    msg = PROMPT.format_messages(question=query, context=context)
    out = llm.invoke(msg)
    citations = [{"source": h["source"], "text": h["text"]} for h in hits]
    return {"answer": out.content, "citations": citations}
