import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .prompts import PROMPT

load_dotenv()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
FAISS_DIR = "out/faiss"

def _clients():
    emb = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    vs = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
    return vs, llm

def retrieve(query: str, top_k: int = 6) -> List[Dict]:
    vs, _ = _clients()
    results = vs.similarity_search_with_score(query, k=top_k)
    hits: List[Dict] = []
    for doc, score in results:
        hits.append({
            "score": float(score),
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        })
    return hits

def _build_context(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] {h['text']}\n(Source: {h['source']})\n")
    return "\n".join(lines)

def answer(query: str, top_k: int = 6) -> Dict:
    vs, llm = _clients()
    hits = retrieve(query, top_k=top_k)
    context = _build_context(hits)
    msg = PROMPT.format_messages(question=query, context=context)
    out = llm.invoke(msg)
    citations = [{"source": h["source"], "text": h["text"]} for h in hits]
    return {"answer": out.content, "citations": citations}
