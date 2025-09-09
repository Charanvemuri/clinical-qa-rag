from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import QARequest, QAResponse, Citation
from .chains_faiss import answer
from .guards import disclaimer

app = FastAPI(title="Clinical QA RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    try:
        out = answer(req.query, top_k=req.top_k)
        return QAResponse(
            answer=out["answer"],
            citations=[Citation(**c) for c in out["citations"]],
            disclaimer=disclaimer()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
