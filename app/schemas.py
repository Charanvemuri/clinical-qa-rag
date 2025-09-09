from pydantic import BaseModel
from typing import List, Optional

class QARequest(BaseModel):
    query: str
    top_k: int = 6

class Citation(BaseModel):
    source: str
    text: str

class QAResponse(BaseModel):
    answer: str
    citations: List[Citation]
    disclaimer: Optional[str] = None
