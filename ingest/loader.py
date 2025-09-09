from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

def load_docs(data_dir: str) -> List[Dict]:
    records = []
    for p in Path(data_dir).rglob("*"):
        if p.is_dir():
            continue
        text = ""
        if p.suffix.lower() == ".txt":
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(p))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception:
                text = ""
        if text.strip():
            records.append({"id": p.stem, "text": text, "source": str(p)})
    return records
