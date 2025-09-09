# LLM-Powered Clinical QA Assistant (RAG)
I built a Retrieval-Augmented Generation (RAG) API to answer clinical registry questions with citations.

## Quickstart
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys
python ingest/run_ingest.py --data data/registry_docs
bash scripts/serve.sh

## Query
curl -sS -X POST "http://127.0.0.1:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the inclusion criteria for registry X?"}'
Built by:Sai Charan Vemuri