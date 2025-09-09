import argparse
from ingest.loader import load_docs
from ingest.chunk import chunk_records
from ingest.index import ensure_index, upsert_chunks

def main(data_dir: str):
    recs = load_docs(data_dir)
    chunks = chunk_records(recs)
    index = ensure_index(dimension=3072)
    upsert_chunks(index, chunks)
    print(f"Indexed {len(chunks)} chunks from {len(recs)} documents.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to directory with .txt/.pdf files")
    args = p.parse_args()
    main(args.data)
