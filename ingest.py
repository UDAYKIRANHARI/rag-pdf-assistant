# ingest.py — multi-PDF ingestion
import os
import json
import faiss
import numpy as np
import pdfplumber
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def read_pdf(path):
    """Read a single PDF and return list of pages with text."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": page_num, "text": text})
    return pages


def chunk_texts(pages, chunk_size=800, overlap=100):
    """Split pages into overlapping chunks."""
    chunks = []
    for p in pages:
        tokens = p["text"].split()
        i = 0
        idx = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(
                {
                    "text": chunk_text,
                    "page": p["page"],
                    "chunk_idx": idx,
                }
            )
            i += chunk_size - overlap
            idx += 1
    return chunks


def embed_texts(texts):
    """Compute embeddings for a list of strings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, show_progress_bar=True)
    return embs.tolist()


def build_faiss(embeddings, metadatas, index_path="data/faiss_index"):
    """Build and save a FAISS index + metadata."""
    if not embeddings:
        raise ValueError("No embeddings to index.")
    dim = len(embeddings[0])
    xb = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(dim)
    index.add(xb)
    faiss.write_index(index, index_path + ".index")
    with open(index_path + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)


def ingest_multiple_pdfs(paths, index_path="data/faiss_index"):
    """
    Ingest multiple PDFs into a single FAISS index.
    Each chunk stores which PDF it came from.
    """
    all_chunks = []

    for path in paths:
        pages = read_pdf(path)
        chunks = chunk_texts(pages)
        for c in chunks:
            meta = {
                "source": os.path.basename(path),
                "page": c["page"],
                "chunk_idx": c["chunk_idx"],
                "text": c["text"][:2000],
            }
            all_chunks.append(meta)

    if not all_chunks:
        raise ValueError("No chunks created from provided PDFs.")

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)
    build_faiss(embeddings, all_chunks, index_path=index_path)
    return len(paths), len(all_chunks)


def ingest_pdf(path, index_path="data/faiss_index"):
    """
    Backwards-compatible single-PDF ingest.
    Just calls ingest_multiple_pdfs([path]).
    """
    return ingest_multiple_pdfs([path], index_path=index_path)


if __name__ == "__main__":
    import sys

    pdfs = sys.argv[1:]
    if not pdfs:
        print("Usage: python ingest.py file1.pdf file2.pdf ...")
        sys.exit(1)

    count_pdfs, count_chunks = ingest_multiple_pdfs(pdfs)
    print(f"Ingested {count_pdfs} PDFs, total chunks: {count_chunks}")
