# ingest.py — PDF ingestion + FAISS index building (appendable)
import os
import json
import faiss
import numpy as np
import pdfplumber
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")


# -------------------------
# PDF reading & chunking
# -------------------------
def read_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": page_num, "text": text})
    return pages


def chunk_texts(pages, chunk_size=800, overlap=100):
    chunks = []
    for p in pages:
        tokens = p["text"].split()
        i = 0
        idx = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(
                {
                    "text": chunk_text,
                    "page": p["page"],
                    "chunk_idx": idx,
                }
            )
            i += max(1, chunk_size - overlap)
            idx += 1
    return chunks


# -------------------------
# Embeddings
# -------------------------
_embed_model = None


def get_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def embed_texts(texts):
    model = get_model()
    embs = model.encode(texts, show_progress_bar=True)
    return embs.astype("float32")


# -------------------------
# FAISS index helpers
# -------------------------
def load_existing_index(index_path=DEFAULT_INDEX_PATH):
    idx_file = index_path + ".index"
    meta_file = index_path + "_meta.json"
    if not os.path.exists(idx_file) or not os.path.exists(meta_file):
        return None, []
    index = faiss.read_index(idx_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        metas = json.load(f)
    return index, metas


def save_index(index, metas, index_path=DEFAULT_INDEX_PATH):
    idx_file = index_path + ".index"
    meta_file = index_path + "_meta.json"
    faiss.write_index(index, idx_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)


def ingest_pdf(path, index_path=DEFAULT_INDEX_PATH):
    """Ingest a single PDF and append its chunks to the FAISS index."""
    print(f"Ingesting {path} ...")
    pages = read_pdf(path)
    chunks = chunk_texts(pages)

    if not chunks:
        print("No text found in PDF.")
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    index, metas = load_existing_index(index_path)
    xb = np.array(embeddings).astype("float32")

    if index is None:
        dim = xb.shape[1]
        index = faiss.IndexFlatL2(dim)
    else:
        if index.d != xb.shape[1]:
            raise ValueError("Dimension mismatch between new embeddings and existing index.")

    index.add(xb)

    for c in chunks:
        meta = {
            "source": os.path.basename(path),
            "page": c["page"],
            "chunk_idx": c["chunk_idx"],
            "text": c["text"][:2000],
        }
        metas.append(meta)

    save_index(index, metas, index_path=index_path)
    print(f"Finished ingesting {path}. Total vectors: {index.ntotal}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/file.pdf")
        raise SystemExit(1)
    ingest_pdf(sys.argv[1])
