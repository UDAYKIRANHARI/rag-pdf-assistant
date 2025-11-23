# retriever.py — FAISS retrieval + Groq/OpenAI answer generation
import os
import json
import faiss
import numpy as np
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------
# Secrets loading (local + Streamlit Cloud)
# -------------------------
load_dotenv()


def get_secret(name: str, default: str | None = None):
    """
    Read secrets in this order:
      1. Environment variables / .env
      2. Streamlit Cloud st.secrets
      3. Default value
    """
    v = os.getenv(name)
    if v is not None:
        return v

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return default


GROQ_API_KEY = get_secret("GROQ_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = get_secret("OPENAI_CHAT_MODEL", "llama-3.1-8b-instant")

DEFAULT_INDEX_PATH = "data/faiss_index"

print("DEBUG: GROQ_API_KEY present?:", bool(GROQ_API_KEY))

# -------------------------
# OpenAI-compatible client (Groq preferred)
# -------------------------
client = None
if GROQ_API_KEY:
    # Groq (OpenAI-compatible endpoint)
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
elif OPENAI_API_KEY:
    # Fallback to OpenAI if you ever set this
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None  # pure local mode, no LLM


# -------------------------
# Embedding model (local)
# -------------------------
_embed_model = None


def get_local_model():
    """Lazy-load the SentenceTransformer model once."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# -------------------------
# Index loading / query
# -------------------------
def load_index(index_path: str = DEFAULT_INDEX_PATH):
    """
    Load FAISS index and metadata from disk.
    Expects:
      index:  index_path + ".index"
      metas:  index_path + "_meta.json"
    """
    idx_file = index_path + ".index"
    meta_file = index_path + "_meta.json"

    if not os.path.exists(idx_file) or not os.path.exists(meta_file):
        raise FileNotFoundError("FAISS index or metadata not found. Ingest PDFs first.")

    index = faiss.read_index(idx_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        metas = json.load(f)

    return index, metas


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string into a vector."""
    model = get_local_model()
    vec = model.encode([text])[0].astype("float32")
    return vec


def retrieve(query: str, k: int = 4, index_path: str = DEFAULT_INDEX_PATH, allowed_sources=None):
    """
    Retrieve chunks for a query.

    - Uses FAISS to get top candidates.
    - Optionally filters to only chunks whose 'source'
      is in allowed_sources (list of filenames).
    - Tries not to take too many chunks from a single PDF.
    """
    index, metas = load_index(index_path)
    qv = embed_query(query)

    total = len(metas)
    if total == 0:
        return []

    # Ask FAISS for more candidates than we finally keep
    search_k = min(total, k * 6)
    D, I = index.search(np.array([qv]), search_k)

    allowed_set = set(allowed_sources) if allowed_sources else None
    results = []
    seen_per_source = {}  # source -> count

    for idx in I[0]:
        if idx < 0 or idx >= total:
            continue

        meta = metas[idx]
        src = meta.get("source", "unknown")

        # if user filtered PDFs, skip others
        if allowed_set and src not in allowed_set:
            continue

        seen_per_source.setdefault(src, 0)
        # at most 2 chunks per PDF in first pass (tweak if you want)
        if seen_per_source[src] >= 2:
            continue

        results.append(meta)
        seen_per_source[src] += 1

        if len(results) >= k:
            break

    # Fallback: if we still don't have enough, fill from top candidates
    if len(results) < k:
        for idx in I[0]:
            if idx < 0 or idx >= total:
                continue
            meta = metas[idx]
            src = meta.get("source", "unknown")

            if allowed_set and src not in allowed_set:
                continue
            if meta in results:
                continue

            results.append(meta)
            if len(results) >= k:
                break

    return results


# -------------------------
# Answer generation
# -------------------------
def generate_answer(query: str, retrieved, temperature: float = 0.0) -> str:
    """
    Use Groq/OpenAI Chat API to produce an answer grounded in retrieved chunks.

    If no API key is configured (client is None), we fall back to a simple
    local response listing the retrieved snippets.
    """
    if not retrieved:
        return "I don't know. No information found in the selected documents."

    # Build context with explicit source info
    context_items = []
    for r in retrieved:
        src = r.get("source", "unknown")
        page = r.get("page", "?")
        cidx = r.get("chunk_idx", "?")
        text = r.get("text", "")
        context_items.append(
            f"Source: {src} (page {page}, chunk {cidx})\n{text}"
        )

    context = "\n\n---\n\n".join(context_items)

    system_message = (
        "You are an assistant that MUST answer using only the provided CONTEXT. "
        "If the answer is not contained in the context, reply exactly: "
        "\"I don't know based on the provided documents.\" "
        "Cite sources (filename and page) for any factual claim."
    )

    user_prompt = f"""CONTEXT:
{context}

QUESTION:
{query}

Provide a concise answer and list the sources you used (filename and page)."""

    # If no client (no API key), fallback
    if client is None:
        return (
            "No LLM API key configured. "
            "Here are the retrieved snippets:\n\n"
            + context
        )

    # Call Groq/OpenAI chat completion
    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        # On error, degrade gracefully
        return (
            f"OpenAI/Groq API error: {e}\n\n"
            "Here are the retrieved snippets:\n\n"
            + context
        )


# -------------------------
# Simple manual test
# -------------------------
if __name__ == "__main__":
    # Quick debug usage example
    q = "What skills are mentioned?"
    try:
        chunks = retrieve(q, k=4)
        ans = generate_answer(q, chunks)
        print("ANSWER:\n", ans)
    except Exception as e:
        print("Error in manual test:", e)
