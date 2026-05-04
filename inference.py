"""
inference.py – Judge entry-point for BIS RAG pipeline.

Usage:
    python inference.py --input hidden_private_dataset.json --output team_results.json
"""

import argparse
import json
import time
import re
import os
import pickle
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH   = Path("vector_index/faiss_index")
BM25_CACHE   = Path("vector_index/bm25_cache.pkl")
TOP_K        = 5
IS_PATTERN   = re.compile(r"\bIS\s+\d{1,5}(?::\d{4})?\b")

RAG_PROMPT = (
    "You are a BIS compliance expert. Given CONTEXT and a PRODUCT DESCRIPTION, "
    "return the top 3-5 applicable BIS standards ONLY as JSON:\n"
    '{"standards": [{"standard_id": "IS X:Y", "title": "...", "rationale": "..."}]}\n'
    "Only cite standards present in CONTEXT. No hallucinations. Return valid JSON only."
)


def hybrid_retrieve(query: str, vectorstore, bm25, chunks, top_k=TOP_K):
    """Hybrid dense+sparse retrieval with Reciprocal Rank Fusion."""
    # Dense
    dense_results = vectorstore.similarity_search_with_score(query, k=top_k)
    dense_rank = {doc.page_content: r for r, (doc, _) in enumerate(dense_results, 1)}

    # Sparse BM25
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    bm25_rank = {chunks[i]["text"]: r for r, i in enumerate(top_idx, 1)}

    # RRF fusion
    k_rrf = 60
    all_texts = set(dense_rank) | set(bm25_rank)
    rrf = {
        t: 1 / (k_rrf + dense_rank.get(t, top_k + 1))
          + 1 / (k_rrf + bm25_rank.get(t, top_k + 1))
        for t in all_texts
    }
    sorted_texts = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
    text_to_chunk = {c["text"]: c for c in chunks}
    return [text_to_chunk[t] for t in sorted_texts if t in text_to_chunk]


def main():
    parser = argparse.ArgumentParser(description="BIS Standards RAG Inference")
    parser.add_argument("--input",  required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Loading FAISS index...")
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{INDEX_PATH}'. "
            "Run the notebook (Cell 5) or rag_pipeline.py to build the index first."
        )
    vectorstore = FAISS.load_local(
        str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )

    # ── Load BM25 cache ────────────────────────────────────────────────────────
    chunks = []
    bm25 = None
    if BM25_CACHE.exists():
        with open(BM25_CACHE, "rb") as f:
            cached = pickle.load(f)
        # New format: dict with "bm25" and "chunks" keys (saved by rag_pipeline.py)
        if isinstance(cached, dict):
            bm25 = cached.get("bm25")
            chunks = cached.get("chunks", [])
        else:
            # Old format: bare BM25Okapi — chunks not persisted, hybrid unavailable
            bm25 = cached
            print("⚠️  BM25 cache is old format (no chunks). Rebuild index to enable hybrid retrieval.")
    if not chunks:
        print("⚠️  No chunks loaded — falling back to dense-only retrieval.")

    # ── LLM client ────────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key)

    # ── Load input ─────────────────────────────────────────────────────────────
    with open(args.input) as f:
        queries = json.load(f)

    print(f"Running inference on {len(queries)} queries...")
    outputs = []

    for item in queries:
        t0 = time.time()
        query = item["query"]

        # Retrieve – use hybrid if BM25 available, else dense-only
        if bm25 and chunks:
            retrieved_chunks = hybrid_retrieve(query, vectorstore, bm25, chunks)
            context = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)
        else:
            docs = vectorstore.similarity_search(query, k=TOP_K)
            context = "\n\n".join(d.page_content for d in docs)

        # Generate
        prompt = (
            f"{RAG_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"PRODUCT DESCRIPTION:\n{query}\n\n"
            "JSON:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content

        # Parse
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            data = json.loads(clean)
            standards = [s["standard_id"] for s in data.get("standards", [])]
        except Exception:
            standards = list(set(IS_PATTERN.findall(raw)))

        outputs.append({
            "id": item["id"],
            "retrieved_standards": standards[:TOP_K],
            "latency_seconds": round(time.time() - t0, 3),
        })

        print(f"  [{item['id']}] {len(standards)} standards | {outputs[-1]['latency_seconds']}s")

    # ── Write output ───────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(outputs, f, indent=2)

    avg_lat = round(sum(o["latency_seconds"] for o in outputs) / len(outputs), 3)
    print(f"\n✅ Results written to {args.output}")
    print(f"   Queries processed : {len(outputs)}")
    print(f"   Avg latency       : {avg_lat}s")


if __name__ == "__main__":
    main()
