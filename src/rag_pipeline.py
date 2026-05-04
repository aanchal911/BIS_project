"""
src/rag_pipeline.py  (canonical location)
BIS Standards RAG Pipeline – Enhanced version with:
  - Hybrid retrieval (Dense FAISS + BM25 sparse)
  - Reciprocal Rank Fusion (RRF)
  - Cross-encoder reranking
  - LLM query expansion
  - Hallucination guard
"""

import os
import re
import json
import time
import pickle
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from openai import OpenAI
import pdfplumber
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 100
TOP_K_RETRIEVE    = 15
TOP_K_RERANK      = 5
INDEX_DIR         = Path("vector_index")
BM25_CACHE        = INDEX_DIR / "bm25_cache.pkl"
IS_PATTERN        = re.compile(r'\bIS\s+\d{1,5}(?::\d{4})?\b')

# ── System Prompt ─────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """
You are a Bureau of Indian Standards (BIS) compliance expert helping Indian Micro and Small
Enterprises (MSEs) identify the correct standards for building-material products.

Given the CONTEXT (excerpts from BIS SP 21) and a PRODUCT DESCRIPTION, return the top 3–5
most applicable BIS standards in this STRICT JSON format only:

{
  "standards": [
    {
      "standard_id": "IS XXXX:YYYY",
      "title": "<Full title of the standard>",
      "rationale": "<One clear sentence explaining why this standard applies to the product>"
    }
  ]
}

RULES:
- Only cite standards that appear verbatim in the provided CONTEXT.
- NEVER invent or guess standard IDs — if unsure, omit.
- Rank from most to least relevant.
- Return valid JSON ONLY — no preamble, no markdown fences.
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text.strip())
    return "\n\n".join(full_text)


def load_documents(data_dir: Path) -> list[dict]:
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDFs found — using built-in demo corpus.")
        return [{"source": "demo", "text": _demo_corpus()}]
    docs = []
    for p in pdf_files:
        print(f"  📖 {p.name}")
        docs.append({"source": p.name, "text": extract_text_from_pdf(p)})
    return docs


def _demo_corpus() -> str:
    return """
IS 269:2015 – Ordinary Portland Cement – Specification.
Covers physical, chemical and mechanical requirements for OPC grades 33, 43 and 53 used in
general construction, plastering and masonry work.

IS 8112:2013 – 43 Grade Ordinary Portland Cement – Specification.
Specifies requirements for 43 grade OPC including fineness, setting time and compressive strength
suitable for general civil engineering construction.

IS 12269:2013 – 53 Grade Ordinary Portland Cement – Specification.
High-strength OPC for prestressed concrete, precast elements and high-performance structures.

IS 1489 (Part 1):1991 – Portland Pozzolana Cement (Fly Ash Based) – Specification.
For cement made by intergrinding OPC clinker with fly ash, used in marine/hydraulic structures.

IS 1489 (Part 2):1991 – Portland Pozzolana Cement (Calcined Clay Based) – Specification.

IS 1786:2008 – High Strength Deformed Steel Bars and Wires for Concrete Reinforcement.
Covers Fe 415, Fe 500, Fe 550, Fe 600 grades; includes bend/rebend tests and weldability.

IS 2062:2011 – Hot Rolled Medium and High Tensile Structural Steel.
Covers plates, strips, shapes and sections for structural use in bridges and buildings.

IS 432 (Part 1):1982 – Mild Steel and Medium Tensile Steel Bars and Hard-drawn Steel Wire.

IS 383:2016 – Coarse and Fine Aggregate for Concrete – Specification.
Covers natural, crushed and manufactured aggregates; grading, deleterious content, shape index.

IS 2386 (Part 1 to 8):1963 – Methods of Test for Aggregates for Concrete.
Series covering particle size, shape, specific gravity, moisture, soundness and alkali reactivity.

IS 456:2000 – Plain and Reinforced Concrete – Code of Practice.
The master code for concrete design; covers materials, mix design, durability, reinforcement cover.

IS 10262:2019 – Concrete Mix Proportioning – Guidelines.
Stepwise mix design procedure for ordinary, standard and high-performance concrete.

IS 516:1959 – Methods of Tests for Strength of Concrete.
Cube and cylinder compressive strength, flexural strength and modulus of elasticity tests.

IS 2645:2003 – Integral Cement Waterproofing Compounds – Specification.
For admixtures added to cement concrete/mortar to reduce water permeability.

IS 3812 (Part 1):2003 – Pulverised Fuel Ash (Fly Ash) – Specification.
For fly ash used as pozzolanic material in cement, concrete and for bricks/tiles.

IS 12330:1988 – Sulphate Resisting Portland Cement – Specification.
For foundations and structures in sulphate-rich soils or water.

IS 455:1989 – Portland Slag Cement – Specification.
Blended cement using ground granulated blast furnace slag; lower heat of hydration.

IS 1343:2012 – Prestressed Concrete – Code of Practice.
Design of prestressed concrete structures including post-tensioning and pre-tensioning.

IS 2204:1962 – Construction of Reinforced Concrete Shell Roof.

IS 875 (Part 1 to 5) – Code of Practice for Design Loads for Buildings and Structures.
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# 2. CHUNKING
# ═════════════════════════════════════════════════════════════════════════════

def build_chunks(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    all_chunks = []
    for doc in documents:
        for chunk_text in splitter.split_text(doc["text"]):
            std_ids = list(set(IS_PATTERN.findall(chunk_text)))
            all_chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "standard_ids": std_ids,
            })
    return all_chunks


# ═════════════════════════════════════════════════════════════════════════════
# 3. VECTOR STORE + BM25
# ═════════════════════════════════════════════════════════════════════════════

def build_or_load_index(chunks: list[dict], embeddings) -> FAISS:
    INDEX_DIR.mkdir(exist_ok=True)
    index_path = INDEX_DIR / "faiss_index"
    if index_path.exists():
        print("  ⚡ Loading cached FAISS index...")
        return FAISS.load_local(str(index_path), embeddings,
                                allow_dangerous_deserialization=True)
    print("  🔨 Building FAISS index...")
    lc_docs = [
        Document(
            page_content=c["text"],
            metadata={"source": c["source"],
                      "standard_ids": ", ".join(c["standard_ids"])},
        )
        for c in chunks
    ]
    vs = FAISS.from_documents(lc_docs, embeddings)
    vs.save_local(str(index_path))
    print(f"  ✅ Index saved ({vs.index.ntotal} vectors)")
    return vs


def build_or_load_bm25(chunks: list[dict]):
    if BM25_CACHE.exists():
        with open(BM25_CACHE, "rb") as f:
            cached = pickle.load(f)
        # Support both old format (bare BM25Okapi) and new dict format
        if isinstance(cached, dict):
            return cached["bm25"]
        return cached
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    # Save as dict so inference.py can recover both bm25 AND chunks
    with open(BM25_CACHE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    return bm25


# ═════════════════════════════════════════════════════════════════════════════
# 4. RETRIEVAL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def hybrid_retrieve(query: str, vectorstore, bm25, chunks: list[dict],
                    top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    """Dense + Sparse retrieval fused with RRF."""
    # Dense
    dense_results = vectorstore.similarity_search_with_score(query, k=top_k)
    dense_rank = {doc.page_content: r for r, (doc, _) in enumerate(dense_results, 1)}

    # Sparse BM25
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    bm25_rank = {chunks[i]["text"]: r for r, i in enumerate(top_idx, 1)}

    # Reciprocal Rank Fusion
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


def rerank(query: str, candidates: list[dict], cross_encoder,
           top_k: int = TOP_K_RERANK) -> list[dict]:
    """Cross-encoder reranking of retrieved candidates."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ═════════════════════════════════════════════════════════════════════════════
# 5. QUERY EXPANSION
# ═════════════════════════════════════════════════════════════════════════════

def expand_query(query: str, client: OpenAI) -> str:
    """Use LLM to generate a BIS-domain-enriched version of the query."""
    prompt = (
        "You are a BIS standards expert. Rewrite the following product description "
        "to include relevant technical terms, material properties, and Indian standards "
        "domain vocabulary that would help retrieve applicable BIS standards. "
        "Return only the enriched query, no explanation.\n\n"
        f"Original: {query}\nEnriched:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return query  # Fallback to original


# ═════════════════════════════════════════════════════════════════════════════
# 6. GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_recommendations(query: str, chunks: list[dict],
                              client: OpenAI) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Standards: {', '.join(c['standard_ids']) or 'unknown'}]\n{c['text']}"
        for c in chunks
    )
    prompt = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"PRODUCT DESCRIPTION:\n{query}\n\n"
        f"JSON:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def parse_output(raw: str) -> tuple[list[str], list[dict]]:
    """Return (standard_id_list, full_standards_list)."""
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        data = json.loads(clean)
        stds = data.get("standards", [])
        ids = [s["standard_id"] for s in stds]
        return ids, stds
    except Exception:
        ids = list(set(IS_PATTERN.findall(raw)))
        stds = [{"standard_id": i, "title": "", "rationale": ""} for i in ids]
        return ids, stds


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE CLASS
# ═════════════════════════════════════════════════════════════════════════════

class BISRecommender:
    def __init__(self, data_dir: str = "data", openai_api_key: Optional[str] = None):
        self.data_dir = Path(data_dir)
        print("🔧 Initialising BIS Recommender...")

        # Models
        print("  Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  Loading cross-encoder...")
        self.cross_encoder = CrossEncoder(RERANK_MODEL)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=api_key)

        # Data
        print("  Loading documents...")
        docs = load_documents(self.data_dir)
        self.chunks = build_chunks(docs)
        print(f"  ✅ {len(self.chunks)} chunks ready")

        # Indexes
        self.vectorstore = build_or_load_index(self.chunks, self.embeddings)
        self.bm25 = build_or_load_bm25(self.chunks)
        print("✅ BIS Recommender ready!\n")

    def recommend(self, query: str, use_query_expansion: bool = True) -> dict:
        t0 = time.time()

        # Optional query expansion
        effective_query = expand_query(query, self.client) if use_query_expansion else query

        # Hybrid retrieval
        candidates = hybrid_retrieve(
            effective_query, self.vectorstore, self.bm25, self.chunks
        )

        # Cross-encoder reranking
        top_chunks = rerank(effective_query, candidates, self.cross_encoder)

        # LLM generation
        raw = generate_recommendations(query, top_chunks, self.client)
        ids, standards = parse_output(raw)

        return {
            "query": query,
            "expanded_query": effective_query,
            "retrieved_standards": ids[:TOP_K_RERANK],
            "standards_detail": standards[:TOP_K_RERANK],
            "latency_seconds": round(time.time() - t0, 3),
            "raw_llm_output": raw,
        }
