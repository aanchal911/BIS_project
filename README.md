# 🏗️ BIS Standards Recommendation Engine

> **Bureau of Indian Standards × Sigma Squad AI Hackathon**  
> Theme: Accelerating MSE Compliance – Automating BIS Standard Discovery

An AI-powered RAG (Retrieval-Augmented Generation) pipeline that turns a plain-English product description into ranked BIS standard recommendations in under 5 seconds — helping Indian Micro and Small Enterprises navigate regulatory compliance effortlessly.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Retrieval Strategy](#retrieval-strategy)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Running Inference (Judges)](#running-inference-judges)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview

Indian MSEs often spend weeks identifying applicable BIS regulations. This pipeline reduces that to seconds by:

1. **Ingesting** BIS SP 21 PDF documents (Building Materials category)
2. **Chunking** text with standard-aware boundary detection
3. **Indexing** chunks into a hybrid FAISS + BM25 retrieval system
4. **Reranking** candidates with a cross-encoder model
5. **Generating** top 3–5 standard recommendations with rationale via GPT-4o-mini
6. **Guarding** against hallucinations (LLM only cites standards present in context)

---

## System Architecture

```
Product Description
        │
        ▼
 ┌─────────────────┐
 │  Query Expansion │  ← LLM rewrites query with BIS domain vocabulary
 └────────┬────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌────────┐
│ FAISS  │  │  BM25  │   Dense + Sparse retrieval
│(Dense) │  │(Sparse)│
└────┬───┘  └───┬────┘
     └─────┬────┘
           ▼
    ┌─────────────┐
    │  RRF Fusion │   Reciprocal Rank Fusion merges both rankings
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │Cross-Encoder│   Reranks top-15 candidates → top-5
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  GPT-4o-mini│   Generates structured JSON output
    └──────┬──────┘
           ▼
  Top 3–5 BIS Standards
  + Rationale per standard
```

---

## Retrieval Strategy

| Component | Details |
|-----------|---------|
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector store** | FAISS (cosine similarity, CPU) |
| **Sparse retrieval** | BM25 Okapi (rank_bm25) |
| **Fusion** | Reciprocal Rank Fusion (k=60) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | `gpt-4o-mini` (temperature=0 for determinism) |
| **Chunk size** | 512 chars, 100 char overlap |
| **Chunking** | Standard-aware — splits at IS pattern boundaries first |

**Why hybrid?** Dense retrieval excels at semantic similarity; BM25 catches exact IS number matches. RRF fusion combines the best of both without requiring score normalisation.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/bis-rag-pipeline.git
cd bis-rag-pipeline

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add BIS SP 21 PDFs

Place the official BIS SP 21 PDF(s) in the `data/` folder:

```
data/
└── SP_21_BIS_Building_Materials.pdf
```

> If no PDFs are present, the pipeline falls back to a built-in demo corpus (useful for testing the pipeline end-to-end).

### 4. Build the Index

Run the notebook `bis_rag_pipeline.ipynb` cells 1–6, **or** import and run the pipeline directly:

```python
from src.rag_pipeline import BISRecommender

recommender = BISRecommender(data_dir="data")
result = recommender.recommend("High-strength deformed steel bars for reinforced concrete slabs")
print(result["standards_detail"])
```

### 5. Run a Quick Test

```bash
python inference.py --input data/public_test.json --output data/my_results.json
python eval_script.py --results data/my_results.json --ground_truth data/public_test.json
```

---

## Project Structure

```
bis-rag-pipeline/
│
├── inference.py              # ← Judge entry-point (REQUIRED)
├── eval_script.py            # ← Evaluation script (REQUIRED)
├── bis_rag_pipeline.ipynb    # Colab-ready notebook with all pipeline steps
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py       # BISRecommender class + all pipeline components
│
└── data/
    ├── public_test.json              # 10 sample queries with expected standards
    ├── public_test_results.json      # Results from public test set
    └── output_schema.json            # Example of required output format
```

---

## Running Inference (Judges)

The evaluation entry-point is `inference.py`. Run it as:

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

**Prerequisites before running:**
1. `pip install -r requirements.txt`
2. Set `OPENAI_API_KEY` in your environment or `.env` file
3. Build the FAISS index by running `bis_rag_pipeline.ipynb` cells 1–6 (this creates `vector_index/faiss_index/`)

**Output format** (strictly followed):

```json
[
  {
    "id": "q1",
    "retrieved_standards": ["IS 269:2015", "IS 8112:2013", "IS 12269:2013"],
    "latency_seconds": 1.234
  }
]
```

---

## Evaluation

```bash
python eval_script.py \
  --results data/my_results.json \
  --ground_truth data/public_test.json
```

**Target metrics:**

| Metric | Target | Description |
|--------|--------|-------------|
| Hit Rate @3 | > 80% | ≥1 correct standard in top-3 results |
| MRR @5 | > 0.7 | Mean Reciprocal Rank of first correct hit |
| Avg Latency | < 5s | Average per-query response time |

---

## Configuration

All tunable parameters are at the top of `src/rag_pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K_RETRIEVE` | 15 | Candidates before reranking |
| `TOP_K_RERANK` | 5 | Final results returned |

To use a **free local LLM** instead of OpenAI, set `USE_LOCAL_LLM = True` in the notebook (Cell 2). This loads `google/flan-t5-large` via HuggingFace Transformers.

---

## Results

Public test set results (10 queries, demo corpus):

| Metric | Score |
|--------|-------|
| Hit Rate @3 | **100.0%** ✅ (target > 80%) |
| MRR @5 | **1.0000** ✅ (target > 0.7) |
| Avg Latency | **2.04s** ✅ (target < 5s) |

> Results generated by running `python eval_script.py --results data/public_test_results.json --ground_truth data/public_test.json`.
> See `data/public_test_results.json` for the full output.

---

## Dependencies

Key libraries used:

- **LangChain** – document loading, chunking, vector store abstraction
- **FAISS** – fast approximate nearest-neighbour search
- **rank_bm25** – BM25 sparse retrieval
- **sentence-transformers** – dense embeddings + cross-encoder reranking
- **openai** – GPT-4o-mini for generation
- **pdfplumber** – accurate PDF text extraction

See `requirements.txt` for pinned versions.

---

## External APIs & Data Sources

| Resource | Usage |
|----------|-------|
| OpenAI API (`gpt-4o-mini`) | LLM generation and optional query expansion |
| BIS SP 21 (official PDF) | Sole source of truth for all standard content |
| HuggingFace Hub | Embedding and reranker model weights |

---

## Team

Built for the **BIS × Sigma Squad AI Hackathon** — helping MSEs navigate compliance with ease.
