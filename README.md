# RAG Hands-On

> A hands-on playground for learning **Retrieval-Augmented Generation (RAG)** and the **text-chunking** strategies that feed it, built with [LangChain](https://python.langchain.com/), [Chroma](https://docs.trychroma.com/), HuggingFace sentence-transformers, [`unstructured`](https://docs.unstructured.io/), and Anthropic's Claude.

## Overview

This repo walks through the full RAG lifecycle one piece at a time — chunk raw text, embed it, store it in a vector database, retrieve the relevant pieces for a question, and have Claude answer from them. The goal is to *see* each concept in isolation, then watch them compose into a working pipeline.

There are two tracks:

- **Text RAG** — a set of small, focused `.py` scripts over plain `.txt` documents (chunking → ingestion → retrieval → conversational generation).
- **Multimodal RAG** — [multi_modal_rag.ipynb](multi_modal_rag.ipynb) applies the same lifecycle to a **real PDF**, keeping text, tables, and figures and using Claude Opus vision to make the visual content searchable.

The detailed explanations live in the handbook below — read it in order, like chapters in a book.

## The Handbook

A sequential walk from first principles to a full multimodal pipeline. Each chapter explains the concept, its trade-offs, and the code that demonstrates it.

| # | Chapter | Covers | Code |
| --- | --- | --- | --- |
| 1 | [Chunking (Text Splitting)](handbook/01-chunking.md) | Character, recursive, semantic, and agentic chunking | [character_text_splitter.py](character_text_splitter.py), [agentic_chunking.py](agentic_chunking.py) |
| 2 | [The RAG Pipeline](handbook/02-rag-pipeline.md) | Embeddings, vector store, retrieval, and RAG generation | [ingestion_pipeline.py](ingestion_pipeline.py), [retrieval_pipeline.py](retrieval_pipeline.py) |
| 3 | [Conversational RAG](handbook/03-conversational-rag.md) | History-aware retrieval with query rewriting | [history_aware_generation.py](history_aware_generation.py) |
| 4 | [Multimodal RAG over PDFs](handbook/04-multimodal-rag.md) | Partitioning, vision summaries, multimodal answers (with screenshots) | [multi_modal_rag.ipynb](multi_modal_rag.ipynb) |
| 5 | [Advanced Retrieval](handbook/05-advanced-retrieval.md) | Score threshold, MMR, multi-query, and reciprocal rank fusion (with screenshots) | [retrieval_methods.py](retrieval_methods.py), [multi_query_retrieval.py](multi_query_retrieval.py), [reciprocal_rank_fusion.py](reciprocal_rank_fusion.py) |
| — | [Glossary](handbook/glossary.md) | Plain-English definitions of every term used | — |

**Suggested path:** read the chapters in order. To run the code, build the vector store with `ingestion_pipeline.py` first, then query it with `retrieval_pipeline.py` and `history_aware_generation.py`. The chunking scripts are standalone. The notebook is self-contained end-to-end.

## Repository Map

| Path | What it is |
| --- | --- |
| [character_text_splitter.py](character_text_splitter.py) | Character, recursive, and semantic chunking side by side |
| [agentic_chunking.py](agentic_chunking.py) | LLM-driven (agentic) chunking with Claude |
| [ingestion_pipeline.py](ingestion_pipeline.py) | Load → chunk → embed → store in Chroma |
| [retrieval_pipeline.py](retrieval_pipeline.py) | Similarity search + RAG answer generation |
| [history_aware_generation.py](history_aware_generation.py) | Conversational RAG with query rewriting |
| [multi_modal_rag.ipynb](multi_modal_rag.ipynb) | Multimodal RAG over a PDF (text + tables + images) with Claude Opus vision |
| [retrieval_methods.py](retrieval_methods.py) | Similarity, score-threshold, and MMR retrieval side by side |
| [multi_query_retrieval.py](multi_query_retrieval.py) | Multi-query generation + reciprocal rank fusion (compact) |
| [reciprocal_rank_fusion.py](reciprocal_rank_fusion.py) | Verbose RRF walkthrough with per-position scores |
| [ingest_tesla.py](ingest_tesla.py) | One-off: add `docs/tesla.txt` to the existing Chroma store |
| `semantic_chunking.py`, `answer_generation.py` | Placeholder / scratch files (currently empty) |
| `docs/` | Source documents (`google.txt`, `nvidia.txt`, `tesla.txt`) used for ingestion |
| [attention-is-all-you-need.pdf](attention-is-all-you-need.pdf) | Source PDF ingested by the multimodal notebook |
| `outputs/` | Screenshots of cell/script outputs (used in Chapters 4 and 5) |
| `handbook/` | The chapter-by-chapter guide |
| `db/chroma_db/`, `dbv2/chroma_db/` | Persisted Chroma vector stores (created by ingestion; gitignored) |

## Setup & Run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install langchain langchain-anthropic langchain-chroma \
            langchain-huggingface langchain-community \
            langchain-text-splitters chromadb \
            sentence-transformers python-dotenv
# For the multimodal notebook, also: "unstructured[all-docs]"
# (needs native deps: poppler, tesseract, libmagic, libheif)

# 3. Add your Anthropic API key
cp .env.example .env
# then edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...

# 4. Build the vector store from docs/
python ingestion_pipeline.py

# 5. Ask questions against it
python retrieval_pipeline.py
python history_aware_generation.py
```

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run and runs locally. Only the generation/agentic-chunking steps call the Anthropic API.

For the multimodal notebook, open [multi_modal_rag.ipynb](multi_modal_rag.ipynb) in Jupyter or VS Code and run it top to bottom — the full walkthrough is in [Chapter 4](handbook/04-multimodal-rag.md).
