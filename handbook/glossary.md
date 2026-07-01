# Glossary

> Part of the [RAG Hands-On handbook](../README.md#the-handbook). Plain-English definitions of the terms used across all chapters.

- **RAG (Retrieval-Augmented Generation)** — answering a question by first retrieving relevant documents, then having an LLM generate the answer from them.
- **Chunk** — a small slice of a document; the unit that gets embedded and retrieved.
- **Chunk size** — the maximum length (in characters here) of a chunk.
- **Chunk overlap** — characters repeated between consecutive chunks so context isn't lost at the boundary.
- **Separator** — the string a splitter cuts on (e.g. `"\n\n"` for paragraphs).
- **Embedding** — a numeric vector representing the meaning of a piece of text.
- **Vector store** — a database that stores embeddings and finds the nearest ones to a query.
- **Cosine similarity** — a measure of how close two vectors point in the same direction; used to rank chunk relevance.
- **HNSW** — Hierarchical Navigable Small World, the index algorithm Chroma uses for fast nearest-neighbor search.
- **top-k** — the number of most-similar chunks retrieved per query (here, 3).
- **Retriever** — the object that turns a query into a set of relevant chunks (`db.as_retriever(...)`).
- **Semantic chunking** — splitting where the *meaning* between sentences shifts, detected via embedding distance.
- **Agentic chunking** — letting an LLM decide where to split the text.
- **Query rewriting** — using the LLM to turn a conversational follow-up into a standalone search query.
- **Context window** — the maximum amount of text an LLM can consider in one request.
- **Partitioning** — splitting a PDF into typed elements (titles, paragraphs, tables, images) rather than one flat string.
- **`hi_res` strategy** — `unstructured`'s high-resolution mode that runs OCR + layout detection to recover tables and images accurately.
- **Title-based chunking** — grouping elements into chunks that break at section titles (`chunk_by_title`).
- **Multimodal / vision summary** — a searchable text description of a chunk's tables and images, written by a vision LLM and embedded in place of (a copy of) the raw content.
- **Base64 image payload** — an image encoded as text so it can be embedded in JSON metadata and passed inline to a vision model.
- **Multimodal RAG** — RAG where retrieved context and the final prompt include images and tables, not just text.
- **Score threshold** — a minimum similarity score a chunk must beat to be returned; results below it are dropped, so fewer than `k` may come back.
- **MMR (Max Marginal Relevance)** — a retrieval mode that picks results that are both relevant to the query and different from each other, to avoid near-duplicate chunks.
- **`fetch_k`** — how many candidate chunks MMR pulls before re-ranking them down to `k`.
- **`lambda_mult`** — the MMR knob trading relevance against diversity (0 = max diversity, 1 = max relevance).
- **Multi-query retrieval** — rephrasing the question into several variations, searching each, and combining the results to improve recall.
- **Reciprocal Rank Fusion (RRF)** — merging several ranked lists by scoring each item `1 / (k + position)` per list and summing; items ranked well across lists rise to the top.
- **RRF constant `k`** — a damping constant (here 60) that limits how much any single top-ranked result dominates the fused score.
- **Hybrid search** — combining dense (vector) and sparse (keyword) retrieval, then fusing the two result lists.
- **Dense retrieval** — ranking by embedding similarity; matches meaning rather than exact words.
- **Sparse retrieval** — ranking by term overlap (keyword search); matches exact words rather than meaning.
- **BM25** — the standard keyword-ranking algorithm, scoring chunks on Term Frequency and Inverse Document Frequency.
- **TF (Term Frequency)** — how often a query term appears in a given chunk.
- **IDF (Inverse Document Frequency)** — how rare a term is across the whole collection; rarer matching terms score higher.
- **EnsembleRetriever** — LangChain retriever that fuses several retrievers' results via weighted RRF.
- **Weighted RRF** — reciprocal rank fusion where each retriever's `1/(k+rank)` contribution is scaled by a weight before summing.
- **Reranker** — a second-stage model that rescores a retrieved candidate pool for relevance and keeps the best `top_n`.
- **Two-stage retrieval** — retrieve a large candidate pool cheaply, then rerank it precisely with a slower, more accurate model.
- **Bi-encoder** — encodes query and chunk separately into vectors that can be precomputed; fast but approximate (the embedding model).
- **Cross-encoder** — encodes query and chunk together to score relevance directly; accurate but slow, with no precomputation (the reranker).
- **Candidate pool** — the larger set of chunks the retriever surfaces to hand to the reranker.
- **`top_n`** — how many chunks a reranker keeps after rescoring the candidate pool.

---

[← Chapter 7 — Reranking](07-reranking.md) · [Handbook contents](../README.md#the-handbook)
