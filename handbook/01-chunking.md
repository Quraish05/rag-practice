# Chapter 1 — Chunking (Text Splitting)

> Part of the [RAG Hands-On handbook](../README.md#the-handbook). Start here, then move on to building the pipeline that consumes these chunks.

**Definition.** Breaking a long document into smaller pieces ("chunks") before embedding. Chunks are the unit of retrieval — the model retrieves *chunks*, not whole documents.

**Why it matters.** Embedding models and LLMs have limited context windows, and similarity search works best when each chunk covers a single, focused idea. Chunk too big and retrieval is noisy and imprecise; chunk too small and you lose the surrounding context needed to answer.

This chapter demonstrates four strategies, from simplest to most sophisticated. The first three live side by side in [character_text_splitter.py](../character_text_splitter.py); the fourth is [agentic_chunking.py](../agentic_chunking.py).

---

## 1a. Character Text Splitting

*Demonstrated in [character_text_splitter.py](../character_text_splitter.py) and [ingestion_pipeline.py](../ingestion_pipeline.py) (`CharacterTextSplitter`).*

**Definition.** Splits text on a single separator (e.g. `"\n\n"`) and packs the pieces into chunks up to `chunk_size` characters, with an optional `chunk_overlap` between consecutive chunks.

**Advantages**
- Dead simple, fast, and fully deterministic.
- No model calls — zero cost, no latency.
- Predictable chunk sizes.

**Disadvantages**
- Can split mid-sentence or mid-idea if the separator doesn't line up with `chunk_size`.
- A single separator is brittle: if the text doesn't contain `"\n\n"`, it may not split where you expect.
- Ignores meaning entirely — purely mechanical.

---

## 1b. Recursive Character Text Splitting

*Demonstrated in [character_text_splitter.py](../character_text_splitter.py) (`RecursiveCharacterTextSplitter`).*

**Definition.** Tries an ordered list of separators (`["\n\n", "\n", " ", ""]`) in turn — paragraphs first, then lines, then words, then characters — recursing to a finer separator only when a chunk is still too big. This is the **recommended general-purpose splitter** in LangChain.

> Note: the parameter is `separators` (plural). A common typo is `separator` (singular), which is the argument for the non-recursive `CharacterTextSplitter`.

**Advantages**
- Keeps semantically related text together as much as possible (prefers paragraph and sentence boundaries).
- Robust across different document structures — degrades gracefully to finer separators.
- Still fast and free (no model calls).

**Disadvantages**
- Boundaries are still based on punctuation/whitespace, not actual meaning.
- Tuning `chunk_size` / `chunk_overlap` per document type still requires some trial and error.

---

## 1c. Semantic Chunking

*Demonstrated in [character_text_splitter.py](../character_text_splitter.py) (`SemanticChunker`).*

**Definition.** Embeds sentences and splits where the **embedding distance** between adjacent sentences spikes — i.e. where the topic shifts. The breakpoint is decided by a threshold (e.g. `percentile`, here ~70th percentile of distances).

> Note: the threshold parameters are `breakpoint_threshold_type` and `breakpoint_threshold_amount`. Watch for the `threshold` spelling.

**Advantages**
- Chunk boundaries follow *meaning*, not just formatting — each chunk tends to be one coherent topic.
- Adapts to the content instead of forcing a fixed character count.

**Disadvantages**
- Requires an embedding model call per sentence — slower and more expensive than character splitting.
- Chunk sizes become unpredictable (can be very large or very small).
- Sensitive to the threshold setting; needs tuning.

---

## 1d. Agentic Chunking

*Demonstrated in [agentic_chunking.py](../agentic_chunking.py).*

**Definition.** Hands the raw text to an **LLM** and asks it to decide the split points, returning the text with marker tokens (here `<<<SPLIT>>>`) inserted at logical boundaries. The code then splits on those markers and cleans up the pieces.

**Advantages**
- Most "human-like" boundaries — the model understands topic structure, headings, and intent.
- Can follow custom rules ("keep related info together", "~200 chars per chunk", etc.) expressed in natural language.

**Disadvantages**
- Slowest and most expensive — a full LLM call per document.
- Non-deterministic: the same input can chunk differently across runs.
- The model may ignore instructions, drop text, or place markers inconsistently — output needs validation.

---

[← Handbook contents](../README.md#the-handbook) · [Next: Chapter 2 — The RAG Pipeline →](02-rag-pipeline.md)
