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

## Code Walkthrough

### `character_text_splitter.py` — three splitters on one sample

The script runs top to bottom, feeding the same `test_text` (a few short paragraphs separated by blank lines) to each strategy in turn.

**1. Character splitting (1a).** Split only on the blank-line separator, packing pieces up to 100 chars.

```python
character_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0,
)
chunks = character_splitter.split_text(test_text)
```

Because it never cuts *inside* a paragraph, any single paragraph longer than 100 chars stays whole — `chunk_size` is a target, not a hard cap, when the separator doesn't line up.

**2. Recursive splitting (1b).** Try a list of separators from coarse to fine.

```python
recursive_splitter = RecursiveCharacterTextSplitter(
    separator=["\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=0,
)
chunks2 = recursive_splitter.split_text(test_text)
```

It recurses to a finer separator (paragraph → line → word → char) only when a chunk is still too big, so a long paragraph gets broken at spaces. **Heads-up:** the keyword should be `separators` (plural); as written, `separator` is ignored and the splitter falls back to its defaults.

**3. Semantic chunking (1c).** Split where the *meaning* between sentences shifts.

```python
semantic_splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_thresold_type="percentile",
        breakpoint_thresold_amount=70,
    ),
)
chunks3 = semantic_splitter.split_text(test_text)
```

It embeds each sentence and splits where adjacent-sentence embedding distance spikes. **Heads-up:** the `breakpoint_*` args belong on `SemanticChunker` itself (and are spelled `breakpoint_threshold_type` / `_amount`); here they're passed into `HuggingFaceEmbeddings` and misspelled, so they have no effect and the chunker uses defaults.

Each strategy is followed by the same loop that prints every chunk and its length, so you can eyeball how the boundaries differ:

```python
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} characters)")
    print(f"Chunk {i}:\n{chunk}\n")
```

### `agentic_chunking.py` — let the LLM place the splits (1d)

**1. Model + input.** A Haiku model and a short multi-section sample (`tesla_text`).

```python
llm = ChatAnthropic(model="claude-haiku-4-5")
tesla_text = """Tesla's Q3 Results ... reduce costs."""
```

**2. The instruction prompt.** Ask the model to insert a literal marker at each boundary it chooses.

```python
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text}
...
"""
```

**3. Call the model, then split on the marker.**

```python
response = llm.invoke(prompt)
marked_text = response.content
chunks = marked_text.split("<<<SPLIT>>>")
```

The model returns the original text with `<<<SPLIT>>>` tokens inserted; the code recovers the chunks by splitting on that token.

**4. Clean up and print.**

```python
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:  # Only keep non-empty chunks
        clean_chunks.append(cleaned)
```

Trims whitespace and drops empty pieces. Because the boundaries come from an LLM, the output varies run-to-run and should be validated — the trade-off called out above.

---

[← Handbook contents](../README.md#the-handbook) · [Next: Chapter 2 — The RAG Pipeline →](02-rag-pipeline.md)
