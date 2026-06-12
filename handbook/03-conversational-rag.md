# Chapter 3 — History-Aware / Conversational RAG

> Part of the [RAG Hands-On handbook](../README.md#the-handbook). This builds on the pipeline from [Chapter 2](02-rag-pipeline.md) to handle multi-turn chat.

*Demonstrated in [history_aware_generation.py](../history_aware_generation.py).*

**Definition.** A RAG variant that handles multi-turn chat. Before retrieving, it uses the LLM to **rewrite a follow-up question into a standalone, searchable query** using the conversation history (so "What about its revenue?" becomes "What was NVIDIA's 2023 revenue?"). It then retrieves and answers with both the docs and the chat history in context.

**Advantages**
- Handles pronouns and follow-ups naturally ("it", "that", "the same company").
- Produces better retrieval for conversational interfaces than feeding the raw follow-up.

**Disadvantages**
- Extra LLM call to rewrite the query — more latency and cost per turn.
- Query rewriting can drift or misinterpret intent.
- Conversation history grows and must be managed to stay within the context window.

---

## Code Walkthrough

### `history_aware_generation.py` — rewrite the follow-up, then answer

Setup mirrors [Chapter 2](02-rag-pipeline.md)'s retrieval script — the same Chroma store, embedding model, and Haiku model — plus a module-level `chat_history` list that accumulates the conversation.

**1. Branch on whether there's history.** This is the key idea of the chapter.

```python
def ask_question(user_question):
    if chat_history:
        messages = (
            [SystemMessage(content="Given the chat history, rewrite to be standalone & searchable")]
            + chat_history
            + [HumanMessage(content=f"New question: {user_question}")]
        )
        result = model.invoke(messages)
        search_question = result.content.strip()
    else:
        search_question = user_question
        ...
```

On a follow-up turn (history present), the LLM rewrites the raw question into a standalone, searchable query — so "What about its revenue?" becomes something a vector search can actually match. On the first turn, the question is used as-is.

**2. Retrieve and build the grounded prompt** (first-turn branch).

```python
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_question)
        ...
        combined_input = f"""Based on following documents, please answer this question: {user_question}

      Documents:
      {"\n\n".join([f"- {doc.page_content}" for doc in docs])}
      ...
      """
```

Same retrieve-then-stuff pattern as Chapter 2, using the (possibly rewritten) `search_question` to retrieve.

**3. Assemble the final messages** — system instruction + prior turns + the grounded question.

```python
    messages = (
            [SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents & conversation history")]
            + chat_history
            + [HumanMessage(content=combined_input)]
        )
```

**Heads-up — this script is a work in progress.** As written, `ask_question` stops at *building* `messages`: it never calls `model.invoke(messages)`, returns nothing, and never appends the turn to `chat_history`. And `combined_input` is only defined in the first-turn (`else`) branch, so the history branch would hit a `NameError`. The concept and trade-offs above describe the intended design; the generation half still needs wiring up.

---

[← Chapter 2 — The RAG Pipeline](02-rag-pipeline.md) · [Handbook contents](../README.md#the-handbook) · [Next: Chapter 4 — Multimodal RAG over PDFs →](04-multimodal-rag.md)
