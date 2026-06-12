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

[← Chapter 2 — The RAG Pipeline](02-rag-pipeline.md) · [Handbook contents](../README.md#the-handbook) · [Next: Chapter 4 — Multimodal RAG over PDFs →](04-multimodal-rag.md)
