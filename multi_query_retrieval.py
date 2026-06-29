from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from pydantic import BaseModel

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatHuggingFace(
    llm=HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
            "return_full_text": False,
        },
    )
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ------------------------------------------------------------
# Step 1: Generate Multiple Query Variations
# ------------------------------------------------------------

# HuggingFace text-generation endpoints don't support function-calling structured
# output, so we steer the model with format instructions and parse the JSON ourselves.
parser = PydanticOutputParser(pydantic_object=QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles.

Respond with ONLY a JSON object containing the actual queries. Do not repeat the schema.
The output must match this exact shape, with your own queries filled in:

{{"queries": ["first rephrased query", "second rephrased query", "third rephrased query"]}}"""

response = parser.invoke(llm.invoke(prompt))
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "=" * 60)

# ------------------------------------------------------------
# Step 2: Search with Each Query Variation & Store Results
# ------------------------------------------------------------

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n----- RESULTS FOR QUERY {i}: {query} -----")
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)
    for rank, doc in enumerate(docs, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"{rank}. {preview}...")

print("\n" + "=" * 60)

# ------------------------------------------------------------
# Step 3: Reciprocal Rank Fusion (RRF)
# ------------------------------------------------------------

k = 60  # RRF constant
fused_scores = {}
doc_lookup = {}

for docs in all_retrieval_results:
    for rank, doc in enumerate(docs):
        doc_id = doc.page_content
        doc_lookup[doc_id] = doc
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)

reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

print("\nFinal Fused Results (RRF):")
for rank, (doc_id, score) in enumerate(reranked, 1):
    preview = doc_lookup[doc_id].page_content[:100].replace("\n", " ")
    print(f"{rank}. (score={score:.4f}) {preview}...")
