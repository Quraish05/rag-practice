from collections import defaultdict
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
            # Return only the generated completion, not the echoed prompt.
            # Otherwise the schema JSON from the format instructions ends up in
            # the output and the parser grabs it instead of the model's answer.
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

{parser.get_format_instructions()}"""

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
# Step 3: Apply Reciprocal Rank Fusion
# ------------------------------------------------------------


def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("=" * 60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}

    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1

    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")

        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content

            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]

            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk

            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)

            # Add to RRF score
            rrf_scores[chunk_content] += position_score

            if verbose:
                print(
                    f"  Position {position}: {chunk_id} +{position_score:.4f} "
                    f"(running total: {rrf_scores[chunk_content]:.4f})"
                )
                print(f"    Preview: {chunk_content[:80]}...")

    if verbose:
        print()

    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [
            (all_unique_chunks[chunk_content], score)
            for chunk_content, score in rrf_scores.items()
        ],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True,  # Highest scores first
    )

    if verbose:
        print(
            f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks "
            f"from {len(chunk_lists)} queries"
        )

    return sorted_chunks


# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)


# ------------------------------------------------------------
# Step 4: Display Final Fused Results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RRF RANKING")
print("=" * 60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)
