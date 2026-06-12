import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "What happened to NVIDIA in 2023?"

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": 0.3, "k": 3},
# )

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print("Context")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: {doc.page_content[:200]}...")

combined_input = f"""Based on the following retrieved documents, answer the question: {query}

Documents:
{chr(10).join([doc.page_content for doc in relevant_docs])}
Please provide a concise answer based on the above documents. If you cant find the answer in the documents, say you don't know.
"""

model = ChatAnthropic(model="claude-haiku-4-5")

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers questions based on the provided documents."
    ),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)
print(f"Answer: {result.content}")
