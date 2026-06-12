import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

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

model = ChatAnthropic(model="claude-haiku-4-5")

chat_history = []


def ask_question(user_question):
    print("You asked a question:", user_question)

    if chat_history:
        messages = (
            [
                SystemMessage(
                    content="Given the chat history, rewrite to be standalone & searchable"
                )
            ]
            + chat_history
            + [HumanMessage(content=f"New question: {user_question}")]
        )

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_question)
        print(f"Found {len(docs)} relevant documents:")
        for i, doc in enumerate(docs, 1):
            lines = doc.page_content.split("\n")[:2]
            preview = "\n".join(lines)
            print(f"Doc {i}. {preview}...")

        combined_input = f"""Based on following documents, please answer this question: {user_question}

      Documents:
      {"\n\n".join([f"- {doc.page_content}" for doc in docs])}

      please provide a clear, helpful answer using only the information provided from these documents. If you cant find the answer tell you dont know.
      """

    messages = (
            [
                SystemMessage(
                    content="You are a helpful assistant that answers questions based on the provided documents & conversation history"
                )
            ]
            + chat_history
            + [HumanMessage(content=combined_input)]
        )
