import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    if not os.path.exists(docs_path):
        print(
            f"No documents directory found. Please create a '{docs_path}' directory and add text files to it."
        )

        return []
    loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()

    if len(documents) == 0:
        print(
            "No text files found in the 'documents' directory. Please add some text files to it."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"Source: {doc.metadata['source']}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i + 1}: {chunk.page_content[:100]}...")

        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks.")

    return chunks


def create_embeddings(chunks, persist_directory="db/chroma_db"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Finished creating vector store...")

    print(f"Created & stored to {persist_directory}...")
    return vector_store


def main():
    print("Main")
    documents = load_documents(docs_path="docs")

    chunks = split_documents(documents)

    vector_store = create_embeddings(chunks)


if __name__ == "__main__":
    main()
