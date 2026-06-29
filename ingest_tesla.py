"""One-off: add docs/tesla.txt to the existing Chroma collection without
re-ingesting google.txt / nvidia.txt (which are already in the DB)."""

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

documents = TextLoader("docs/tesla.txt").load()
chunks = CharacterTextSplitter(chunk_size=800, chunk_overlap=0).split_documents(
    documents
)
print(f"tesla.txt -> {len(chunks)} chunks")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

db.add_documents(chunks)
print(f"Added {len(chunks)} Tesla chunks. Collection size: {db._collection.count()}")
