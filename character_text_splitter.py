from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticChunker,
)

test_text = """This is a test document. 

It contains multiple sentences. 

The purpose is to test the character text splitter. 

We want to see how it handles splitting the text into chunks based on a specified character limit. The splitter should create chunks that are no longer than the specified chunk size, and it should use the separator to determine where to split the text. This is important for processing large documents and ensuring that we can manage the text effectively for tasks like embedding or summarization."""

character_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0,
)

chunks = character_splitter.split_text(test_text)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} characters)")
    print(f"Chunk {i}:\n{chunk}\n")
    print()

recursive_splitter = RecursiveCharacterTextSplitter(
    separator=["\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=0,
)

chunks2 = recursive_splitter.split_text(test_text)
for i, chunk in enumerate(chunks2, 1):
    print(f"Chunk {i}: ({len(chunk)} characters)")
    print(f"Chunk {i}:\n{chunk}\n")
    print()


semantic_splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_thresold_type="percentile",
        breakpoint_thresold_amount=70,
    ),
)

chunks3 = semantic_splitter.split_text(test_text)
for i, chunk in enumerate(chunks3, 1):
    print(f"Chunk {i}: ({len(chunk)} characters)")
    print(f"Chunk {i}:\n{chunk}\n")
    print()