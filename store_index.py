
"""
store_index.py
---------------
Load medical knowledge-base files, split them into chunks,
generate Hugging Face embeddings, and upload them to Pinecone.

Supported formats:
PDF, DOCX, CSV, XLSX, TXT, MD, MARKDOWN
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env
load_dotenv()

# -------------------- Configuration --------------------

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")

# Prefer the new knowledge_base folder; fall back to data if present.
if os.getenv("KNOWLEDGE_BASE_DIR"):
    DATA_DIR = Path(os.environ["KNOWLEDGE_BASE_DIR"])
elif Path("knowledge_base").is_dir():
    DATA_DIR = Path("knowledge_base")
else:
    DATA_DIR = Path("data")

# -------------------- Import Helpers --------------------

try:
    # Use this import if helper.py is inside src/
    from src.helper import (
        load_knowledge_files,
        text_split,
        download_hugging_face_embeddings,
    )
except ImportError:
    try:
        # Use this import if helper.py is in the project root
        from helper import (
            load_knowledge_files,
            text_split,
            download_hugging_face_embeddings,
        )
    except ImportError as exc:
        print("ERROR: Could not import the helper functions.")
        print("Check that helper.py exists in src/ or the project root.")
        print(f"Details: {exc}")
        sys.exit(1)


def main():
    # -------------------- Validate Configuration --------------------

    if not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY is missing from your .env file.")
        sys.exit(1)

    if not DATA_DIR.exists():
        print(f"ERROR: Knowledge-base folder not found: {DATA_DIR}")
        print("Create the folder and add your medical documents.")
        sys.exit(1)

    # -------------------- Load Documents --------------------

    print("\n" + "=" * 60)
    print("MEDICAL CHATBOT - DOCUMENT INDEXING")
    print("=" * 60)
    print(f"Knowledge-base folder: {DATA_DIR.resolve()}")
    print(f"Pinecone index: {INDEX_NAME}")

    print("\nLoading supported medical documents...")

    try:
        documents = load_knowledge_files(str(DATA_DIR))
    except Exception as exc:
        print(f"ERROR loading documents: {exc}")
        sys.exit(1)

    if not documents:
        print("ERROR: No documents were loaded.")
        print("Check the folder and supported file formats.")
        sys.exit(1)

    print(f"Loaded {len(documents)} document pages/records.")

    # -------------------- Split Documents --------------------

    print("\nSplitting documents into chunks...")

    try:
        chunks = text_split(documents)
    except Exception as exc:
        print(f"ERROR splitting documents: {exc}")
        sys.exit(1)

    if not chunks:
        print("ERROR: No text chunks were created.")
        sys.exit(1)

    print(f"Created {len(chunks)} chunks.")

    # -------------------- Generate Embeddings --------------------

    print("\nLoading Hugging Face embedding model...")

    try:
        embeddings = download_hugging_face_embeddings()
    except Exception as exc:
        print(f"ERROR loading embeddings: {exc}")
        sys.exit(1)

    # -------------------- Upload to Pinecone --------------------

    print(f"\nUploading chunks to Pinecone index '{INDEX_NAME}'...")
    print("Ensure this index already exists and its dimension matches")
    print("your embedding model.")

    try:
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=INDEX_NAME,
        )
    except Exception as exc:
        print(f"\nERROR uploading documents to Pinecone: {exc}")
        print("\nCheck the following:")
        print("1. PINECONE_API_KEY is valid.")
        print("2. The Pinecone index exists.")
        print("3. The index dimension matches your embedding model.")
        print("4. Your network and Pinecone permissions are working.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DOCUMENT INDEXING COMPLETED")
    print("=" * 60)
    print(f"Source folder: {DATA_DIR}")
    print(f"Chunks submitted: {len(chunks)}")
    print(f"Pinecone index: {INDEX_NAME}")
    print("\nYour medical knowledge base has been uploaded.")
    print("=" * 60)


if __name__ == "__main__":
    main()
