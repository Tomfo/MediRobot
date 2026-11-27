"""Ingestion pipeline: load -> split -> embed -> upsert to Pinecone.

Usage:
    python ingest.py                       # ingest ./data
    python ingest.py --path ./docs --reset # wipe the namespace first
"""
import argparse
import hashlib
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

import config

SUPPORTED = {".pdf", ".txt", ".md"}
BATCH_SIZE = 100


def load_documents(root: Path):
    """Load every supported file under `root` into LangChain Documents."""
    docs = []
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    for path in files:
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)
        print(f"  loaded {path.name}: {len(loaded)} section(s)")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def chunk_id(doc) -> str:
    """Deterministic ID so re-running ingestion updates rather than duplicates."""
    key = f"{doc.metadata.get('source')}|{doc.metadata.get('page', 0)}|{doc.metadata.get('start_index', 0)}|{doc.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def ensure_index(pc: Pinecone):
    if config.PINECONE_INDEX not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{config.PINECONE_INDEX}'...")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBEDDING_DIMS[config.EMBEDDING_MODEL],
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
            time.sleep(2)
    return pc.Index(config.PINECONE_INDEX)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data", help="file or directory to ingest")
    parser.add_argument("--reset", action="store_true", help="delete the namespace before ingesting")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = ensure_index(pc)

    if args.reset:
        try:
            index.delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
            print(f"Cleared namespace '{config.PINECONE_NAMESPACE}'")
        except Exception as exc:  # namespace may not exist yet
            print(f"Reset skipped: {exc}")

    print("Loading documents...")
    docs = load_documents(root)
    if not docs:
        raise SystemExit("No supported documents found (.pdf, .txt, .md).")

    chunks = split_documents(docs)
    ids = [chunk_id(c) for c in chunks]
    print(f"Split into {len(chunks)} chunks")

    store = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        namespace=config.PINECONE_NAMESPACE,
    )
    for i in range(0, len(chunks), BATCH_SIZE):
        store.add_documents(chunks[i : i + BATCH_SIZE], ids=ids[i : i + BATCH_SIZE])
        print(f"  upserted {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Done.")


if __name__ == "__main__":
    main()
