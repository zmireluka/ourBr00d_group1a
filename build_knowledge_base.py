"""
Offline indexer for the static knowledge base (phase 2 RAG).

Reads the corpus from config.KNOWLEDGE_DIR (.txt — books, character dossiers,
lore from the prof/OMSK), chunks it, and embeds it ONCE into a persistent
ChromaDB store (config.KNOWLEDGE_DB_PATH). server.py later reads the store
read-only when config.USE_STATIC_KNOWLEDGE=True.

Deliberately separate from the in-session collection: that one is ephemeral
(dies with the process); this one is persistent (pre-indexed, fully available
from turn 1).

Usage (inside the whisper container, where the corpus is mounted via .:/app):
    python3 build_knowledge_base.py            # builds the index
    python3 build_knowledge_base.py --dry-run  # counts files + chunks only

Re-running rebuilds the collection from scratch (old one is discarded).
"""

import argparse
import re

import config


def chunk_text(text, size, overlap):
    """Splits text into chunks of ~size characters with overlap.
    Prefers to cut at sentence boundaries, otherwise at word boundaries —
    so no sentence is split mid-word. Whitespace is normalised first."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    pos, n = 0, len(text)
    while pos < n:
        end = min(pos + size, n)
        if end < n:
            window = text[pos:end]
            # prefer sentence boundary, fall back to last space
            cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if cut < size * 0.5:
                cut = window.rfind(" ")
            if cut > size * 0.3:
                end = pos + cut + 1
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)
    return chunks


def collect_chunks():
    """Reads all .txt files from KNOWLEDGE_DIR, returns list of (id, text, metadata)."""
    files = sorted(config.KNOWLEDGE_DIR.glob("*.txt"))
    records = []
    for path in files:
        source = path.stem  # filename without .txt = source/title
        text = path.read_text(encoding="utf-8", errors="ignore")
        pieces = chunk_text(text, config.KNOWLEDGE_CHUNK_CHARS,
                            config.KNOWLEDGE_CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            cid = f"{source}__{i}"
            records.append((cid, piece, {"source": source, "chunk": i}))
        print(f"  {len(pieces):4d} chunks  ←  {path.name}")
    return files, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="count only, write nothing (chromadb not required)")
    args = parser.parse_args()

    print(f"Corpus: {config.KNOWLEDGE_DIR}/")
    files, records = collect_chunks()
    print(f"\n{len(files)} files  →  {len(records)} chunks "
          f"(~{config.KNOWLEDGE_CHUNK_CHARS} chars, "
          f"{config.KNOWLEDGE_CHUNK_OVERLAP} overlap)")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    # import chromadb here → --dry-run works without it installed
    import chromadb
    from chromadb.utils import embedding_functions

    # MUST use the same embedding function as server.py (all-MiniLM on GPU),
    # otherwise stored vectors are incompatible at query time.
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.KNOWLEDGE_EMBED_MODEL, device="cuda")

    client = chromadb.PersistentClient(path=str(config.KNOWLEDGE_DB_PATH))
    # rebuild from scratch: discard existing collection (idempotent re-index)
    try:
        client.delete_collection(name=config.KNOWLEDGE_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.KNOWLEDGE_COLLECTION, embedding_function=ef)

    # add in batches (ChromaDB embeds automatically via all-MiniLM)
    BATCH = 500
    for start in range(0, len(records), BATCH):
        batch = records[start:start + BATCH]
        collection.add(
            ids=[r[0] for r in batch],
            documents=[r[1] for r in batch],
            metadatas=[r[2] for r in batch],
        )
        print(f"  embedded {min(start + BATCH, len(records))}/{len(records)}")

    print(f"\nCollection '{config.KNOWLEDGE_COLLECTION}' "
          f"with {collection.count()} chunks at {config.KNOWLEDGE_DB_PATH}/")


if __name__ == "__main__":
    main()
