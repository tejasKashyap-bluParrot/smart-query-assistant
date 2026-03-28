#!/usr/bin/env python3
"""
02_build_rag.py
===============
Step 2 of the YouTube RAG pipeline  —  run this ONCE after 01_fetch_transcripts.py.

Loads   youtube/transcripts.json
Chunks  each transcript with RecursiveCharacterTextSplitter
Embeds  chunks using sentence-transformers/LaBSE
Stores  everything in a persistent ChromaDB collection

Safe to re-run: videos already in the DB are automatically skipped.

Run:
    conda activate bluParrot
    python youtube/02_build_rag.py
"""

import json
import os
import warnings
import logging
from pathlib import Path

# ── Silence noisy third-party output ──────────────────────────────────────────
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Paths ────────────────────────────────────────────────────────────────────

THIS_DIR       = Path(__file__).parent
TRANSCRIPTS    = THIS_DIR / "transcripts.json"
CHROMA_DIR     = str(THIS_DIR / "chroma_db")
EMBED_MODEL_DIR= str(THIS_DIR / "embedding_model")

# ─── Config ───────────────────────────────────────────────────────────────────

EMBED_MODEL     = "sentence-transformers/LaBSE"
COLLECTION_NAME = "genai_playlist"


# ─── Embedding model ──────────────────────────────────────────────────────────

def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Load LaBSE — a multilingual sentence embedding model.
    Downloaded once to EMBED_MODEL_DIR; served from cache on subsequent runs.
    """
    os.environ["HF_HOME"] = EMBED_MODEL_DIR
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ─── Vector store ─────────────────────────────────────────────────────────────

def build_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Open (or create) the persistent ChromaDB collection.
    The collection lives on disk at CHROMA_DIR and survives process restarts.
    """
    return Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )


# ─── Transcript processing ────────────────────────────────────────────────────

def split(text_transcript: str) -> list[str]:
    """
    Split a full transcript into overlapping chunks.

    chunk_size=1200  — large enough to hold a complete idea/explanation
    chunk_overlap=250 — overlap preserves context at chunk boundaries
    separators       — prefer natural paragraph/sentence breaks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", "।", " ", ""],
    )
    return splitter.split_text(text_transcript)


def split2docs(chunks: list[str], metadata: dict) -> list[Document]:
    """
    Convert plain text chunks into LangChain Document objects.
    Each document carries the video's metadata for retrieval-time filtering.
    """
    return [
        Document(page_content=chunk, metadata=metadata)
        for chunk in chunks
    ]


def video_already_ingested(vector_store: Chroma, video_id: str) -> bool:
    """
    Check whether this video's chunks are already in the ChromaDB collection.
    Prevents duplicate embeddings on re-runs.
    """
    result = vector_store.get(
        where={"video_id": video_id},
        limit=1,
    )
    return len(result["ids"]) > 0


def process_video(video: dict, vector_store: Chroma) -> None:
    """
    Full pipeline for a single video:
      1. Skip if already in DB
      2. Split transcript into chunks
      3. Convert chunks to Documents with metadata
      4. Add to ChromaDB
    """
    video_id = video["video_id"]
    title    = video["title"][:55]

    # ── Guard: skip duplicates ─────────────────────────────────────────────
    if video_already_ingested(vector_store, video_id):
        print(f"  [skip]  {video_id}  {title}")
        return

    # ── Metadata attached to every chunk of this video ────────────────────
    # Crisp 6-field metadata — immediately readable by a human
    metadata = {
        "video_id":        video["video_id"],
        "title":           video["title"],
        "url":             video["url"],
        "duration_min":    video["duration_min"],
        "channel":         video["channel"],
        "transcript_lang": video.get("transcript_lang", "unknown"),
    }

    # ── Split → Docs → Store ──────────────────────────────────────────────
    chunks = split(video["transcript"])
    docs   = split2docs(chunks, metadata)
    vector_store.add_documents(docs)

    print(f"  [added] {video_id}  {title}  ({len(chunks)} chunks)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load transcripts ──────────────────────────────────────────────────────
    if not TRANSCRIPTS.exists():
        print(f"✗  {TRANSCRIPTS} not found.")
        print("   Run 01_fetch_transcripts.py first.")
        return

    with open(TRANSCRIPTS, "r") as f:
        videos = json.load(f)

    print(f"Loaded {len(videos)} transcripts from {TRANSCRIPTS}\n")

    # ── Init embeddings + DB ──────────────────────────────────────────────────
    print(f"Loading embedding model: {EMBED_MODEL}")
    print(f"  HF_HOME  → {EMBED_MODEL_DIR}")
    embeddings   = build_embeddings()
    vector_store = build_vector_store(embeddings)
    print(f"ChromaDB   → {CHROMA_DIR}  (collection: {COLLECTION_NAME!r})\n")

    # ── Process each video ────────────────────────────────────────────────────
    added   = 0
    skipped = 0

    for video in videos:
        was_ingested = video_already_ingested(vector_store, video["video_id"])
        process_video(video, vector_store)
        if was_ingested:
            skipped += 1
        else:
            added += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total_docs = vector_store._collection.count()
    print(f"\n{'─'*55}")
    print(f"  Added   : {added} new video(s)")
    print(f"  Skipped : {skipped} already-ingested video(s)")
    print(f"  Total   : {total_docs} chunks in ChromaDB")
    print(f"{'─'*55}")
    print("\nDone. ChromaDB is ready for pipeline.py to use.")


if __name__ == "__main__":
    main()
