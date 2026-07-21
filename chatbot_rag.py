"""
chatbot_rag.py
--------------
RAG (Retrieval-Augmented Generation) helpers.

What this file does:
1. Lets each chat thread have its own folder of uploaded documents (PDF, TXT, MD).
2. Builds a FAISS vector index from those documents so we can search them.
3. Given a user question, finds the most relevant chunks and returns them as context.

Folder layout:
  knowledge_base/<thread_id>/   ← uploaded documents
  faiss_index/<thread_id>/      ← saved FAISS index for that thread
"""

import os
import re
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Optional heavy imports — the app still works without them
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    GOOGLE_EMBEDDINGS_AVAILABLE = False


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DOCS_ROOT = BASE_DIR / "knowledge_base"   # uploaded files go here
INDEX_ROOT = BASE_DIR / "faiss_index"     # FAISS indexes saved here

# Simple in-memory caches so we don't rebuild the index on every question
_retriever_cache: dict[str, object] = {}


# ══════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _safe_id(thread_id: str) -> str:
    """Remove unsafe characters from a thread ID so it can be used as a folder name.
    Hyphens are kept since UUIDs use them and they are safe on all filesystems."""
    value = str(thread_id or "default").strip()
    return re.sub(r"[^A-Za-z0-9._\-]", "_", value) or "default"


def get_docs_dir(thread_id: str) -> Path:
    """Return the folder where uploaded documents for this thread are stored."""
    return DOCS_ROOT / _safe_id(thread_id)


def get_index_dir(thread_id: str) -> Path:
    """Return the folder where the FAISS index for this thread is stored."""
    return INDEX_ROOT / _safe_id(thread_id)


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDING BACKEND
# Uses Google embeddings if available, otherwise falls back to a simple
# hash-based embedding that needs no API key.
# ══════════════════════════════════════════════════════════════════════════

def _get_embeddings():
    """Return the best available embedding model."""
    backend = os.getenv("RAG_EMBEDDING_BACKEND", "hash").lower()

    if backend == "google" and GOOGLE_EMBEDDINGS_AVAILABLE:
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model)

    # Fallback: local hash embeddings (no API key needed)
    return _HashEmbeddings()


class _HashEmbeddings:
    """
    A very simple embedding model that works offline with no API key.
    It converts text into a vector using word hashes.
    Good enough for small document sets.
    """

    DIM = 384  # vector size

    def _embed(self, text: str) -> list[float]:
        import hashlib, numpy as np
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            idx = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        # Normalize so all vectors have length 1
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════════════════

def _get_supported_files(thread_id: str) -> list[Path]:
    """Return all PDF, TXT, and MD files uploaded for this thread."""
    docs_dir = get_docs_dir(thread_id)
    if not docs_dir.exists():
        return []
    supported = {".pdf", ".txt", ".md"}
    return sorted(p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported)


def _load_documents(files: list[Path]):
    """Load each file into LangChain Document objects."""
    docs = []
    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Could not load {path.name}: {e}")
    return docs


# ══════════════════════════════════════════════════════════════════════════
# INDEX BUILDING & RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════

def _build_retriever(thread_id: str, force_rebuild: bool = False):
    """
    Build (or load from disk) the FAISS retriever for a thread.
    Returns None if there are no documents or RAG is not available.
    """
    if not RAG_AVAILABLE:
        return None

    tid = _safe_id(thread_id)
    index_dir = get_index_dir(tid)
    files = _get_supported_files(tid)

    if not files:
        return None  # No documents uploaded yet

    embeddings = _get_embeddings()

    # Load existing index from disk if it exists and we are not forcing a rebuild
    if index_dir.exists() and not force_rebuild:
        try:
            vectorstore = FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )
            return vectorstore.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            print(f"Could not load FAISS index, rebuilding: {e}")

    # Build a fresh index from the uploaded documents
    docs = _load_documents(files)
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"FAISS build error: {e}")
        return None

    # Save to disk so we can reload next time without rebuilding
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def rebuild_rag_index(thread_id: str) -> str:
    """Force a full rebuild of the FAISS index and update the cache."""
    tid = _safe_id(thread_id)
    retriever = _build_retriever(tid, force_rebuild=True)
    _retriever_cache[tid] = retriever

    files = _get_supported_files(tid)
    if retriever:
        return f"RAG index rebuilt successfully with {len(files)} file(s)."
    return "RAG index rebuild failed — no documents found or load error."


def get_rag_context(query: str, thread_id: str) -> str:
    """
    Search the document index and return the most relevant text chunks.

    The result is formatted like:
      [1] filename.pdf (page 1): ... chunk text ...
      [2] notes.txt: ... chunk text ...

    These [1], [2] tags are used as citations in the final answer.
    Returns an empty string if no documents are indexed.
    """
    if not query.strip():
        return ""

    tid = _safe_id(thread_id)

    # Build or reuse the cached retriever
    if tid not in _retriever_cache:
        _retriever_cache[tid] = _build_retriever(tid)

    retriever = _retriever_cache.get(tid)
    if retriever is None:
        return ""

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""

    if not docs:
        return ""

    snippets = []
    for i, doc in enumerate(docs, start=1):
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        page_info = f" (page {page + 1})" if isinstance(page, int) else ""
        text = doc.page_content.strip().replace("\n", " ")
        snippets.append(f"[{i}] {source}{page_info}: {text}")

    return "\n\n".join(snippets)


def has_documents(thread_id: str) -> bool:
    """Return True if this thread has any uploaded documents."""
    return len(_get_supported_files(_safe_id(thread_id))) > 0
