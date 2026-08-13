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
    from langchain_community.retrievers import BM25Retriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.retrievers import EnsembleRetriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    GOOGLE_EMBEDDINGS_AVAILABLE = False

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.embeddings.base import Embeddings


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


class _HashEmbeddings(Embeddings):
    """
    A simple offline embedding model — no API key needed.
    Inherits from LangChain's Embeddings base class so FAISS accepts it.
    Converts text to a fixed-size vector using word hashes.
    """

    DIM = 384

    def _embed(self, text: str) -> list[float]:
        import hashlib
        import numpy as np
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            idx = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
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
                try:
                    loader = PyPDFLoader(str(path))
                    loaded = loader.load()
                    if loaded:
                        docs.extend(loaded)
                        print(f"[RAG] Loaded PDF: {path.name} ({len(loaded)} pages)")
                    else:
                        print(f"[RAG] Warning: PDF loaded but no pages found: {path.name}")
                except Exception as pdf_error:
                    print(f"[RAG] PDF Loading Error for {path.name}: {pdf_error}")
                    # Try alternative: read as text
                    try:
                        import PyPDF2
                        with open(path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() + "\n"
                        if text.strip():
                            from langchain_core.documents import Document
                            docs.append(Document(page_content=text, metadata={"source": str(path.name)}))
                            print(f"[RAG] Loaded PDF (fallback): {path.name}")
                    except Exception as fallback_error:
                        print(f"[RAG] Fallback also failed for {path.name}: {fallback_error}")
            else:
                loader = TextLoader(str(path), encoding="utf-8")
                loaded = loader.load()
                docs.extend(loaded)
                print(f"[RAG] Loaded text file: {path.name}")
        except Exception as e:
            print(f"[RAG] Could not load {path.name}: {e}")
    
    if not docs:
        print(f"[RAG] WARNING: No documents loaded from {len(files)} file(s)")
    
    return docs


# ══════════════════════════════════════════════════════════════════════════
# INDEX BUILDING & RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════

def _build_retriever(thread_id: str, force_rebuild: bool = False):
    """
    Build (or load from disk) a hybrid retriever for a thread.
    
    Hybrid retrieval combines:
    1. Semantic search (FAISS + embeddings) — catches meaning-based queries
    2. BM25 keyword search — catches exact term matches
    
    Returns None if there are no documents or RAG is not available.
    """
    if not RAG_AVAILABLE:
        print("[RAG] ERROR: RAG not available (missing dependencies)")
        return None

    tid = _safe_id(thread_id)
    index_dir = get_index_dir(tid)
    files = _get_supported_files(tid)

    if not files:
        print(f"[RAG] No documents found for thread {tid}")
        return None  # No documents uploaded yet

    embeddings = _get_embeddings()

    # Load existing index from disk if it exists and we are not forcing a rebuild
    if index_dir.exists() and not force_rebuild:
        try:
            print(f"[RAG] Loading existing index for thread {tid}")
            vectorstore = FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            # Rebuild BM25 from raw documents (not persisted, rebuilt on load)
            docs = _load_documents(files)
            if docs:
                print(f"[RAG] Building BM25 from {len(docs)} documents")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)
                print(f"[RAG] Split into {len(chunks)} chunks")
                keyword_retriever = BM25Retriever.from_documents(chunks)
                # Combine semantic (60%) and keyword (40%)
                print("[RAG] Creating hybrid retriever (FAISS 60% + BM25 40%)")
                return EnsembleRetriever(
                    retrievers=[semantic_retriever, keyword_retriever],
                    weights=[0.6, 0.4]
                )
            return semantic_retriever
        except Exception as e:
            print(f"[RAG] Could not load FAISS index, rebuilding: {e}")

    # Build a fresh index from the uploaded documents
    print(f"[RAG] Building fresh index from {len(files)} file(s)")
    docs = _load_documents(files)
    
    if not docs:
        print(f"[RAG] ERROR: No documents were loaded!")
        return None

    print(f"[RAG] Loaded {len(docs)} documents, splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        print(f"[RAG] ERROR: No chunks created from documents!")
        return None

    print(f"[RAG] Created {len(chunks)} chunks, building FAISS index...")

    try:
        # Build FAISS for semantic search
        vectorstore = FAISS.from_documents(chunks, embeddings)
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Build BM25 for keyword search
        print("[RAG] Building BM25 keyword retriever...")
        keyword_retriever = BM25Retriever.from_documents(chunks)
        
        # Combine both: 60% semantic, 40% keyword
        print("[RAG] Creating hybrid retriever (FAISS 60% + BM25 40%)")
        hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever],
            weights=[0.6, 0.4]
        )
    except Exception as e:
        print(f"[RAG] Hybrid retriever build error: {e}")
        return None

    # Save FAISS to disk so we can reload next time without rebuilding
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_dir))
        print(f"[RAG] Index saved to {index_dir}")
    except Exception as e:
        print(f"[RAG] Warning: Could not save index: {e}")

    print(f"[RAG] ✓ Hybrid retriever built successfully!")
    return hybrid_retriever


def rebuild_rag_index(thread_id: str) -> str:
    """Force a full rebuild of the FAISS index and update the cache."""
    tid = _safe_id(thread_id)
    
    # Check if files exist first
    files = _get_supported_files(tid)
    if not files:
        return "Error: No supported files found (PDF, TXT, MD)"
    
    print(f"[RAG] Found {len(files)} file(s) to index")
    for f in files:
        print(f"[RAG]   - {f.name}")
    
    retriever = _build_retriever(tid, force_rebuild=True)
    _retriever_cache[tid] = retriever

    if retriever:
        return f"RAG index rebuilt successfully with {len(files)} file(s)."
    return "RAG index rebuild failed — check console for errors above"


def _extract_filename_from_query(query: str, thread_id: str) -> str:
    """
    Check if the user mentioned a specific filename in their query.
    Returns the matching filename if found, empty string otherwise.

    Examples:
      "give me summary of A2_Solution.pdf"  -> "A2_Solution.pdf"
      "what does notes.txt say"             -> "notes.txt"
      "summarize everything"                -> ""
    """
    files = _get_supported_files(_safe_id(thread_id))
    q_lower = query.lower()
    for f in files:
        # Match full filename (e.g. "a2_solution.pdf") or stem only (e.g. "a2_solution")
        if f.name.lower() in q_lower or f.stem.lower() in q_lower:
            return f.name
    return ""


def get_rag_context(query: str, thread_id: str, filename_filter: str = "") -> str:
    """
    Search the document index and return the most relevant text chunks.

    If the user mentioned a specific filename (e.g. "summarize A2_Solution.pdf"),
    only chunks from that file are returned — so the answer won't mix in other docs.

    If no filename is mentioned, returns the top-4 most relevant chunks from all docs.

    The result is formatted like:
      [1] filename.pdf (page 1): ... chunk text ...
      [2] filename.pdf (page 2): ... chunk text ...

    These [1], [2] tags are used as citations in the final answer.
    Returns an empty string if no documents are indexed.
    """
    if not query.strip():
        return ""

    tid = _safe_id(thread_id)

    # Auto-detect filename from query if not explicitly passed
    if not filename_filter:
        filename_filter = _extract_filename_from_query(query, tid)

    if filename_filter:
        print(f"[RAG] filename filter active: {filename_filter}")

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

    # Filter to only the requested file if one was detected
    if filename_filter:
        filtered = [
            doc for doc in docs
            if Path(str(doc.metadata.get("source", ""))).name.lower() == filename_filter.lower()
        ]
        # Keep top 4 from filtered; fall back to all docs only if filter returned nothing
        docs = filtered[:4] if filtered else docs[:4]
    else:
        docs = docs[:4]

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
