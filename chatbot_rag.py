from __future__ import annotations

import hashlib
import json
import os
import re
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except Exception:
    FAISS = None
    PyPDFLoader = None
    TextLoader = None

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RAG_DOCS_ROOT = BASE_DIR / "knowledge_base"
RAG_INDEX_ROOT = BASE_DIR / "faiss_index"

rag_retriever_cache: dict[str, object] = {}
rag_status_cache: dict[str, str] = {}

active_thread_id = ContextVar("active_thread_id", default="default")


def _safe_thread_id(thread_id: str) -> str:
    value = str(thread_id or "default").strip()
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value or "default"


def get_thread_rag_docs_dir(thread_id: str) -> Path:
    return RAG_DOCS_ROOT / _safe_thread_id(thread_id)


def get_thread_rag_index_dir(thread_id: str) -> Path:
    return RAG_INDEX_ROOT / _safe_thread_id(thread_id)


def _get_rag_status(thread_id: str) -> str:
    return rag_status_cache.get(_safe_thread_id(thread_id), "RAG not initialized")


def _set_rag_status(thread_id: str, status: str) -> None:
    rag_status_cache[_safe_thread_id(thread_id)] = status


class HashEmbeddings(Embeddings):
    """Deterministic local embeddings fallback for FAISS without external API calls."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tolist()

        for token in tokens:
            vec[self._hash_token(token)] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _embedding_from_backend(backend: str):
    if backend == "google":
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model_name), {"backend": "google", "model": model_name}

    dim = int(os.getenv("RAG_HASH_EMBEDDING_DIM", "384"))
    return HashEmbeddings(dim=dim), {"backend": "hash", "dim": dim}


def _save_rag_meta(meta: dict, thread_id: str) -> None:
    index_dir = get_thread_rag_index_dir(thread_id)
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "meta.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)
    except Exception as err:
        print(f"RAG meta save warning: {err}")


def _load_rag_meta(thread_id: str) -> dict:
    meta_path = get_thread_rag_index_dir(thread_id) / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _collect_rag_files(thread_id: str) -> list[Path]:
    supported = {".txt", ".md", ".pdf"}
    files: list[Path] = []

    docs_dir = get_thread_rag_docs_dir(thread_id)
    if docs_dir.exists():
        files.extend([p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported])

    unique_files = list(dict.fromkeys(files))
    return sorted(unique_files)


def _load_documents(paths: list[Path]) -> list[Document]:
    documents: list[Document] = []

    for path in paths:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                if PyPDFLoader is None:
                    continue
                loader = PyPDFLoader(str(path))
                documents.extend(loader.load())
            else:
                if TextLoader is None:
                    continue
                loader = TextLoader(str(path), encoding="utf-8")
                documents.extend(loader.load())
        except Exception as err:
            name = getattr(path, "name", str(path))
            print(f"RAG load warning for {name}: {err}")

    return documents


def _build_rag_retriever(thread_id: str, force_rebuild: bool = False):
    thread_key = _safe_thread_id(thread_id)
    index_dir = get_thread_rag_index_dir(thread_key)
    docs_dir = get_thread_rag_docs_dir(thread_key)

    if FAISS is None:
        _set_rag_status(thread_key, "RAG unavailable: install faiss-cpu and document loaders")
        return None

    files = _collect_rag_files(thread_key)
    if not files:
        _set_rag_status(thread_key, f"RAG knowledge base is empty for this chat ({docs_dir})")
        return None

    preferred_backend = os.getenv("RAG_EMBEDDING_BACKEND", "hash").lower()

    try:
        if index_dir.exists() and not force_rebuild:
            meta = _load_rag_meta(thread_key)
            load_backend = meta.get("backend", preferred_backend)
            embeddings, _ = _embedding_from_backend(load_backend)
            vectorstore = FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            _set_rag_status(thread_key, f"RAG ready (loaded index, {len(files)} files, backend={load_backend})")
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as err:
        print(f"RAG index load warning, rebuilding: {err}")

    docs = _load_documents(files)
    if not docs:
        _set_rag_status(thread_key, "RAG could not load readable documents")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        _set_rag_status(thread_key, "RAG found no chunks after splitting")
        return None

    embeddings, meta = _embedding_from_backend(preferred_backend)

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as err:
        _set_rag_status(thread_key, f"RAG embedding/index build failed: {err}")
        return None

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    meta.update({"files_indexed": len(files), "chunks": len(chunks)})
    _save_rag_meta(meta, thread_key)

    status = (
        f"RAG ready (built index with {len(chunks)} chunks from {len(files)} files, "
        f"backend={meta.get('backend')})"
    )
    _set_rag_status(thread_key, status)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def ensure_rag_ready(thread_id: str, force_rebuild: bool = False):
    thread_key = _safe_thread_id(thread_id)

    if thread_key not in rag_retriever_cache or force_rebuild:
        rag_retriever_cache[thread_key] = _build_rag_retriever(thread_key, force_rebuild=force_rebuild)

    return rag_retriever_cache.get(thread_key)


def rebuild_rag_index(thread_id: str) -> str:
    thread_key = _safe_thread_id(thread_id)
    retriever = ensure_rag_ready(thread_key, force_rebuild=True)
    if retriever is None:
        return f"RAG rebuild failed: {_get_rag_status(thread_key)}"
    return f"RAG rebuild successful: {_get_rag_status(thread_key)}"


def _rag_context_for_query(query: str, thread_id: str, k: int = 4) -> str:
    if not query.strip():
        return ""

    retriever = ensure_rag_ready(thread_id)
    if retriever is None:
        return ""

    try:
        docs = retriever.invoke(query)
    except Exception:
        return ""

    if not docs:
        return ""

    snippets = []
    for i, doc in enumerate(docs[:k], start=1):
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        page_info = f" (page {page + 1})" if isinstance(page, int) else ""
        chunk = doc.page_content.strip().replace("\n", " ")
        snippets.append(f"[{i}] {source}{page_info}: {chunk}")

    return "\n\n".join(snippets)


def _extract_rag_source_tags(rag_context: str) -> list[str]:
    tags = re.findall(r"\[(\d+)\]", rag_context or "")
    seen = set()
    ordered = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        ordered.append(f"[{tag}]")
    return ordered


def _ensure_rag_citations(text: str, rag_context: str) -> str:
    if not rag_context:
        return text

    if re.search(r"\[\d+\]", text or ""):
        return text

    tags = _extract_rag_source_tags(rag_context)
    if not tags:
        return text

    suffix = "Sources: " + ", ".join(tags)
    if text.strip().endswith("."):
        return f"{text}\n\n{suffix}"
    return f"{text}\n\n{suffix}"
