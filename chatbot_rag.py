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
import logging
import warnings
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Suppress transformers/torchvision import warnings (harmless)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Load Chatbot/.env first, then fall back to root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(find_dotenv())

# Optional heavy imports — the app still works without them
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # Note: EnsembleRetriever has pydantic_v1 compatibility issues, we'll implement hybrid search manually
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"[RAG] WARNING: RAG dependencies missing: {e}")
    print("[RAG] Fix: pip install langchain-community langchain-text-splitters rank-bm25")

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
_embeddings_cache = None  # Cache embeddings model (loaded once)

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
    """Return the best available embedding model (prioritized by quality & speed).
    
    CACHED in memory - loaded only ONCE, then reused.
    
    1. Sentence-Transformers (best, free, local)
    2. Google (needs GOOGLE_API_KEY)
    3. Hash (poor, offline fallback)
    """
    global _embeddings_cache
    
    # Return cached if already loaded
    if _embeddings_cache is not None:
        return _embeddings_cache
    
    # Try sentence-transformers first (best quality, free, local)
    try:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: Fast, good quality (384 dims)
        # Production choice: balances speed and accuracy
        print("[RAG] Loading embeddings model (all-MiniLM-L6-v2, CACHED in memory)")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Wrap in LangChain-compatible class
        class SentenceTransformerEmbeddings(Embeddings):
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.model.encode(texts, convert_to_numpy=True).tolist()
            
            def embed_query(self, text: str) -> list[float]:
                return self.model.encode(text, convert_to_numpy=True).tolist()
        
        _embeddings_cache = SentenceTransformerEmbeddings(model)
        print("[RAG] ✅ Embeddings model cached and ready")
        return _embeddings_cache
    except ImportError:
        print("[RAG] ERROR: sentence-transformers not installed. REQUIRED: pip install sentence-transformers")
    
    # Try Google second
    if GOOGLE_EMBEDDINGS_AVAILABLE:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
            print("[RAG] Using Google Generative AI embeddings (high quality, costs $)")
            _embeddings_cache = GoogleGenerativeAIEmbeddings(model=model)
            return _embeddings_cache
    
    # Fallback to hash (poor quality)
    print("[RAG] WARNING: Using hash embeddings (poor quality, offline fallback)")
    _embeddings_cache = _HashEmbeddings()
    return _embeddings_cache

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
    
    # Remove header/footer artifacts before returning
    cleaned_docs = []
    for doc in docs:
        lines = doc.page_content.split('\n')
        # Remove common PDF artifacts
        filtered_lines = []
        for line in lines:
            line_lower = line.lower()
            # Skip header/footer spam
            if any(x in line_lower for x in ["free pdf", "amanai lab", "amanailab.com", "youtube", "linkedin", "github"]):
                continue
            if line.strip():
                filtered_lines.append(line)
        if filtered_lines:
            doc.page_content = '\n'.join(filtered_lines)
            cleaned_docs.append(doc)
    
    if not docs:
        print(f"[RAG] WARNING: No documents loaded from {len(files)} file(s)")
    
    return cleaned_docs

def _extract_section_from_text(text: str) -> str:
    """
    Extract section number from text.
    Matches patterns like:
      - "1. Introduction"
      - "2.1 Model Architecture"
      - "Stage 3: Training Setup"
      - "Section 5.2: Advanced Topics"
    """
    patterns = [
        r"^(\d+(?:\.\d+)?)\s+[A-Z]",  # "1. Title" or "2.1 Title"
        r"^Stage\s+(\d+)",  # "Stage 3"
        r"^Section\s+(\d+(?:\.\d+)?)",  # "Section 2.1"
    ]
    
    for line in text.split('\n')[:3]:  # Check first 3 lines only
        line = line.strip()
        if not line:
            continue
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    
    return ""

# ══════════════════════════════════════════════════════════════════════════
# INDEX BUILDING & RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════

def _build_retriever(thread_id: str, force_rebuild: bool = False):
    """
    Build (or load from disk) a hybrid retriever for a thread.
    
    Hybrid retrieval combines:
    1. Semantic search (FAISS + embeddings) — catches meaning-based queries
    2. BM25 keyword search — catches exact term matches
    3. Reranking — scores chunks by relevance to query
    
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
            print(f"[RAG] ✅ Loading existing index for thread {tid}")
            vectorstore = FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # Rebuild BM25 from raw documents (not persisted, rebuilt on load)
            docs = _load_documents(files)
            if docs:
                print(f"[RAG] Building BM25 from {len(docs)} documents")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)
                print(f"[RAG] Split into {len(chunks)} chunks")
                keyword_retriever = BM25Retriever.from_documents(chunks)
                
                # Create hybrid retriever (same as fresh build)
                class OptimizedHybridRetriever:
                    def __init__(self, semantic, keyword, all_chunks):
                        self.semantic = semantic
                        self.keyword = keyword
                        self.all_chunks = all_chunks
                    
                    def get_relevant_documents(self, query):
                        sem_docs = []
                        kw_docs = []
                        
                        if not isinstance(query, str):
                            query = str(query)
                        
                        try:
                            result = self.semantic.invoke(query)
                            sem_docs = result if isinstance(result, list) else [result]
                        except Exception as e:
                            logger.debug(f"FAISS failed: {e}")
                        
                        try:
                            result = self.keyword.invoke(query)
                            kw_docs = result if isinstance(result, list) else [result]
                        except Exception as e:
                            logger.debug(f"BM25 failed: {e}")
                        
                        combined = {}
                        for i, doc in enumerate(sem_docs[:5]):
                            doc_id = hash(doc.page_content[:100])
                            score = (1.0 - i/5) * 0.8
                            combined[doc_id] = (score, doc)
                        
                        for i, doc in enumerate(kw_docs[:5]):
                            doc_id = hash(doc.page_content[:100])
                            score = (1.0 - i/5) * 0.2
                            if doc_id in combined:
                                old_score, old_doc = combined[doc_id]
                                combined[doc_id] = (old_score + score, old_doc)
                            else:
                                combined[doc_id] = (score, doc)
                        
                        sorted_docs = sorted(combined.values(), key=lambda x: x[0], reverse=True)
                        return [doc for _, doc in sorted_docs[:3]]
                    
                    def invoke(self, input_dict):
                        query = input_dict.get("query", input_dict) if isinstance(input_dict, dict) else input_dict
                        return self.get_relevant_documents(query)
                
                hybrid_retriever = OptimizedHybridRetriever(semantic_retriever, keyword_retriever, chunks)
                return hybrid_retriever

        except Exception as e:
            print(f"[RAG] Could not load FAISS index, rebuilding: {e}")

    # Build a fresh index from the uploaded documents
    print(f"[RAG] Building fresh index from {len(files)} file(s)")
    docs = _load_documents(files)
    
    if not docs:
        print(f"[RAG] ERROR: No documents were loaded!")
        return None

    print(f"[RAG] Loaded {len(docs)} documents, splitting into chunks...")
    # OPTIMIZED: 1000 chars with 150 overlap = tight, focused chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    # Add section metadata to each chunk
    for chunk in chunks:
        if not chunk.metadata.get("section"):
            section = _extract_section_from_text(chunk.page_content)
            if section:
                chunk.metadata["section"] = section
    
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
        
        # OPTIMIZED: Hybrid retriever with reranking
        print("[RAG] Creating hybrid retriever with reranking (FAISS + BM25 + MMR)")
        
        class OptimizedHybridRetriever:
            def __init__(self, semantic, keyword, all_chunks):
                self.semantic = semantic
                self.keyword = keyword
                self.all_chunks = all_chunks
            
            def _score_relevance(self, query: str, chunk_text: str) -> float:
                """Score chunk relevance to query (0-1 scale)"""
                query_lower = query.lower()
                chunk_lower = chunk_text.lower()
                
                # Exact match boost
                if query_lower in chunk_lower:
                    return 0.9
                
                # Keyword match count
                query_words = set(query_lower.split())
                chunk_words = chunk_lower.split()
                matches = sum(1 for word in query_words if word in chunk_words)
                keyword_score = min(matches / len(query_words), 1.0) if query_words else 0
                
                # Length penalty (prefer concise, focused chunks)
                length_score = 1.0 / (1.0 + (len(chunk_text) / 1500))
                
                return 0.6 * keyword_score + 0.4 * length_score
            
            def get_relevant_documents(self, query):
                sem_docs = []
                kw_docs = []
                
                if not isinstance(query, str):
                    query = str(query)
                
                # Get both FAISS and BM25 results
                try:
                    result = self.semantic.invoke(query)
                    sem_docs = result if isinstance(result, list) else [result]
                except Exception as e:
                    logger.debug(f"FAISS failed: {e}")
                
                try:
                    result = self.keyword.invoke(query)
                    kw_docs = result if isinstance(result, list) else [result]
                except Exception as e:
                    logger.debug(f"BM25 failed: {e}")
                
                # Combine: FAISS 80%, BM25 20% (prioritize semantic)
                combined = {}
                for i, doc in enumerate(sem_docs[:5]):  # Top 5 from FAISS
                    doc_id = hash(doc.page_content[:100])
                    score = (1.0 - i/5) * 0.8  # 80% weight for FAISS
                    combined[doc_id] = (score, doc)
                
                for i, doc in enumerate(kw_docs[:5]):  # Top 5 from BM25
                    doc_id = hash(doc.page_content[:100])
                    score = (1.0 - i/5) * 0.2  # 20% weight for BM25
                    if doc_id in combined:
                        old_score, old_doc = combined[doc_id]
                        combined[doc_id] = (old_score + score, old_doc)
                    else:
                        combined[doc_id] = (score, doc)
                
                # Sort by combined score
                sorted_docs = sorted(combined.values(), key=lambda x: x[0], reverse=True)
                # Return top 3 chunks (more complete answers while maintaining quality)
                # Still returns best-ranked chunks, provides better full context
                return [doc for _, doc in sorted_docs[:3]]
            
            def invoke(self, input_dict):
                """Support .invoke() method for compatibility"""
                query = input_dict.get("query", input_dict) if isinstance(input_dict, dict) else input_dict
                return self.get_relevant_documents(query)
        
        hybrid_retriever = OptimizedHybridRetriever(semantic_retriever, keyword_retriever, chunks)

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

    print(f"[RAG] Optimized hybrid retriever built successfully!")
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
    """Fast RAG retrieval - return top 3 chunks with clean structured formatting."""
    if not query.strip():
        return ""

    tid = _safe_id(thread_id)
    
    # Get cached retriever
    if tid not in _retriever_cache:
        _retriever_cache[tid] = _build_retriever(tid)

    retriever = _retriever_cache.get(tid)
    if retriever is None:
        return ""

    # Retrieve documents
    try:
        all_docs = retriever.invoke(query)
    except Exception as e:
        print(f"[RAG] Error: {e}")
        return ""

    if not all_docs:
        return ""

    docs = all_docs[:3]  # Top 3 chunks

    # Format chunks with clean structure
    snippets = []
    for i, doc in enumerate(docs, start=1):
        text = doc.page_content.strip()
        
        # Clean up: remove decorative lines, excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip: empty, decorative (■, –), or too short
            if not line or line.startswith('■') or line.startswith('–'):
                continue
            if len(line) < 3:
                continue
            cleaned_lines.append(line)
        
        # Rejoin with single newlines
        text = '\n'.join(cleaned_lines)
        
        # Structure: break into paragraphs for readability
        paragraphs = text.split('\n\n')
        structured_text = '\n\n'.join(p.strip() for p in paragraphs if p.strip())
        
        # Truncate if too long
        if len(structured_text) > 2000:
            structured_text = structured_text[:2000]
            # Find last sentence boundary
            last_dot = structured_text.rfind('.')
            if last_dot > 1500:
                structured_text = structured_text[:last_dot+1]
        
        snippets.append(f"[{i}] {structured_text}")

    context = "\n\n".join(snippets)
    
    # Add sources footer
    if docs:
        context += "\n\n---\n\n**Sources:**\n"
        seen = set()
        for i, doc in enumerate(docs, start=1):
            source = Path(str(doc.metadata.get("source", "unknown"))).name
            page = doc.metadata.get("page")
            
            source_line = f"[{i}] {source}"
            if isinstance(page, int):
                source_line += f" — Page {page + 1}"
            
            if source_line not in seen:
                context += source_line + "\n"
                seen.add(source_line)
    
    return context

def has_documents(thread_id: str) -> bool:
    """Return True if this thread has any uploaded documents."""
    return len(_get_supported_files(_safe_id(thread_id))) > 0
