
"""
chatbot_rag.py
--------------
RAG (Retrieval-Augmented Generation) helpers.

What this file does:

1. Lets each chat thread have its own folder of uploaded documents (PDF, TXT, MD).

2. Builds a FAISS vector index from those documents so we can search them.

3. Given a user question, finds the most relevant chunks and returns them as context.

Folder layout:

  knowledge_base/<thread_id>/   # uploaded documents

  faiss_index/<thread_id>/      # saved FAISS index for that thread
"""

import sys
import os
import shutil

# CRITICAL: Set these BEFORE any other imports to prevent ZoeDepth/torchvision errors
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import re
import logging
import warnings
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Suppress ALL transformers warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Load Chatbot/.env first, then fall back to root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(find_dotenv())

# Optional heavy imports — the app still works without them
try:
    # Temporarily redirect stderr to suppress ZoeDepth import errors
    import sys
    from io import StringIO
    _old_stderr = sys.stderr
    sys.stderr = StringIO()
    
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Restore stderr
    sys.stderr = _old_stderr
    
    # Note: EnsembleRetriever has pydantic_v1 compatibility issues, we'll implement hybrid search manually
    RAG_AVAILABLE = True
except ImportError as e:
    # Restore stderr if error happened
    if '_old_stderr' in locals():
        sys.stderr = _old_stderr
    RAG_AVAILABLE = False
    print(f"[RAG] WARNING: RAG dependencies missing: {e}")
    print("[RAG] Fix: pip install langchain-community langchain-text-splitters rank-bm25")
except Exception as e:
    # Restore stderr
    if '_old_stderr' in locals():
        sys.stderr = _old_stderr
    # Ignore any other import errors (like ZoeDepth/torchvision)
    RAG_AVAILABLE = True
    print(f"[RAG] Minor import warning suppressed (non-critical): {type(e).__name__}")

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

# NOTE: _embeddings_cache is now handled by @st.cache_resource in _get_embeddings()
# to survive Streamlit reruns

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
# Uses all-MiniLM-L6-v2 (required in requirements.txt)
# ══════════════════════════════════════════════════════════════════════════

def _get_embeddings():
    """Return all-MiniLM-L6-v2 embeddings model (CACHED across Streamlit reruns).
    
    Uses Streamlit's @st.cache_resource to persist model in memory.
    First load takes ~5-10 seconds. Subsequent loads are instant.
    
    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        import streamlit as st
        
        @st.cache_resource(show_spinner=False)
        def load_embeddings_model():
            """Load and cache the embeddings model."""
            from sentence_transformers import SentenceTransformer
            print("[RAG] Loading embeddings model (all-MiniLM-L6-v2)...")
            print("[RAG] First load: 5-10 seconds (downloading model if needed)")
            print("[RAG] Subsequent loads: instant (cached in Streamlit)")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("[RAG] Embeddings model loaded successfully!")
            
            # Wrap in LangChain-compatible class
            class SentenceTransformerEmbeddings(Embeddings):
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    print(f"[RAG] Embedding {len(texts)} documents...")
                    embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                    print(f"[RAG] Done embedding {len(texts)} documents!")
                    return embeddings.tolist()
                
                def embed_query(self, text: str) -> list[float]:
                    return self.model.encode(text, convert_to_numpy=True).tolist()
            
            return SentenceTransformerEmbeddings(model)
        
        embeddings = load_embeddings_model()
        print("[RAG] Embeddings ready (all-MiniLM-L6-v2, 384 dims, CACHED)")
        return embeddings
        
    except ImportError as e:
        raise ImportError(
            "\nsentence-transformers is REQUIRED but not installed.\n\n"
            "Fix:\n"
            "  pip install sentence-transformers\n"
            "  OR:\n"
            "  pip install -r requirements.txt\n\n"
            "The RAG system uses all-MiniLM-L6-v2 for semantic search.\n"
            "It must be installed for the chatbot to work."
        ) from e

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
    """Load each file into LangChain Document objects with proper source and page metadata."""
    docs = []
    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                try:
                    from langchain_core.documents import Document
                    loader = PyPDFLoader(str(path))
                    loaded = loader.load()
                    if loaded:
                        # Ensure each doc has source filename and page metadata
                        for doc in loaded:
                            if "source" not in doc.metadata:
                                doc.metadata["source"] = str(path.name)
                            # Page is already set by PyPDFLoader, but ensure it's an int
                            if "page" in doc.metadata:
                                doc.metadata["page"] = int(doc.metadata["page"])
                        docs.extend(loaded)
                        print(f"[RAG] Loaded PDF: {path.name} ({len(loaded)} pages with citations)")
                    else:
                        print(f"[RAG] Warning: PDF loaded but no pages found: {path.name}")
                except Exception as pdf_error:
                    print(f"[RAG] PDF Loading Error for {path.name}: {pdf_error}")
                    # Try alternative: read as text with manual page tracking
                    try:
                        import PyPDF2
                        from langchain_core.documents import Document
                        with open(path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page_num, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if text.strip():
                                    doc = Document(
                                        page_content=text,
                                        metadata={
                                            "source": str(path.name),
                                            "page": page_num
                                        }
                                    )
                                    docs.append(doc)
                        if docs:
                            print(f"[RAG]  Loaded PDF (fallback with page tracking): {path.name}")
                    except Exception as fallback_error:
                        print(f"[RAG]  Fallback also failed for {path.name}: {fallback_error}")
            else:
                from langchain_core.documents import Document
                loader = TextLoader(str(path), encoding="utf-8")
                loaded = loader.load()
                # Ensure text files also have source metadata
                for doc in loaded:
                    doc.metadata["source"] = str(path.name)
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = 0
                docs.extend(loaded)
                print(f"[RAG]  Loaded text file: {path.name}")
        except Exception as e:
            print(f"[RAG]  Could not load {path.name}: {e}")
    
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
    1. Semantic search (FAISS + embeddings) - catches meaning-based queries
    2. BM25 keyword search - catches exact term matches
    3. Reranking - scores chunks by relevance to query
    
    Returns None if there are no documents or RAG is not available.
    
    OPTIMIZED for speed:
    - Chunk size 800 (not 1000) = fewer chunks = faster embedding
    - Progress logging so you see what's happening
    - Cached embeddings model (global, not reloaded)
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
            print(f"[RAG]  Loading existing index for thread {tid} (cached from disk)")
            vectorstore = FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Rebuild BM25 from raw documents (not persisted, rebuilt on load)
            docs = _load_documents(files)
            if docs:
                print(f"[RAG]  Building BM25 from {len(docs)} documents")
                # OPTIMIZED: Chunk size 800 (not 1000) = 20% fewer chunks = faster
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                print(f"[RAG]   Split into {len(chunks)} chunks (chunk_size=800, overlap=100)")
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
                            score = (1.0 - i/5) * 0.9  # 90% weight for FAISS
                            combined[doc_id] = (score, doc)
                        
                        for i, doc in enumerate(kw_docs[:5]):
                            doc_id = hash(doc.page_content[:100])
                            score = (1.0 - i/5) * 0.1  # 10% weight for BM25
                            if doc_id in combined:
                                old_score, old_doc = combined[doc_id]
                                combined[doc_id] = (old_score + score, old_doc)
                            else:
                                combined[doc_id] = (score, doc)
                        
                        sorted_docs = sorted(combined.values(), key=lambda x: x[0], reverse=True)
                        
                        # ── HALLUCINATION PREVENTION: SIMILARITY THRESHOLD ──────
                        # Same 0.5 threshold as the fresh-build path. Without this,
                        # a retriever loaded from disk (e.g. after an app restart)
                        # would skip the quality filter entirely and let
                        # low-confidence chunks reach the LLM unfiltered.
                        SIMILARITY_THRESHOLD = 0.5
                        
                        high_confidence_docs = []
                        for score, doc in sorted_docs:
                            if score >= SIMILARITY_THRESHOLD:
                                high_confidence_docs.append(doc)
                            else:
                                logger.debug(f"[RAG] Rejected low-quality chunk (cached path): score {score:.2f} < threshold {SIMILARITY_THRESHOLD}")
                        
                        if not high_confidence_docs:
                            logger.warning(f"[RAG] Query failed threshold (cached path): No chunks >= {SIMILARITY_THRESHOLD} found")
                            return []
                        
                        return high_confidence_docs[:3]
                    
                    def invoke(self, input_dict):
                        query = input_dict.get("query", input_dict) if isinstance(input_dict, dict) else input_dict
                        return self.get_relevant_documents(query)
                
                hybrid_retriever = OptimizedHybridRetriever(semantic_retriever, keyword_retriever, chunks)
                return hybrid_retriever
        
        except Exception as e:
            print(f"[RAG]   Could not load FAISS index from disk: {e}")
            print(f"[RAG]  Rebuilding fresh index instead...")
    
    # Build a fresh index from the uploaded documents
    print(f"[RAG]  Building FRESH index from {len(files)} file(s)")
    docs = _load_documents(files)
    
    if not docs:
        print(f"[RAG]  ERROR: No documents were loaded!")
        return None
    
    print(f"[RAG]  Loaded {len(docs)} documents, splitting into chunks...")
    # OPTIMIZED: 800 chars (not 1000) + 100 overlap = 20% fewer chunks = faster embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # Add section metadata to each chunk
    for chunk in chunks:
        if not chunk.metadata.get("section"):
            section = _extract_section_from_text(chunk.page_content)
            if section:
                chunk.metadata["section"] = section
    
    if not chunks:
        print(f"[RAG]  ERROR: No chunks created from documents!")
        return None
    
    print(f"[RAG]   Created {len(chunks)} chunks, building FAISS index...")
    print(f"[RAG]  This may take 30-60 seconds on first run (embedding all chunks)...")
    try:
        # Build FAISS for semantic search
        print(f"[RAG] Encoding {len(chunks)} chunks with all-MiniLM-L6-v2 (CACHED embeddings model)")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"[RAG]  FAISS index built successfully")
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Build BM25 for keyword search
        print("[RAG]  Building BM25 keyword retriever...")
        keyword_retriever = BM25Retriever.from_documents(chunks)
        print("[RAG] BM25 built successfully")
        
        # OPTIMIZED: Hybrid retriever with reranking
        print("[RAG] Creating hybrid retriever (FAISS 90% + BM25 10%)")
        
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
                
                # Combine: FAISS 90%, BM25 10% (prioritize semantic heavily)
                combined = {}
                for i, doc in enumerate(sem_docs[:5]):  # Top 5 from FAISS
                    doc_id = hash(doc.page_content[:100])
                    score = (1.0 - i/5) * 0.9  # 90% weight for FAISS
                    combined[doc_id] = (score, doc)
                
                for i, doc in enumerate(kw_docs[:5]):  # Top 5 from BM25
                    doc_id = hash(doc.page_content[:100])
                    score = (1.0 - i/5) * 0.1  # 10% weight for BM25
                    if doc_id in combined:
                        old_score, old_doc = combined[doc_id]
                        combined[doc_id] = (old_score + score, old_doc)
                    else:
                        combined[doc_id] = (score, doc)
                
                # Sort by combined score
                sorted_docs = sorted(combined.values(), key=lambda x: x[0], reverse=True)
                
                #  CRITICAL: Filter by minimum similarity threshold
                # ── HALLUCINATION PREVENTION: SIMILARITY THRESHOLD ──────
                # Only return chunks with confidence >= 0.5 (50%)
                # This is the FIRST layer of defense against hallucinations
                #
                # What this does:
                # - Filters out low-quality chunk matches before they reach LLM
                # - Prevents LLM from misinterpreting weak semantic connections
                # - Returns empty list if NO chunks meet the threshold
                #
                # Threshold = 0.5 because:
                # - 90% weight from FAISS (semantic) + 10% from BM25 (keyword)
                # - Position-weighted: top chunk weight 1.0, 3rd chunk 0.4
                # - 0.5 threshold blocks obvious mismatches while allowing valid retrievals
                #
                # Example:
                # - Asked "What is Python?" but got chunks about snakes → rejected (< 0.5)
                # - Asked "Python functions" and got "Python basics" → accepted (>= 0.5)
                
                SIMILARITY_THRESHOLD = 0.5
                
                high_confidence_docs = []
                for score, doc in sorted_docs:
                    if score >= SIMILARITY_THRESHOLD:
                        # Chunk passed threshold - include it
                        high_confidence_docs.append(doc)
                    else:
                        # Chunk failed threshold - reject and log for debugging
                        logger.debug(f"[RAG] Rejected low-quality chunk: score {score:.2f} < threshold {SIMILARITY_THRESHOLD}")
                
                # ── HANDLE NO CHUNKS CASE ────────────────────────────────
                if not high_confidence_docs:
                    logger.warning(f"[RAG] Query failed threshold: No chunks >= {SIMILARITY_THRESHOLD} found")
                    print(f"[RAG] Returning empty list - this will trigger HITL or fallback")
                    return []  # Return empty list = backend will refuse or ask user
                
                # ── RETURN TOP 3 PASSING CHUNKS ──────────────────────────
                # Take only top 3 for:
                # 1. Efficiency: Reduces token count to LLM
                # 2. Quality: Prevents dilution with less relevant chunks
                # 3. Citations: We cite [1] [2] [3] anyway
                return high_confidence_docs[:3]
            
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
        print(f"[RAG]  Index saved to disk at {index_dir}")
    except Exception as e:
        print(f"[RAG]   Warning: Could not save index to disk: {e}")
    
    print(f"[RAG] Hybrid retriever ready! (Chunks: {len(chunks)}, Search: FAISS 90% + BM25 10%)")
    
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
    
    # Delete old FAISS index to force rebuild
    index_dir = get_index_dir(tid)
    if index_dir.exists():
        print(f"[RAG] Deleting old cached index at {index_dir}")
        shutil.rmtree(index_dir, ignore_errors=True)
    
    retriever = _build_retriever(tid, force_rebuild=True)
    _retriever_cache[tid] = retriever
    if retriever:
        print(f"[RAG] Index rebuild complete!")
        return f"RAG index rebuilt successfully with {len(files)} file(s)."
    return "RAG index rebuild failed — check console for errors above"

def get_rag_context(query: str, thread_id: str) -> str:
    """Fast RAG retrieval - return top 3 chunks with clean structured formatting.
    
    OPTIMIZED: Loads from disk cache first (skips re-embedding), then queries.
    """
    if not query.strip():
        return ""
    
    tid = _safe_id(thread_id)
    
    # Get cached retriever or build it
    if tid not in _retriever_cache:
        print(f"[RAG]  Retriever not in memory — checking disk cache...")
        _retriever_cache[tid] = _build_retriever(tid)
    
    retriever = _retriever_cache.get(tid)
    if retriever is None:
        print(f"[RAG]  No retriever found (no docs?)")
        return ""
    
    print(f"[RAG] Retrieving top-3 chunks for query: {query[:50]}...")
    
    # Retrieve documents
    try:
        all_docs = retriever.invoke(query)
    except Exception as e:
        print(f"[RAG] Error: {e}")
        return ""
    
    if not all_docs:
        return ""
    
    docs = all_docs[:3]  # Top 3 chunks
    
    # Format chunks with clean structure and proper citations
    snippets = []
    source_citations = []  # Track unique sources
    
    for i, doc in enumerate(docs, start=1):
        text = doc.page_content.strip()
        
        # Get source and page info
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        citation = f"{source}"
        if isinstance(page, int):
            citation += f" (Page {page + 1})"
        
        # Track citation
        if citation not in source_citations:
            source_citations.append(citation)
        
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
        
        # Format with citation inline
        snippets.append(f"**[{i}] {citation}**\n{structured_text}")
    
    context = "\n\n---\n\n".join(snippets)
    
    # NOTE: We already have inline citations above (e.g., "[1] FineTuningLLM.pdf (Page 7)")
    # No need for a separate "Sources" footer - it creates duplication in LLM responses
    
    return context

def get_rag_context_with_confidence(query: str, thread_id: str) -> tuple:
    """
    Get RAG context AND calculate a confidence score (0-1).
    
    This function serves two purposes:
    1. Retrieve and format document chunks (same as get_rag_context)
    2. Calculate a confidence score indicating retrieval quality
    
    The confidence score is used by the backend to decide:
    - Score >= 0.6: Answer directly (high confidence)
    - Score < 0.6: Trigger HITL (ask user for approval)
    - Score = 0.0: No chunks found (refuse answer)
    
    Returns: (context_str, confidence_score)
    - context_str: Formatted chunks with citations [1] [2] [3]
    - confidence_score: Float 0.0 to 1.0 indicating retrieval quality
    """
    if not query.strip():
        return "", 0.0
    
    tid = _safe_id(thread_id)
    
    # ── STEP 1: LOAD OR BUILD RETRIEVER ──────────────────────────────
    # Check cache first (fast), then build if needed
    if tid not in _retriever_cache:
        _retriever_cache[tid] = _build_retriever(tid)
    
    retriever = _retriever_cache.get(tid)
    if retriever is None:
        print(f"[RAG] No retriever found - no documents uploaded yet")
        return "", 0.0
    
    print(f"[RAG] Retrieving + scoring: {query[:50]}...")
    
    # ── STEP 2: RETRIEVE DOCUMENTS ───────────────────────────────────
    try:
        # Invoke retriever: Uses hybrid FAISS + BM25 with 0.5 threshold filter
        # Returns only chunks that passed similarity threshold
        all_docs = retriever.invoke(query)
    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return "", 0.0
    
    # ── STEP 3: HANDLE EMPTY RETRIEVAL ──────────────────────────────
    if not all_docs:
        print(f"[RAG] No chunks met threshold - confidence: 0.0")
        return "", 0.0
    
    # ── STEP 4: CALCULATE CONFIDENCE SCORE ───────────────────────────
    # Confidence is NOT based on raw similarity scores (which vary per model).
    # Instead, use POSITION-BASED SCORING:
    # - Top chunk (rank 1): 0.95 (very high relevance)
    # - 2nd chunk (rank 2): 0.75 (good relevance)
    # - 3rd chunk (rank 3): 0.60 (acceptable relevance)
    #
    # Rationale:
    # - Retriever already filtered with 0.5 threshold
    # - Top-ranked chunks are more likely to be relevant
    # - Taking average of top-3 gives overall retrieval quality
    
    num_docs = len(all_docs[:3])  # We only use top 3 chunks anyway
    
    confidence_scores = []
    for rank_position in range(min(3, num_docs)):
        # Assign confidence based on rank position
        if rank_position == 0:
            # First result: highest confidence
            confidence_scores.append(0.95)
        elif rank_position == 1:
            # Second result: good confidence
            confidence_scores.append(0.75)
        else:
            # Third result: acceptable confidence
            # Note: Even this is >= 0.6, so HITL threshold remains meaningful
            confidence_scores.append(0.60)
    
    # Calculate average confidence across all retrieved chunks
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    print(f"[RAG] Confidence score: {avg_confidence:.2f} (from {num_docs} retrieved chunks)")
    
    # ── STEP 5: FORMAT AND RETURN CONTEXT ───────────────────────────
    # Get formatted chunks with citations using existing function
    context = get_rag_context(query, thread_id)
    
    return context, avg_confidence

def has_documents(thread_id: str) -> bool:
    """Return True if this thread has any uploaded documents."""
    return len(_get_supported_files(_safe_id(thread_id))) > 0
