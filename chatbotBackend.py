# ==============================
# IMPORTS
# ==============================
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, FunctionMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
try:
    from langgraph.store.postgres import PostgresStore
except Exception:
    PostgresStore = None
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except Exception:
    FAISS = None
    PyPDFLoader = None
    TextLoader = None

import sqlite3
import os
import math
import json
import hashlib
from datetime import datetime
from contextlib import contextmanager
import uuid
from contextvars import ContextVar
import re
import shutil
import numpy as np
from ddgs import DDGS
import yfinance as yf


# ==============================
# ENV + LLM
# ==============================
load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # change if needed
    temperature=0,
    streaming=True
)


# ==============================
# STATE
# ==============================
class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: str
    thread_id: str
    user_id: str
    allow_tools: bool
    awaiting_approval: bool
    approval_request: str
    approval_type: str
    approval_tool_calls: list[dict]
    approval_decision: str


# ==============================
# RAG CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RAG_DOCS_ROOT = BASE_DIR / "knowledge_base"
RAG_INDEX_ROOT = BASE_DIR / "faiss_index"

# ==============================
# LTM CONFIG (Postgres + Memory)
# ==============================
LTM_DB_URI = os.getenv(
    "LTM_POSTGRES_URI",
    "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable",
)
LTM_TOP_K = int(os.getenv("LTM_TOP_K", "4"))
LTM_STM_MAX_MESSAGES = int(os.getenv("LTM_STM_MAX_MESSAGES", "12"))
LTM_SUMMARY_EVERY_N = int(os.getenv("LTM_SUMMARY_EVERY_N", "10"))
LTM_SUMMARY_KEEP_LAST = int(os.getenv("LTM_SUMMARY_KEEP_LAST", "6"))
LTM_EMBEDDING_BACKEND = os.getenv("LTM_EMBEDDING_BACKEND", "hash").lower()

rag_retriever_cache = {}
rag_status_cache = {}

# Keep current thread context for tools that execute during an agent step.
active_thread_id = ContextVar("active_thread_id", default="default")

# HITL settings: keep only low-confidence RAG approval.
LOW_CONFIDENCE_RAG_MIN_CHARS = 220


def _safe_thread_id(thread_id: str) -> str:
    """Sanitize thread id for safe folder paths."""
    value = str(thread_id or "default").strip()
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value or "default"


def get_thread_rag_docs_dir(thread_id: str) -> Path:
    """Return per-thread docs folder path."""
    return RAG_DOCS_ROOT / _safe_thread_id(thread_id)


def get_thread_rag_index_dir(thread_id: str) -> Path:
    """Return per-thread FAISS index folder path."""
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
    """Create embedding model instance from backend selector."""
    if backend == "google":
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model_name), {"backend": "google", "model": model_name}

    dim = int(os.getenv("RAG_HASH_EMBEDDING_DIM", "384"))
    return HashEmbeddings(dim=dim), {"backend": "hash", "dim": dim}


MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.

Return ONLY valid JSON in this format:
{"should_write": true|false, "memories": [{"text": "...", "is_new": true|false}]}
"""


SUMMARY_PROMPT = """Summarize the following conversation history into short, factual memory notes.
Focus on stable facts about the user, preferences, goals, and ongoing work.
Return 3-6 short bullet-like sentences, each on its own line. No speculation.

Conversation:
{history}
"""


def _init_memory_store() -> bool:
    """Initialize PostgresStore for long-term memory."""
    if PostgresStore is None:
        print("LTM store unavailable: langgraph.store.postgres is not installed.")
        return False
    try:
        with PostgresStore.from_conn_string(LTM_DB_URI) as store:
            store.setup()
        return True
    except Exception as err:
        print(f"LTM store init warning: {err}")
        return False


memory_store_available = _init_memory_store()
memory_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
memory_embeddings, _memory_meta = _embedding_from_backend(LTM_EMBEDDING_BACKEND)


@contextmanager
def _open_memory_store():
    if not memory_store_available or PostgresStore is None:
        yield None
        return
    with PostgresStore.from_conn_string(LTM_DB_URI) as store:
        yield store


def _memory_namespace(user_id: str) -> tuple:
    return ("user", user_id, "details")


def _memory_texts(items: list) -> str:
    if not items:
        return "(empty)"
    lines = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        text = value.get("data", "")
        if text:
            lines.append(text)
    return "\n".join(lines) if lines else "(empty)"


def _parse_memory_json(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except Exception:
        return {"should_write": False, "memories": []}

    should_write = bool(data.get("should_write"))
    memories = data.get("memories") or []
    parsed = []
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        text = str(mem.get("text", "")).strip()
        is_new = bool(mem.get("is_new"))
        if text:
            parsed.append({"text": text, "is_new": is_new})

    return {"should_write": should_write, "memories": parsed}


def _embed_text(text: str) -> list[float]:
    try:
        return memory_embeddings.embed_query(text)
    except Exception:
        return []


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _search_memory(items: list, query: str, k: int) -> list[str]:
    if not items:
        return []
    query_vec = _embed_text(query)
    scored = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        text = value.get("data", "")
        emb = value.get("embedding", [])
        score = _cosine_similarity(query_vec, emb) if query_vec and emb else 0.0
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:k] if text]


def _maybe_store_summary(messages: list[BaseMessage], user_id: str) -> None:
    if not memory_store_available:
        return
    if len(messages) < LTM_SUMMARY_EVERY_N:
        return

    if len(messages) % LTM_SUMMARY_EVERY_N != 0:
        return

    history_messages = messages[:-LTM_SUMMARY_KEEP_LAST]
    if not history_messages:
        return

    history_text = "\n".join(
        f"{msg.type}: {msg.content}" for msg in history_messages if getattr(msg, "content", None)
    )
    if not history_text:
        return

    summary_text = llm.invoke(
        SystemMessage(content=SUMMARY_PROMPT.format(history=history_text))
    ).content.strip()

    if summary_text:
        ns = _memory_namespace(user_id)
        with _open_memory_store() as store:
            if store is None:
                return
            store.put(
                ns,
                str(uuid.uuid4()),
                {
                    "data": summary_text,
                    "embedding": _embed_text(summary_text),
                    "kind": "summary",
                    "ts": datetime.utcnow().isoformat(),
                },
            )


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
    """Collect supported knowledge files for indexing (thread-scoped)."""
    supported = {".txt", ".md", ".pdf"}
    files: list[Path] = []

    docs_dir = get_thread_rag_docs_dir(thread_id)
    if docs_dir.exists():
        files.extend([p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported])

    # Deduplicate while preserving order
    unique_files = list(dict.fromkeys(files))
    return sorted(unique_files)


def _load_documents(paths: list[Path]) -> list[Document]:
    """Load text and PDF files into LangChain documents."""
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
            print(f"RAG load warning for {path.name}: {err}")

    return documents


def _build_rag_retriever(thread_id: str, force_rebuild: bool = False):
    """Build or load FAISS retriever from local knowledge base files."""
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
    """Lazy initialize retriever so startup remains fast."""
    thread_key = _safe_thread_id(thread_id)

    if thread_key not in rag_retriever_cache or force_rebuild:
        rag_retriever_cache[thread_key] = _build_rag_retriever(thread_key, force_rebuild=force_rebuild)

    return rag_retriever_cache.get(thread_key)


def rebuild_rag_index(thread_id: str) -> str:
    """Rebuild FAISS index from files in Chatbot/knowledge_base."""
    thread_key = _safe_thread_id(thread_id)
    retriever = ensure_rag_ready(thread_key, force_rebuild=True)
    if retriever is None:
        return f"RAG rebuild failed: {_get_rag_status(thread_key)}"
    return f"RAG rebuild successful: {_get_rag_status(thread_key)}"


def _latest_user_query(messages: list[BaseMessage]) -> str:
    """Return the most recent human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def _rag_context_for_query(query: str, thread_id: str, k: int = 4) -> str:
    """Fetch compact RAG context for prompt grounding."""
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
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        page_info = f" (page {page + 1})" if isinstance(page, int) else ""
        chunk = doc.page_content.strip().replace("\n", " ")
        snippets.append(f"[{i}] {source}{page_info}: {chunk}")

    return "\n\n".join(snippets)


def _is_self_query(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    markers = [
        "tell me about myself",
        "about myself",
        "who am i",
        "my details",
        "my information",
        "my profile",
        "what do you know about me",
    ]
    return any(marker in q for marker in markers)


def _extract_name_from_memory(memory_context: str) -> str:
    match = re.search(r"name is ([A-Za-z][A-Za-z\s'-]{0,40})", memory_context, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _filter_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    filtered = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                continue
        if isinstance(msg, (ToolMessage, FunctionMessage)):
            continue
        filtered.append(msg)
    return filtered


def _heuristic_memories(text: str) -> list[str]:
    if not text:
        return []
    items = []
    name_match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,40})", text, re.IGNORECASE)
    if name_match:
        items.append(f"User's name is {name_match.group(1).strip()}.")

    uni_match = re.search(r"\b(?:from|at)\s+([A-Za-z][A-Za-z\s'-]{2,60}university)\b", text, re.IGNORECASE)
    if uni_match:
        items.append(f"User studies at {uni_match.group(1).strip()}.")

    degree_match = re.search(r"\b(?:doing|studying)\s+([A-Za-z\.\s]{2,30})\b", text, re.IGNORECASE)
    if degree_match:
        items.append(f"User is studying {degree_match.group(1).strip()}.")

    interest_match = re.search(r"\binterest in\s+([A-Za-z\s,&-]{2,60})", text, re.IGNORECASE)
    if interest_match:
        items.append(f"User is interested in {interest_match.group(1).strip()}.")

    age_match = re.search(r"\b(\d{1,2})\s*(?:years? old|yo)\b", text, re.IGNORECASE)
    if age_match:
        items.append(f"User is {age_match.group(1)} years old.")

    return list(dict.fromkeys(items))


def _is_document_intent(query: str) -> bool:
    """Detect whether user is explicitly asking about uploaded/local documents."""
    q = query.lower().strip()
    if not q:
        return False

    doc_markers = [
        "pdf", "document", "doc", "file", "upload", "resume", "letter", "in this",
        "from this", "according to", "provided", "page",
    ]
    return any(marker in q for marker in doc_markers)


def _needs_external_tools(query: str) -> bool:
    """Heuristic gate to avoid tool calls for simple knowledge questions."""
    q = query.lower().strip()
    if not q:
        return False

    tool_markers = [
        "latest", "today", "current", "news", "headline", "price", "stock",
        "weather", "score", "live", "market", "exchange rate", "time now",
    ]
    return any(marker in q for marker in tool_markers)


def _approval_reset_state() -> dict:
    """Clear HITL approval state after a decision is handled."""
    return {
        "awaiting_approval": False,
        "approval_request": "",
        "approval_type": "",
        "approval_tool_calls": [],
        "approval_decision": "",
    }


def _resolve_hitl_decision(state: ChatState, latest_query: str, rag_context: str) -> dict | None:
    """Resume flow after human approval decision."""
    decision = str(state.get("approval_decision", "")).lower().strip()
    if not decision:
        return None

    approval_type = str(state.get("approval_type", ""))

    if approval_type == "low_confidence_rag":
        if decision == "approve":
            forced_prompt = [
                SystemMessage(content=(
                    "Human approved answering from weak document context. "
                    "Answer only from the provided context and cite sources like [1], [2]."
                )),
                HumanMessage(content=f"Question: {latest_query}\n\nContext:\n{rag_context}"),
            ]
            forced_answer = llm.invoke(forced_prompt)
            return {**_approval_reset_state(), "messages": [forced_answer]}

        # regenerate/reject path for low confidence context.
        clarify = AIMessage(content=(
            "I need a little more direction before answering. "
            "Please refine your question or upload a clearer/more relevant document."
        ))
        return {**_approval_reset_state(), "messages": [clarify]}

    return _approval_reset_state()


# ==============================
# TOOLS
# ==============================

@tool
def search_tool(query: str) -> str:
    """Search latest information from web"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return "\n".join([r["body"] for r in results])


@tool
def calculator(expression: str) -> str:
    """Solve math expressions"""
    try:
        return str(eval(expression, {"__builtins__": None}, vars(math)))
    except:
        return "Error in calculation"


@tool
def get_stock_price(symbol: str) -> str:
    """Get stock price"""
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"{symbol} price is {round(price,2)} USD"
    except:
        return "Stock not found"


@tool
def get_current_date_time() -> str:
    """Get current local date and time."""
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


@tool
def rag_search(question: str) -> str:
    """Search local knowledge base documents for RAG context."""
    thread_id = active_thread_id.get()
    docs_dir = get_thread_rag_docs_dir(thread_id)
    retriever = ensure_rag_ready(thread_id)

    if retriever is None:
        return (
            "RAG is not ready. "
            f"{_get_rag_status(thread_id)}. "
            f"Add .txt/.md/.pdf files in {docs_dir} and rebuild index."
        )

    try:
        docs = retriever.invoke(question)
    except Exception as err:
        return f"RAG retrieval error: {err}"

    if not docs:
        return "No relevant context found in local knowledge base."

    contexts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_info = f" (page {page + 1})" if isinstance(page, int) else ""
        chunk = doc.page_content.strip().replace("\n", " ")
        contexts.append(f"[{i}] {Path(source).name}{page_info}: {chunk}")

    return "\n\n".join(contexts)


tools = [search_tool, calculator, get_stock_price, get_current_date_time, rag_search]
agent_tools = [search_tool, calculator, get_stock_price, get_current_date_time]


# ==============================
# LLM WITH TOOLS
# ==============================
llm_with_tools = llm.bind_tools(tools)
llm_with_agent_tools = llm.bind_tools(agent_tools)


# ==============================
# CHAT NODE
# ==============================
def _get_user_memory_context(user_id: str, query: str) -> str:
    if not memory_store_available:
        return ""

    ns = _memory_namespace(user_id)
    with _open_memory_store() as store:
        if store is None:
            return ""
        try:
            items = store.search(ns)
        except Exception as err:
            print(f"LTM search warning: {err}")
            return ""

    if not items:
        return ""

    matches = _search_memory(items, query, LTM_TOP_K) if query else _memory_texts(items).splitlines()
    if not matches:
        return ""

    return "\n".join(f"- {item}" for item in matches)


def remember_node(state: ChatState):
    if not memory_store_available:
        return {"messages": []}

    user_id = _safe_thread_id(state.get("user_id") or state.get("thread_id", "default"))
    ns = _memory_namespace(user_id)

    last_text = _latest_user_query(state.get("messages", []))
    if not last_text:
        return {"messages": []}

    with _open_memory_store() as store:
        if store is None:
            return {"messages": []}
        try:
            items = store.search(ns)
        except Exception as err:
            print(f"LTM search warning: {err}")
            return {"messages": []}

    existing = _memory_texts(items)

    try:
        raw = memory_llm.invoke(
            [
                SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
                HumanMessage(content=last_text),
            ]
        ).content
        decision = _parse_memory_json(raw)
    except Exception as err:
        print(f"LTM extract warning: {err}")
        return {"messages": []}

    memories_to_write = []
    if decision.get("should_write"):
        for mem in decision.get("memories", []):
            text = mem.get("text", "").strip()
            if mem.get("is_new") and text:
                memories_to_write.append(text)

    if not memories_to_write:
        memories_to_write = _heuristic_memories(last_text)

    if memories_to_write:
        with _open_memory_store() as store:
            if store is None:
                return {"messages": []}
            for text in memories_to_write:
                store.put(
                    ns,
                    str(uuid.uuid4()),
                    {
                        "data": text,
                        "embedding": _embed_text(text),
                        "kind": "fact",
                        "ts": datetime.utcnow().isoformat(),
                    },
                )

    _maybe_store_summary(state.get("messages", []), user_id)
    return {"messages": []}


def chat_node(state: ChatState):
    latest_query = _latest_user_query(state["messages"])
    thread_id = _safe_thread_id(state.get("thread_id", "default"))
    user_id = _safe_thread_id(state.get("user_id") or state.get("thread_id", "default"))
    memory_context = _get_user_memory_context(user_id, latest_query)

    # If waiting for human decision and none provided, keep asking for approval.
    if state.get("awaiting_approval") and not state.get("approval_decision"):
        pending_request = state.get("approval_request", "Approval required. Please choose Approve or Regenerate.")
        return {"messages": [AIMessage(content=pending_request)]}

    mode = str(state.get("mode", "auto")).lower()
    if mode not in {"auto", "hybrid", "agent_only", "rag_only"}:
        mode = "auto"

    if mode == "auto":
        has_knowledge_files = len(_collect_rag_files(thread_id)) > 0
        if not has_knowledge_files:
            mode = "agent_only"
        else:
            mode = "rag_only" if _is_document_intent(latest_query) else "hybrid"

    rag_context = _rag_context_for_query(latest_query, thread_id) if mode in {"hybrid", "rag_only"} else ""

    # Handle explicit HITL decision and resume graph execution.
    decision_result = _resolve_hitl_decision(state, latest_query, rag_context)
    if decision_result is not None:
        return decision_result

    if memory_context and _is_self_query(latest_query):
        name = _extract_name_from_memory(memory_context)
        greeting = f"Sure {name}, " if name else "Sure, "
        response_text = f"{greeting}here is what I remember about you:\n{memory_context}"
        return {"messages": [AIMessage(content=response_text)]}
    if _is_self_query(latest_query) and not memory_context:
        response_text = (
            "I do not have any saved details about you yet. "
            "Tell me your name, school, or interests and I will remember them."
        )
        return {"messages": [AIMessage(content=response_text)]}

    # HITL gate for low-confidence document answers.
    if mode in {"hybrid", "rag_only"} and _is_document_intent(latest_query):
        if not rag_context or len(rag_context) < LOW_CONFIDENCE_RAG_MIN_CHARS:
            request = (
                "HITL approval needed: document context confidence is low. "
                "Choose Approve to answer from available context or Regenerate to refine the request."
            )
            return {
                "awaiting_approval": True,
                "approval_request": request,
                "approval_type": "low_confidence_rag",
                "approval_tool_calls": [],
                "approval_decision": "",
                "messages": [AIMessage(content=request)],
            }

    rag_instructions = (
        f"RAG Context (highest priority if relevant):\n{rag_context}\n\n"
        "If RAG context clearly answers the question, answer from it and cite source tags like [1], [2]. "
        "If context is not relevant, then use tools as needed."
    ) if rag_context else "No RAG context available for this query."

    allow_tools = _needs_external_tools(latest_query)

    system_prompt = f"""
You are a smart AI assistant.

Rules:
- Current response mode: {mode}
- Use get_current_date_time for date/time questions
- If mode is rag_only: answer only from RAG context and clearly say when answer is not found in context
- If mode is agent_only: use tools for web/stock/math and do not rely on RAG context
- If mode is hybrid: prefer RAG context when relevant, otherwise use tools
- Only call tools when the user asks for live/current data or explicitly requests a tool
- Keep responses concise and useful

{rag_instructions}
"""

    if memory_context:
        system_prompt += f"\n\nUser memory (use only if relevant):\n{memory_context}\n"

    short_term_messages = state["messages"][-LTM_STM_MAX_MESSAGES:]
    safe_messages = _filter_messages_for_llm(short_term_messages)
    if not safe_messages and latest_query:
        safe_messages = [HumanMessage(content=latest_query)]
    messages = [SystemMessage(content=system_prompt)] + safe_messages

    token = active_thread_id.set(thread_id)
    try:
        if mode == "rag_only":
            response = llm.invoke(messages)
        elif mode == "agent_only":
            response = llm_with_agent_tools.invoke(messages) if allow_tools else llm.invoke(messages)
        else:
            response = llm_with_tools.invoke(messages) if allow_tools else llm.invoke(messages)
    finally:
        active_thread_id.reset(token)

    return {"messages": [response], "allow_tools": allow_tools}


def route_after_chat(state: ChatState):
    """Route to tools, end, or wait for human based on state and latest message."""
    if state.get("awaiting_approval"):
        return "wait_for_human"

    if state.get("allow_tools") is False:
        return "__end__"

    decision = tools_condition(state)
    if decision == "tools":
        return "tools"
    return "__end__"


# ==============================
# DATABASE (MEMORY)
# ==============================
def init_checkpointer():
    """Initialize checkpointer with automatic error recovery."""
    db_path = "chatbot_db"
    
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("SELECT 1")  # Test connection
        checkpointer = SqliteSaver(conn=conn)
        return checkpointer, conn
    except Exception as e:
        print(f"Checkpoint error detected: {e}. Recovering...")
        
        # Backup corrupted database
        if os.path.exists(db_path):
            backup_path = f"{db_path}.backup_{os.getpid()}"
            try:
                os.rename(db_path, backup_path)
                print(f"Backed up corrupted database to {backup_path}")
            except Exception as backup_err:
                print(f"Backup failed: {backup_err}")
        
        # Create fresh database
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        print("Fresh checkpoint database created")
        return checkpointer, conn

checkpointer, conn = init_checkpointer()


def delete_thread_history(thread_id: str) -> str:
    """Permanently delete a thread from all checkpoint tables containing thread_id."""
    if not thread_id or not thread_id.strip():
        return "Delete failed: invalid thread id."

    thread_key = _safe_thread_id(thread_id)

    try:
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]

        total_deleted = 0
        for table in tables:
            # Only delete from tables that actually contain thread_id column
            columns = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")').fetchall()]
            if "thread_id" not in columns:
                continue

            cur.execute(f'DELETE FROM "{table}" WHERE thread_id = ?', (thread_key,))
            if cur.rowcount and cur.rowcount > 0:
                total_deleted += cur.rowcount

        conn.commit()

        # Remove thread-scoped RAG docs/index and in-memory cache.
        for folder in [get_thread_rag_docs_dir(thread_key), get_thread_rag_index_dir(thread_key)]:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

        rag_retriever_cache.pop(thread_key, None)
        rag_status_cache.pop(thread_key, None)

        if memory_store_available:
            try:
                ns = _memory_namespace(thread_key)
                with _open_memory_store() as store:
                    if store is not None:
                        items = store.search(ns)
                        for item in items:
                            key = getattr(item, "key", None)
                            if key is not None:
                                store.delete(ns, key)
            except Exception as err:
                print(f"LTM delete warning: {err}")

        if total_deleted == 0:
            return "Chat deleted permanently (no checkpoints found, local docs/index removed)."

        return f"Chat deleted permanently ({total_deleted} checkpoint rows removed)."
    except Exception as err:
        try:
            conn.rollback()
        except Exception:
            pass
        return f"Delete failed: {err}"


# ==============================
# GRAPH (AGENT)
# ==============================
builder = StateGraph(ChatState)

builder.add_node("remember", remember_node)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat_node")

builder.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {
        "tools": "tools",
        "wait_for_human": END,
        "__end__": END,
    }
)

builder.add_edge("tools", "chat_node")

chatbot = builder.compile(checkpointer=checkpointer)


# ==============================
# CHAT TITLE GENERATION
# ==============================
def generate_chat_title(user_message: str) -> str:
    """Generate a concise chat title from the first user message."""
    if not user_message or not user_message.strip():
        return "New Chat"
    
    msg = user_message.strip()[:200]  # Limit to first 200 chars
    
    try:
        # Try LLM-based title generation
        prompt = f"""Given this user message, generate a very short chat title (2-5 words max, no punctuation). 
Only return the title, nothing else.

User message: {msg}

Title:"""
        title = llm.invoke(prompt).content.strip()
        title = title.strip('"\'.!?,;:').strip()
        if title and len(title) > 0:
            return title[:50]
    except Exception as e:
        print(f"Title generation error (using fallback): {str(e)[:50]}")
    
    # Fallback: Use first 4-5 words from message
    words = msg.split()
    if len(words) > 0:
        fallback_title = ' '.join(words[:5])
        if len(fallback_title) > 50:
            fallback_title = fallback_title[:47] + "..."
        return fallback_title
    
    return "New Chat"


# ==============================
# THREAD UTIL
# ==============================
def unique_thread_pointer():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)