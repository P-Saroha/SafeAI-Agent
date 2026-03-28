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
    pending_weather: bool
    last_weather_link: str
    awaiting_approval: bool
    approval_request: str
    approval_type: str
    approval_tool_calls: list[dict]
    approval_decision: str
    tools_called_in_turn: int  # Track how many times we've called tools for this user input


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
LTM_MAX_ENTRIES = int(os.getenv("LTM_MAX_ENTRIES", "25"))

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


STRUCTURED_MEMORY_PROMPT = """You extract user memory into a strict JSON structure.

EXISTING USER MEMORY:
{user_details_content}

TASK:
- Read the latest user message.
- Extract ONLY new or updated facts that are explicitly stated.
- If nothing new is present, set should_write=false and all fields empty.
- Keep values short, clean, and precise.
- Normalize abbreviations like B.E., E.C.E., ML, AI, DL.
- If the user explicitly asks to remember something, set should_write=true and capture any relevant fields.

Return ONLY valid JSON in this exact format:
{
    "should_write": true|false,
    "memory": {
        "name": "",
        "age": "",
        "education": "",
        "university": "",
        "favorite_language": "",
        "favorite_color": "",
        "favorite_laptop": "",
        "interests": [],
        "likes": [],
        "projects": [],
        "travel_plans": []
    }
}
"""

AUTO_MEMORY_PROMPT = """You decide what user facts are worth remembering.

EXISTING USER MEMORY:
{user_details_content}

TASK:
- Read the latest user message.
- Extract only stable, long-term facts about the user.
- Ignore casual chatter, one-off opinions, and temporary states.
- Avoid duplicating facts already present in existing memory.

Return ONLY valid JSON in this exact format:
{
    "should_write": true|false,
    "facts": [
        "You are 23 years old.",
        "You completed 12th class in 2022."
    ]
}
"""

SUMMARY_PROMPT = """Summarize the following conversation history into short, factual memory notes.
Focus on stable facts about the user, preferences, goals, and ongoing work.
Return 3-6 short bullet-like sentences, each on its own line. No speculation.

Conversation:
{history}
"""


memory_store_last_error = ""


def _init_memory_store() -> bool:
    """Initialize PostgresStore for long-term memory."""
    global memory_store_last_error
    memory_store_last_error = ""
    if PostgresStore is None:
        memory_store_last_error = "Postgres store unavailable (missing langgraph.store.postgres)."
        print("LTM store unavailable: langgraph.store.postgres is not installed.")
        return False
    try:
        with PostgresStore.from_conn_string(LTM_DB_URI) as store:
            store.setup()
        memory_store_last_error = ""
        return True
    except Exception as err:
        memory_store_last_error = str(err)
        print(f"LTM store init warning: {err}")
        return False


memory_store_available = _init_memory_store()
memory_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
memory_embeddings, _memory_meta = _embedding_from_backend(LTM_EMBEDDING_BACKEND)


def _ensure_memory_store() -> bool:
    global memory_store_available
    if memory_store_available:
        return True
    memory_store_available = _init_memory_store()
    return memory_store_available


@contextmanager
def _open_memory_store():
    if PostgresStore is None:
        yield None
        return
    if not _ensure_memory_store():
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
            cleaned = text
            if cleaned.lower().startswith("user's name is"):
                cleaned = re.split(r"\b i \b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            lines.append(cleaned)
    return "\n".join(lines) if lines else "(empty)"


def _parse_auto_memory_json(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except Exception:
        return {"should_write": False, "facts": []}

    facts = data.get("facts") or []
    if isinstance(facts, str):
        facts = [facts]
    facts = [str(f).strip() for f in facts if str(f).strip()]
    return {"should_write": bool(data.get("should_write")), "facts": facts}


def _heuristic_auto_facts(text: str) -> list[str]:
    if not text:
        return []
    clean = re.sub(r"\s+", " ", text.strip())
    facts = []
    age_match = re.search(r"\b(?:my age is|i am)\s+(\d{1,2})\s*(?:years? old|yo)?\b", clean, re.IGNORECASE)
    if age_match:
        facts.append(f"You are {age_match.group(1)} years old.")

    class_match = re.search(r"\b(?:done|completed|passed|finished)\s+12(?:th)?\s*(?:class)?\s*(?:in)?\s*(\d{4})\b", clean, re.IGNORECASE)
    if class_match:
        facts.append(f"You completed 12th class in {class_match.group(1)}.")

    plan_match = re.search(r"\b(?:planning to|plan to|going to)\s+go\s+to\s+([A-Za-z\s'-]{2,60})\b", clean, re.IGNORECASE)
    if plan_match:
        destination = plan_match.group(1).strip()
        facts.append(f"You are planning to go to {destination}.")

    return _dedupe_memory_list(facts)


def _filter_auto_facts(facts: list[str], structured: dict) -> list[str]:
    if not facts:
        return []
    skip_terms = []
    if structured.get("name"):
        skip_terms.extend(["name is", "your name"])
    if structured.get("education"):
        skip_terms.extend(["studying", "education", "b.e", "b.s", "b.sc"])
    if structured.get("university"):
        skip_terms.append("university")
    if structured.get("favorite_language"):
        skip_terms.append("favorite programming language")
    if structured.get("favorite_laptop"):
        skip_terms.append("favorite laptop")
    if structured.get("interests"):
        skip_terms.append("interested in")
    if structured.get("likes"):
        skip_terms.append("you like")
    if structured.get("projects"):
        skip_terms.append("worked on")

    filtered = []
    for fact in facts:
        lowered = str(fact).lower()
        if any(term in lowered for term in skip_terms):
            continue
        filtered.append(fact)
    return _dedupe_memory_list(filtered)


def _normalize_list(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in values or []:
        cleaned = re.sub(r"\s+", " ", str(item).strip())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _structured_memory_summary(memory: dict) -> str:
    if not memory:
        return "(empty)"

    parts = []
    for key in ["name", "age", "education", "university", "favorite_language", "favorite_color", "favorite_laptop"]:
        value = str(memory.get(key) or "").strip()
        if value:
            parts.append(f"{key}: {value}")

    for key in ["interests", "likes", "projects", "travel_plans"]:
        values = _normalize_interest_list(memory.get(key) or []) if key == "interests" else _normalize_list(memory.get(key) or [])
        if values:
            parts.append(f"{key}: {', '.join(values)}")

    return "\n".join(parts) if parts else "(empty)"


def _parse_structured_memory_json(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except Exception:
        return {"should_write": False, "memory": {}}

    memory = data.get("memory") or {}
    if not isinstance(memory, dict):
        memory = {}

    interests_raw = memory.get("interests") or []
    likes_raw = memory.get("likes") or []
    projects_raw = memory.get("projects") or []
    travel_raw = memory.get("travel_plans") or []
    if isinstance(interests_raw, str):
        interests_raw = _split_list_terms(interests_raw)
    if isinstance(likes_raw, str):
        likes_raw = _split_list_terms(likes_raw)
    if isinstance(projects_raw, str):
        projects_raw = _split_list_terms(projects_raw)
    if isinstance(travel_raw, str):
        travel_raw = _split_list_terms(travel_raw)

    parsed = {
        "name": str(memory.get("name", "")).strip(),
        "age": str(memory.get("age", "")).strip(),
        "education": str(memory.get("education", "")).strip(),
        "university": str(memory.get("university", "")).strip(),
        "favorite_language": str(memory.get("favorite_language", "")).strip(),
        "favorite_color": str(memory.get("favorite_color", "")).strip(),
        "favorite_laptop": str(memory.get("favorite_laptop", "")).strip(),
        "interests": _normalize_interest_list(interests_raw),
        "likes": _normalize_list(likes_raw),
        "projects": _normalize_list(projects_raw),
        "travel_plans": _normalize_list(travel_raw),
    }

    return {
        "should_write": bool(data.get("should_write")),
        "memory": parsed,
    }


def _structured_memory_to_sentences(memory: dict) -> list[str]:
    if not memory:
        return []

    sentences = []
    name = _clean_name_value(memory.get("name") or "")
    age = _clean_age_value(memory.get("age") or "")
    education = _normalize_degree_text(str(memory.get("education") or "").strip()).rstrip(".")
    university = str(memory.get("university") or "").strip()
    favorite_language = _clean_favorite_language(memory.get("favorite_language") or "")
    favorite_color = _clean_favorite_color(memory.get("favorite_color") or "")
    favorite_laptop = _clean_favorite_device(memory.get("favorite_laptop") or "")

    if name:
        sentences.append(f"Your name is {name}.")
    if age:
        sentences.append(f"You are {age} years old.")
    if education:
        sentences.append(f"You are studying {education}.")
    if university:
        sentences.append(f"You study at {university}.")
    if favorite_language:
        sentences.append(f"Your favorite programming language is {favorite_language}.")
    if favorite_color:
        sentences.append(f"Your favorite color is {favorite_color}.")
    if favorite_laptop:
        sentences.append(f"Your favorite laptop is {favorite_laptop}.")

    for interest in _normalize_list(memory.get("interests") or []):
        sentences.append(f"You are interested in {interest}.")
    for like in _normalize_list(memory.get("likes") or []):
        sentences.append(f"You like {like}.")
    for project in _normalize_list(memory.get("projects") or []):
        sentences.append(f"You worked on a {project}.")
    for plan in _normalize_list(memory.get("travel_plans") or []):
        sentences.append(f"You plan to go to {plan}.")

    return sentences


def _structured_memory_from_items(items: list) -> dict:
    memory = {}
    for item in items:
        value = getattr(item, "value", {}) or {}
        if value.get("kind") != "structured":
            continue
        field = value.get("field") or ""
        if not field:
            key = getattr(item, "key", "")
            if isinstance(key, str) and key.startswith("structured:"):
                field = key.split(":", 1)[1]
        if not field:
            continue

        if field in {"interests", "likes", "projects", "travel_plans"}:
            raw_list = value.get("list") or value.get("data") or []
            if isinstance(raw_list, str):
                raw_list = _split_list_terms(raw_list)
            memory[field] = _normalize_list(raw_list)
        else:
            memory[field] = str(value.get("data", "")).strip()

    memory["name"] = _clean_name_value(memory.get("name", ""))
    memory["age"] = _clean_age_value(memory.get("age", ""))
    memory["education"] = _normalize_degree_text(memory.get("education", "")).rstrip(".")
    memory["favorite_language"] = _clean_favorite_language(memory.get("favorite_language", ""))
    memory["favorite_color"] = _clean_favorite_color(memory.get("favorite_color", ""))
    memory["favorite_laptop"] = _clean_favorite_device(memory.get("favorite_laptop", ""))

    return memory


def _merge_structured_memory(base: dict, incoming: dict) -> dict:
    result = dict(base or {})
    for key in ["name", "age", "education", "university", "favorite_language", "favorite_color", "favorite_laptop"]:
        value = str(incoming.get(key) or "").strip()
        if key == "name":
            value = _clean_name_value(value)
        if key == "age":
            value = _clean_age_value(value)
        if key == "education":
            value = _normalize_degree_text(value).rstrip(".")
        if key == "favorite_language":
            value = _clean_favorite_language(value)
        if key == "favorite_color":
            value = _clean_favorite_color(value)
        if key == "favorite_laptop":
            value = _clean_favorite_device(value)
        existing = str(result.get(key, "") or "").strip()
        if value and value.lower() != existing.lower():
            result[key] = value

    for key in ["interests", "likes", "projects", "travel_plans"]:
        existing = _normalize_list(result.get(key) or [])
        incoming_values = _normalize_list(incoming.get(key) or [])
        if key == "interests":
            merged = _normalize_interest_list(existing + incoming_values)
            result[key] = merged
        else:
            merged = existing + incoming_values
            result[key] = _normalize_list(merged)

    return result


def _store_structured_memory(user_id: str, memory: dict) -> None:
    if not _ensure_memory_store():
        return

    user_key = _safe_thread_id(user_id)
    ns = _memory_namespace(user_key)
    with _open_memory_store() as store:
        if store is None:
            return

        # Remove summary/history items so latest structured facts are authoritative.
        try:
            existing_items = store.search(ns, limit=200)
        except Exception:
            existing_items = []
        for item in existing_items:
            value = getattr(item, "value", {}) or {}
            if value.get("kind") == "summary":
                key = getattr(item, "key", None)
                if key:
                    try:
                        store.delete(ns, key)
                    except Exception:
                        pass

        timestamp = datetime.utcnow().isoformat()
        for key in ["name", "education", "university", "favorite_language", "favorite_laptop"]:
            value = str(memory.get(key) or "").strip()
            if key == "name":
                value = _clean_name_value(value)
            if key == "education":
                value = _normalize_degree_text(value).rstrip(".")
            if key == "favorite_language":
                value = _clean_favorite_language(value)
            if key == "favorite_laptop":
                value = _clean_favorite_device(value)
            if not value:
                continue
            sentence = _structured_memory_to_sentences({key: value})[0]
            store.put(
                ns,
                f"structured:{key}",
                {
                    "data": value,
                    "sentence": sentence,
                    "embedding": _embed_text(sentence),
                    "kind": "structured",
                    "field": key,
                    "ts": timestamp,
                },
            )

        for key in ["interests", "likes", "projects"]:
            values = _normalize_list(memory.get(key) or [])
            if not values:
                continue
            if key == "interests":
                sentence = f"You are interested in {', '.join(values)}."
            elif key == "likes":
                sentence = f"You like {', '.join(values)}."
            else:
                sentence = f"You worked on {', '.join(values)}."

            store.put(
                ns,
                f"structured:{key}",
                {
                    "data": values,
                    "list": values,
                    "sentence": sentence,
                    "embedding": _embed_text(sentence),
                    "kind": "structured",
                    "field": key,
                    "ts": timestamp,
                },
            )


def _store_explicit_memory(user_id: str, text: str) -> None:
    if not _ensure_memory_store():
        return
    value = str(text or "").strip()
    if not value:
        return
    ns = _memory_namespace(_safe_thread_id(user_id))
    with _open_memory_store() as store:
        if store is None:
            return
        store.put(
            ns,
            str(uuid.uuid4()),
            {
                "data": value,
                "embedding": _embed_text(value),
                "kind": "explicit",
                "ts": datetime.utcnow().isoformat(),
            },
        )


def _store_auto_memory(user_id: str, facts: list[str]) -> None:
    if not _ensure_memory_store():
        return
    if not facts:
        return
    ns = _memory_namespace(_safe_thread_id(user_id))
    with _open_memory_store() as store:
        if store is None:
            return
        for fact in _dedupe_memory_list(facts):
            store.put(
                ns,
                str(uuid.uuid4()),
                {
                    "data": fact,
                    "embedding": _embed_text(fact),
                    "kind": "auto",
                    "ts": datetime.utcnow().isoformat(),
                },
            )


def _search_memory_texts(texts: list[str], query: str, k: int) -> list[str]:
    if not texts:
        return []
    query_vec = _embed_text(query)
    scored = []
    for text in texts:
        score = _cosine_similarity(query_vec, _embed_text(text)) if query_vec else 0.0
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:k] if text]


def _normalize_memory_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _dedupe_memory_list(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        norm = _normalize_memory_text(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(item)
    return result


def _memory_matches_query(text: str, query: str) -> bool:
    q = query.lower()
    t = text.lower()

    if "about myself" in q or "about me" in q:
        return True

    if "favorite color" in q:
        return "favorite color" in t
    if "favorite" in q and "language" in q:
        return "favorite programming language" in t
    if "favorite" in q and "laptop" in q:
        return "favorite laptop" in t
    if "laptop" in q:
        return "laptop" in t
    if "age" in q:
        return "years old" in t or "age" in t
    if "12" in q or "12th" in q or "12 class" in q:
        return "12th" in t or "12 class" in t
    if "interest" in q:
        return "interested in" in t
    if "like" in q:
        return "likes" in t or "like" in t
    if "plan" in q or "going" in q or "travel" in q:
        return "plans to go" in t
    if "name" in q:
        return "name is" in t
    if "study" in q or "education" in q:
        return "stud" in t or "education" in t

    return True


def _normalize_degree_text(text: str) -> str:
    value = text
    value = re.sub(r"\b(studying|doing)\s+B\.\b", r"\1 B.E.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bB\.E\.E\b", "B.E.", value, flags=re.IGNORECASE)
    return value


def _clean_age_value(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\b(\d{1,2})\b", text)
    return match.group(1) if match else ""


def _clean_favorite_color(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,")


def _clean_name_value(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.split(
        r"\b(i am|i'm|im|doing|studying|from|my|and)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return ""
    return " ".join(words[:4])


def _clean_favorite_language(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.match(r"[A-Za-z]+(?:\+\+|#)?", text)
    if match:
        return match.group(0)
    return text.split()[0] if text.split() else ""


def _clean_favorite_device(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,")


def _is_explicit_remember(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"\b(remember|save)\b", text, re.IGNORECASE))


def _extract_explicit_remember_text(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    patterns = [
        r"^(?:please\s+)?remember\s+that\s+(.*)$",
        r"^(?:please\s+)?remember\s+(.*)$",
        r"^remember:\s*(.*)$",
        r"^(?:please\s+)?save\s+this\s*:\s*(.*)$",
        r"^(?:please\s+)?save\s+this\s+(.*)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _format_memory_item(text: str, name: str) -> str:
    value = _normalize_degree_text(str(text).strip())
    value = re.sub(r"\bgo to go\b", "go to", value, flags=re.IGNORECASE)
    value = value.replace("B.E..", "B.E.")

    if value.lower().startswith("your name is") or value.lower().startswith("user's name is"):
        raw_name = re.sub(r"^your name is\s+|^user's name is\s+", "", value, flags=re.IGNORECASE).strip()
        cleaned = _clean_name_value(raw_name)
        if cleaned:
            value = f"Your name is {cleaned}."

    if value.lower().startswith("your favorite programming language is"):
        raw_lang = re.sub(r"^your favorite programming language is\s+", "", value, flags=re.IGNORECASE).strip()
        cleaned_lang = _clean_favorite_language(raw_lang)
        if cleaned_lang:
            value = f"Your favorite programming language is {cleaned_lang}."

    if value.lower().startswith("your favorite laptop is"):
        raw_laptop = re.sub(r"^your favorite laptop is\s+", "", value, flags=re.IGNORECASE).strip()
        cleaned_laptop = _clean_favorite_device(raw_laptop)
        if cleaned_laptop:
            value = f"Your favorite laptop is {cleaned_laptop}."

    if name:
        value = re.sub(r"^User's\s+", f"{name}'s ", value)
        value = re.sub(r"^User\s+", f"{name} ", value)
        value = re.sub(r"^I am\s+", f"{name} is ", value, flags=re.IGNORECASE)
        value = re.sub(r"^I'm\s+", f"{name} is ", value, flags=re.IGNORECASE)
        value = re.sub(r"^I\s+", f"{name} ", value, flags=re.IGNORECASE)
    else:
        value = re.sub(r"^User's\s+", "Your ", value)
        value = re.sub(r"^User\s+", "You ", value)
        value = re.sub(r"^I am\s+", "You are ", value, flags=re.IGNORECASE)
        value = re.sub(r"^I'm\s+", "You are ", value, flags=re.IGNORECASE)
        value = re.sub(r"^I\s+", "You ", value, flags=re.IGNORECASE)

    value = re.sub(r"\bYou likes\b", "You like", value)
    value = re.sub(r"\bYou does\b", "You do", value)
    value = re.sub(r"\bYou is\b", "You are", value)

    return value


def _memory_display_key(text: str, name: str) -> str:
    value = text.lower().strip()
    if name:
        value = value.replace(name.lower(), "")
    value = re.sub(r"^user's\s+|^user\s+|^you\s+|^your\s+", "", value)
    value = value.replace("does gym", "gym")
    value = value.replace("likes gym", "gym")
    value = re.sub(r"\s+", " ", value)
    return value


def _memory_priority(text: str) -> int:
    t = text.lower()
    if "user's name" in t:
        return 5
    if "stud" in t or "education" in t:
        return 4
    if "interested" in t or "likes" in t or "favorite" in t:
        return 3
    if "plans to go" in t:
        return 2
    return 1


def _parse_memory_ts(value: dict) -> str:
    ts = value.get("ts") if isinstance(value, dict) else None
    return str(ts or "")


def _prune_memory(user_id: str) -> None:
    if not _ensure_memory_store():
        return

    if LTM_MAX_ENTRIES <= 0:
        return

    user_key = _safe_thread_id(user_id)
    ns = _memory_namespace(user_key)
    with _open_memory_store() as store:
        if store is None:
            return
        try:
            items = store.search(ns, limit=200)
        except Exception:
            return

        if len(items) <= LTM_MAX_ENTRIES:
            return

        scored = []
        for item in items:
            value = getattr(item, "value", {}) or {}
            if value.get("kind") == "structured":
                continue
            text = value.get("data", "")
            score = _memory_priority(text)
            ts = _parse_memory_ts(value)
            scored.append((score, ts, item))

        scored.sort(key=lambda x: (x[0], x[1]))
        to_remove = scored[: max(0, len(scored) - LTM_MAX_ENTRIES)]
        for _, __, item in to_remove:
            key = getattr(item, "key", None)
            if key is not None:
                store.delete(ns, key)


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
    if not _ensure_memory_store():
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
        [SystemMessage(content=SUMMARY_PROMPT.format(history=history_text))]
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
            name = getattr(path, "name", str(path))
            print(f"RAG load warning for {name}: {err}")

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
        "my interest",
        "my interests",
        "my favorite",
        "favorite programming language",
        "my hobby",
        "my hobbies",
        "do i like",
        "what do i like",
    ]
    return any(marker in q for marker in markers)


def _extract_name_from_memory(memory_context: str) -> str:
    match = re.search(r"name is ([A-Za-z][A-Za-z\s'-]{0,40})", memory_context, re.IGNORECASE)
    if not match:
        return ""

    raw = match.group(1).strip()
    raw = re.split(r"\b(i am|i'm|doing|studying|currently|from|likes|favorite|plans)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    words = [w for w in re.split(r"\s+", raw) if w]
    if not words or len(words) > 4:
        return ""
    return " ".join(words)


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


def _split_list_terms(text: str) -> list[str]:
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.replace("/", ",")
    cleaned = cleaned.replace(" and ", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return _normalize_list(parts)


def _normalize_interest_list(values: list[str]) -> list[str]:
    cleaned = []
    for item in values or []:
        raw = str(item).strip()
        value = re.sub(r"^learning\s+", "", raw, flags=re.IGNORECASE)
        value = re.sub(r"^leaning\s+", "", value, flags=re.IGNORECASE)
        if value:
            cleaned.append(value)
    return _normalize_list(cleaned)


def _heuristic_structured_memory(text: str) -> dict:
    if not text:
        return {}

    clean = re.sub(r"\s+", " ", text.strip())
    memory = {
        "name": "",
        "education": "",
        "university": "",
        "favorite_language": "",
        "favorite_laptop": "",
        "interests": [],
        "likes": [],
        "projects": [],
    }

    name_match = re.search(
        r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,40})(?=\b(?:i am|i'm|i do|i like|my|from|at|favorite|interest|interested)\b|[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if name_match:
        memory["name"] = name_match.group(1).strip()

    uni_match = re.search(r"\b(?:from|at)\s+([A-Za-z][A-Za-z\s'-]{2,60}university)\b", clean, re.IGNORECASE)
    if uni_match:
        memory["university"] = uni_match.group(1).strip()

    degree_match = re.search(
        r"\b(?:doing|studying|pursuing)\s+([A-Za-z\.\s]{2,40}?)(?=\b(?:in|from|at)\b|[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if degree_match:
        memory["education"] = _normalize_degree_text(degree_match.group(1).strip())

    fav_lang_match = re.search(
        r"\bfavorite programming language is\s+([A-Za-z#+-]+)(?=\s|[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if fav_lang_match:
        memory["favorite_language"] = fav_lang_match.group(1).strip()

    fav_laptop_match = re.search(
        r"\bfavorite laptop is\s+([A-Za-z0-9][A-Za-z0-9\s+-]{1,60})(?=[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if fav_laptop_match:
        memory["favorite_laptop"] = _clean_favorite_device(fav_laptop_match.group(1).strip())

    interest_match = re.search(
        r"\b(?:my\s+interest\s+is|interests?\s+are|interested\s+in)\s+([A-Za-z\s,&/+-]{2,120}?)(?=\b(?:i like|i do|i am|and i|my|from|at)\b|[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if interest_match:
        interest_text = interest_match.group(1).replace("leaning", "learning")
        interests = _split_list_terms(interest_text)
        memory["interests"].extend(_normalize_interest_list(interests))

    likes_match = re.findall(r"\bi (?:really )?like\s+([A-Za-z\s,&/+-]{2,120}?)(?=\b(?:i do|i am|my|and i|from|at)\b|[\.,]|$)", clean, re.IGNORECASE)
    for like in likes_match:
        memory["likes"].extend(_split_list_terms(like))

    if re.search(r"\blearning\b", clean, re.IGNORECASE) and re.search(r"\bai agent", clean, re.IGNORECASE):
        memory["likes"].append("learning to build AI agents")

    if re.search(r"\bbuilding\b", clean, re.IGNORECASE) and re.search(r"\bai agent", clean, re.IGNORECASE):
        memory["likes"].append("building AI agents")

    project_match = re.search(r"\b(done|built|created)\s+([A-Za-z\s-]{2,60}project)", clean, re.IGNORECASE)
    if project_match:
        memory["projects"].append(project_match.group(2).strip())

    memory["interests"] = _normalize_interest_list(memory["interests"])
    memory["likes"] = _normalize_list(memory["likes"])
    memory["projects"] = _normalize_list(memory["projects"])

    return memory


def _heuristic_memories(text: str) -> list[str]:
    if not text:
        return []
    items = []
    clean = re.sub(r"\s+", " ", text.replace(";", ".")).strip()

    clauses = re.split(r"[\.!?]+", clean)
    extra = re.split(r"\b(?:and i|also)\b", clean, flags=re.IGNORECASE)
    parts = [c.strip() for c in clauses + extra if c.strip()]

    name_match = re.search(
        r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,40})(?=\b(?:i am|i'm|i do|i like|my|from|at|favorite|interest|interested)\b|[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if name_match:
        items.append(f"User's name is {name_match.group(1).strip()}.")

    uni_match = re.search(r"\b(?:from|at)\s+([A-Za-z][A-Za-z\s'-]{2,60}university)\b", clean, re.IGNORECASE)
    if uni_match:
        items.append(f"User studies at {uni_match.group(1).strip()}.")

    degree_match = re.search(
        r"\b(?:doing|studying|pursuing)\s+([A-Za-z\.\s]{2,40}?)(?=\b(?:in|from|at)\b|[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if degree_match:
        items.append(f"User is studying {degree_match.group(1).strip()}.")

    for part in parts:
        interest_match = re.search(
            r"\b(?:my\s+interest\s+is|interests?\s+are|interested\s+in)\s+([A-Za-z\s,&/+-]{2,120})",
            part,
            re.IGNORECASE,
        )
        if interest_match:
            items.append(f"User is interested in {interest_match.group(1).strip()}.")

    fav_lang_match = re.search(
        r"\bfavorite programming language is\s+([A-Za-z#+-]+)(?=\s|[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if fav_lang_match:
        items.append(f"User's favorite programming language is {fav_lang_match.group(1).strip()}.")

    fav_laptop_match = re.search(
        r"\bfavorite laptop is\s+([A-Za-z0-9][A-Za-z0-9\s+-]{1,60})(?=[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if fav_laptop_match:
        items.append(f"User's favorite laptop is {_clean_favorite_device(fav_laptop_match.group(1).strip())}.")

    fav_color_match = re.search(r"\bfavorite color is\s+([A-Za-z\s-]{2,30})(?=[\.,]|$)", clean, re.IGNORECASE)
    if fav_color_match:
        items.append(f"User's favorite color is {fav_color_match.group(1).strip()}.")

    likes_match = re.findall(r"\bi (?:really )?like\s+([A-Za-z\s,&/+-]{2,120}?)(?=\b(?:i do|i am|my|and i|from|at)\b|[\.,]|$)", clean, re.IGNORECASE)
    for like in likes_match:
        items.append(f"User likes {str(like).strip()}.")

    hobbies_match = re.findall(r"\bi do\s+([A-Za-z\s,&/+-]{2,120}?)(?=\b(?:i like|i am|my|and i|from|at)\b|[\.,]|$)", clean, re.IGNORECASE)
    for hobby in hobbies_match:
        items.append(f"User does {hobby.strip()}.")

    for part in parts:
        if re.search(r"\blearning\b", part, re.IGNORECASE) and re.search(r"\bai agent", part, re.IGNORECASE):
            items.append("User is learning to build AI agents.")
        if re.search(r"\bbuilding\b", part, re.IGNORECASE) and re.search(r"\bai agent", part, re.IGNORECASE):
            items.append("User likes building AI agents.")
        project_match = re.search(r"\b(done|built|created)\s+([A-Za-z\s-]{2,60}project)", part, re.IGNORECASE)
        if project_match:
            items.append(f"User completed a {project_match.group(2).strip()}.")

    plan_match = re.search(r"\b(?:plan(?:ning)? to|going to|travel(?:ing)? to|trip to)\s+([A-Za-z\s'-]{2,60})", clean, re.IGNORECASE)
    if plan_match:
        items.append(f"User plans to go to {plan_match.group(1).strip()}.")

    age_match = re.search(r"\b(\d{1,2})\s*(?:years? old|yo)\b", clean, re.IGNORECASE)
    if age_match:
        items.append(f"User is {age_match.group(1)} years old.")

    return _dedupe_memory_list(items)


def _clean_memory_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text).strip())
    value = re.sub(r"\s*(,|\.)\s*", r"\1 ", value).strip()

    if value.lower().startswith("user's name is"):
        value = re.split(r"\b(i am|i'm|doing|studying|from|at|favorite|interest|interested)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if not value.endswith("."):
            value += "."

    if "favorite programming language" in value.lower():
        value = re.split(r"\b(my interest|interest|interested|i like|and i)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if not value.endswith("."):
            value += "."

    if "interested in" in value.lower():
        value = re.split(r"\b(i like|and i|i do)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if not value.endswith("."):
            value += "."

    return value


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


def _is_news_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ["news", "headline", "trending", "top trending", "latest"])


def _is_time_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ["time", "date", "today date", "current time", "what is today"])


def _is_stock_query(query: str) -> bool:
    q = query.lower()
    return "stock" in q or "price" in q


def _is_weather_query(query: str) -> bool:
    return "weather" in query.lower()


def _has_location_hint(query: str) -> bool:
    q = query.lower()
    if " in " in q or " at " in q or " of " in q:
        return True
    location = _extract_weather_location(query)
    return bool(location and location.strip())


def _extract_weather_location(query: str) -> str:
    q = query.lower()
    # Pattern 1: "weather in/of/for <location>"
    match = re.search(r"weather\s+(?:in|of|for)\s+([a-z\s]+?)(?:\s+(?:today|tomorrow|yesterday|now|forecast))?[\?\.,]?$", q)
    if match:
        return match.group(1).strip()
    # Pattern 2: "<location> weather" - take just the last word/words before weather, not everything
    match = re.search(r"(?:^|\s)([a-z]+(?:\s+[a-z]+)?)\s+weather", q)
    if match:
        location = match.group(1).strip()
        # Filter out time words that might be captured
        time_words = {"today", "tomorrow", "yesterday", "now", "tonight"}
        words = location.split()
        filtered = [w for w in words if w not in time_words]
        return " ".join(filtered) if filtered else location
    return ""


def _truncate_text(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _extract_first_url(text: str) -> str:
    match = re.search(r"https?://\S+", text or "")
    if not match:
        return ""
    return match.group(0).rstrip(")].,;!")


def _collapse_duplicate_phrase(text: str, phrase: str) -> str:
    if not text or not phrase:
        return text
    doubled = f"{phrase}{phrase}"
    if doubled in text:
        return text.replace(doubled, phrase)
    return text


def _extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)]+", text or "")
    seen = set()
    ordered = []
    for url in urls:
        cleaned = url.rstrip("].,;!")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _format_sources(urls: list[str], fallback: str = "Internal knowledge") -> str:
    if not urls:
        return f"Sources:\n- {fallback}"
    lines = ["Sources:"]
    for url in urls:
        lines.append(f"- {url}")
    return "\n".join(lines)


def _format_search_results(raw_text: str, max_items: int = 5) -> str:
    lines = [l.strip() for l in (raw_text or "").splitlines() if l.strip()]
    bullets = []
    sources = []
    for line in lines[:max_items]:
        url_match = re.search(r"\((https?://[^)]+)\)$", line)
        url = url_match.group(1) if url_match else ""
        if url:
            sources.append(url)
            line = line[: line.rfind("(")].strip()
        parts = [p.strip() for p in line.split(" - ") if p.strip()]
        title = parts[0] if parts else "Result"
        snippet = _truncate_text(" ".join(parts[1:]), 160) if len(parts) > 1 else ""
        if snippet:
            bullets.append(f"- {title}: {snippet}")
        else:
            bullets.append(f"- {title}")
    if not bullets:
        bullets.append("- No results found.")
    return "\n".join(bullets) + "\n\n" + _format_sources(sources, fallback="Web search")



def _extract_stock_symbol(query: str) -> str:
    q = query.lower()
    mapping = {
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "microsoft": "MSFT",
        "apple": "AAPL",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        "tesla": "TSLA",
        "nvidia": "NVDA",
    }
    for name, symbol in mapping.items():
        if name in q:
            return symbol

    match = re.search(r"\b([A-Z]{1,5})\b", query)
    if match:
        return match.group(1).upper()
    return ""


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
            content = _ensure_rag_citations(getattr(forced_answer, "content", ""), rag_context)
            if content != getattr(forced_answer, "content", ""):
                forced_answer = AIMessage(content=content)
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
    search_query = query
    if _is_weather_query(query):
        location = _extract_weather_location(query)
        if location:
            search_query = f"{location} weather today"

    with DDGS() as ddgs:
        results = list(ddgs.text(search_query, max_results=10))

    snippets = []
    seen = set()
    for r in results:
        # Clean up text by fixing spacing issues (multiple spaces, missing spaces)
        body_raw = str(r.get("body", "")).strip()
        body_raw = re.sub(r"\s+", " ", body_raw)  # Fix multiple spaces
        body = _truncate_text(body_raw, 200)
        
        title_raw = str(r.get("title", "")).strip()
        title_raw = re.sub(r"\s+", " ", title_raw)  # Fix multiple spaces
        title = _truncate_text(title_raw, 120)
        
        url = str(r.get("href", "")).strip()
        if _is_weather_query(query):
            text_blob = f"{title} {body}".lower()
            if "weather" not in text_blob:
                continue
        
        # Create unique key for deduplication (normalize for comparison)
        dedup_key = re.sub(r"\s+", " ", f"{title} {body}".lower())[:100]
        if not title or dedup_key in seen:
            continue
        
        seen.add(dedup_key)
        text = " - ".join(part for part in [title, body] if part)
        if url:
            text = f"{text} ({url})"
        snippets.append(text)
        
        # Get up to 5 results
        if len(snippets) >= 5:
            break

    return "\n".join(snippets) if snippets else "No results found."


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
        contexts.append(f"[{i}] {Path(str(source)).name}{page_info}: {chunk}")

    return "\n\n".join(contexts)


def _call_search_tool(query: str) -> str:
    return str(search_tool.invoke({"query": query}))


def _call_stock_tool(symbol: str) -> str:
    return str(get_stock_price.invoke({"symbol": symbol}))


def _call_time_tool() -> str:
    return str(get_current_date_time.invoke({}))


# ==============================
# CHAT NODE
# ==============================
def _get_user_memory_context(user_id: str, query: str) -> str:
    items = _get_user_memory_items(user_id, query)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def _get_user_memory_items(user_id: str, query: str) -> list[str]:
    if not _ensure_memory_store():
        return []

    ns = _memory_namespace(user_id)
    with _open_memory_store() as store:
        if store is None:
            return []
        try:
            items = store.search(ns)
        except Exception as err:
            print(f"LTM search warning: {err}")
            return []

    if not items:
        return []

    structured = _structured_memory_from_items(items)
    structured_texts = _structured_memory_to_sentences(structured)
    auto_texts = []
    unstructured_texts = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        kind = value.get("kind")
        data = value.get("data")
        if not data:
            continue
        if kind == "auto":
            auto_texts.append(str(data))
        elif kind not in {"structured", "summary"}:
            unstructured_texts.append(str(data))

    auto_texts = _dedupe_memory_list([t for t in auto_texts if t])
    unstructured_texts = _dedupe_memory_list([t for t in unstructured_texts if t])

    if _is_self_query(query):
        about_me = "about myself" in query.lower() or "about me" in query.lower()
        if structured_texts:
            if about_me:
                combined = structured_texts + _filter_auto_facts(auto_texts, structured)
                return _dedupe_memory_list(combined)
            filtered = [item for item in structured_texts if _memory_matches_query(item, query)]
            filtered_auto = [item for item in auto_texts if _memory_matches_query(item, query)]
            combined = filtered + filtered_auto
            return _dedupe_memory_list(combined or structured_texts)
        # No structured memory yet
        return _dedupe_memory_list(auto_texts)

    candidates = structured_texts + auto_texts + unstructured_texts
    if not candidates:
        return []
    if query:
        return _search_memory_texts(candidates, query, LTM_TOP_K)
    return candidates


def remember_node(state: ChatState):
    # Reset tool call counter for each new user message
    tools_count = 0  # Reset on each remember_node call (new turn)
    
    if not _ensure_memory_store():
        return {"messages": [], "tools_called_in_turn": tools_count}

    user_id = _safe_thread_id(state.get("user_id") or state.get("thread_id", "default"))
    ns = _memory_namespace(user_id)

    last_text = _latest_user_query(state.get("messages", []))
    if not last_text:
        return {"messages": [], "tools_called_in_turn": tools_count}

    explicit_text = ""
    explicit_requested = _is_explicit_remember(last_text)
    if explicit_requested:
        explicit_text = _extract_explicit_remember_text(last_text)

    with _open_memory_store() as store:
        if store is None:
            return {"messages": [], "tools_called_in_turn": tools_count}
        try:
            items = store.search(ns)
        except Exception as err:
            print(f"LTM search warning: {err}")
            return {"messages": [], "tools_called_in_turn": tools_count}

    existing_structured = _structured_memory_from_items(items)
    structured_summary = _structured_memory_summary(existing_structured)

    try:
        raw = memory_llm.invoke(
            [
                SystemMessage(content=STRUCTURED_MEMORY_PROMPT.format(user_details_content=structured_summary)),
                HumanMessage(content=last_text),
            ]
        ).content
        parsed = _parse_structured_memory_json(raw)
    except Exception as err:
        print(f"LTM extract warning: {err}")
        parsed = {"should_write": False, "memory": {}}

    auto_facts = []
    try:
        raw_auto = memory_llm.invoke(
            [
                SystemMessage(content=AUTO_MEMORY_PROMPT.format(user_details_content=structured_summary)),
                HumanMessage(content=last_text),
            ]
        ).content
        parsed_auto = _parse_auto_memory_json(raw_auto)
        if parsed_auto.get("should_write"):
            auto_facts.extend(parsed_auto.get("facts") or [])
    except Exception as err:
        print(f"Auto memory warning: {err}")

    heuristics = _heuristic_structured_memory(last_text)
    should_write = bool(parsed.get("should_write")) or explicit_requested
    incoming = parsed.get("memory", {}) if should_write else {}
    merged = _merge_structured_memory(existing_structured, incoming)
    merged = _merge_structured_memory(merged, heuristics)

    auto_facts.extend(_heuristic_auto_facts(last_text))
    auto_facts = _filter_auto_facts(auto_facts, merged)

    if merged and merged != existing_structured:
        _store_structured_memory(user_id, merged)
        _prune_memory(user_id)

    if explicit_requested and explicit_text:
        _store_explicit_memory(user_id, explicit_text)
        _prune_memory(user_id)

    if auto_facts:
        _store_auto_memory(user_id, auto_facts)
        _prune_memory(user_id)

    _maybe_store_summary(state.get("messages", []), user_id)
    _prune_memory(user_id)
    return {"messages": [], "tools_called_in_turn": tools_count}


def chat_node(state: ChatState):
    latest_query = _latest_user_query(state["messages"])
    thread_id = _safe_thread_id(state.get("thread_id", "default"))
    user_id = _safe_thread_id(state.get("user_id") or state.get("thread_id", "default"))
    memory_context = _get_user_memory_context(user_id, latest_query)
    last_weather_link = str(state.get("last_weather_link") or "")

    if latest_query and re.search(r"\b(link|that link|the link|share link|give me that link)\b", latest_query.lower()):
        if last_weather_link:
            return {
                "messages": [AIMessage(content=last_weather_link)],
                "allow_tools": False,
            }

    if state.get("pending_weather") and latest_query:
        weather_answer = _call_search_tool(f"weather in {latest_query}")
        if weather_answer:
            weather_link = _extract_first_url(weather_answer)
            summary_prompt = (
                "Summarize the weather in 1-2 short sentences using the provided snippet. "
                "If the snippet lacks a clear forecast, say that only a link is available."
            )
            summary = llm.invoke(
                [
                    SystemMessage(content=summary_prompt),
                    HumanMessage(content=weather_answer),
                ]
            ).content.strip()
            summary = _collapse_duplicate_phrase(summary, "Only a link to the weather forecast is available.")
            sources = [weather_link] if weather_link else _extract_urls_from_text(weather_answer)
            response_text = (
                f"Weather:\n- Location: {latest_query.strip()}\n- Summary: {summary or weather_answer}\n\n"
                f"{_format_sources(sources, fallback='Web search')}"
            )
            return {
                "messages": [AIMessage(content=response_text)],
                "allow_tools": False,
                "pending_weather": False,
                "last_weather_link": weather_link or last_weather_link,
            }
        # If no weather answer, just return the empty message
        return {
            "messages": [
                AIMessage(
                    content="Weather:\n- Summary: Could not fetch weather data.\n\n" + _format_sources([], fallback="Web search")
                )
            ],
            "allow_tools": False,
            "pending_weather": False,
        }

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

    if _is_self_query(latest_query):
        memory_items = _get_user_memory_items(user_id, latest_query)
        if memory_items:
            name = _extract_name_from_memory("\n".join(memory_items))
            greeting = "Sure, "
            seen = set()
            lines = []
            for item in memory_items:
                formatted = _format_memory_item(item, "")
                key = _memory_display_key(formatted, "")
                if not key or key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {formatted}")

            response_text = f"{greeting}here is what I remember about you:\n" + "\n".join(lines)
            return {"messages": [AIMessage(content=response_text)]}

        response_text = (
            "I do not have any saved details about you yet. "
            "Tell me your name, school, or interests and I will remember them."
        )
        return {"messages": [AIMessage(content=response_text)]}

    # Date/time query handling
    if _is_time_query(latest_query):
        now_text = _call_time_tool()
        response_text = f"Date/Time:\n- {now_text}\n\n" + _format_sources([], fallback="System clock")
        return {"messages": [AIMessage(content=response_text)], "allow_tools": False}

    # News/trending query handling
    if _is_news_query(latest_query):
        raw_news = _call_search_tool(latest_query)
        response_text = "Top results:\n" + _format_search_results(raw_news)
        return {"messages": [AIMessage(content=response_text)], "allow_tools": False}

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

    # Weather query handling - try to extract location first
    if _is_weather_query(latest_query):
        location = _extract_weather_location(latest_query)
        if not location:
            # No location found, ask user
            return {
                "messages": [AIMessage(content="Which city are you asking about?")],
                "allow_tools": False,
                "pending_weather": True,
            }
        # Location found, use search tool
        location = location or latest_query
        aqi_hint = " AQI" if "aqi" in latest_query.lower() else ""
        raw_weather = _call_search_tool(f"weather{aqi_hint} in {location}")
        weather_link = _extract_first_url(raw_weather)
        summary_prompt = (
            "Summarize the weather in 1-2 short sentences using the provided snippet. "
            "If the snippet lacks a clear forecast, say that only a link is available."
        )
        summary = llm.invoke(
            [
                SystemMessage(content=summary_prompt),
                HumanMessage(content=raw_weather),
            ]
        ).content.strip()
        summary = _collapse_duplicate_phrase(summary, "Only a link to the weather forecast is available.")
        sources = [weather_link] if weather_link else _extract_urls_from_text(raw_weather)
        response_text = (
            f"Weather:\n- Location: {location}\n- Summary: {summary or raw_weather}\n\n"
            f"{_format_sources(sources, fallback='Web search')}"
        )
        return {
            "messages": [AIMessage(content=response_text)],
            "allow_tools": False,
            "pending_weather": False,
            "last_weather_link": weather_link or last_weather_link,
        }

    if _is_stock_query(latest_query):
        symbol = _extract_stock_symbol(latest_query)
        if symbol:
            stock_text = _call_stock_tool(symbol)
            response_text = f"Stock:\n- {stock_text}\n\n" + _format_sources([], fallback="Yahoo Finance")
            return {
                "messages": [AIMessage(content=response_text)],
                "allow_tools": False,
            }

    # Generic tool-needed queries: run search directly to avoid tool-calling loops.
    if _needs_external_tools(latest_query):
        raw = _call_search_tool(latest_query)
        response_text = "Top results:\n" + _format_search_results(raw)
        return {"messages": [AIMessage(content=response_text)], "allow_tools": False}

    rag_instructions = (
        f"RAG Context (highest priority if relevant):\n{rag_context}\n\n"
        "If RAG context clearly answers the question, answer from it and cite source tags like [1], [2]. "
        "If context is not relevant, then use tools as needed."
    ) if rag_context else "No RAG context available for this query."

    allow_tools = False

    system_prompt = f"""
You are a smart AI assistant.

CRITICAL INSTRUCTION: After receiving results from a tool, STOP immediately. 
Generate your final answer from those results. DO NOT call any tools again in the same conversation turn.

Rules:
- Current response mode: {mode}
- Use get_current_date_time for date/time questions
- If mode is rag_only: answer only from RAG context and clearly say when answer is not found in context
- If mode is agent_only: use tools for web/stock/math and do not rely on RAG context
- If mode is hybrid: prefer RAG context when relevant, otherwise use tools
- When you call a tool and receive results: synthesize them into a clear, concise answer immediately
- Do not request the same tool again in this turn
- Do not call multiple tools sequentially unless absolutely necessary
- Format responses with short bullet points and clear labels
- Always include a "Sources:" section at the end
- Keep responses concise and useful

{rag_instructions}
"""

    if memory_context:
        system_prompt += f"\n\nUser memory (use only if relevant):\n{memory_context}\n"

    short_term_messages = state["messages"][-LTM_STM_MAX_MESSAGES:]
    safe_messages = _filter_messages_for_llm(short_term_messages)
    if not safe_messages and latest_query:
        safe_messages = [HumanMessage(content=latest_query)]
    if not safe_messages:
        return {
            "messages": [AIMessage(content="I need a question to answer. Please ask again.")],
            "allow_tools": False,
            "tools_called_in_turn": state.get("tools_called_in_turn", 0),
        }
    messages = [SystemMessage(content=system_prompt)] + safe_messages

    token = active_thread_id.set(thread_id)
    try:
        response = llm.invoke(messages)
    finally:
        active_thread_id.reset(token)
    if rag_context:
        content = _ensure_rag_citations(getattr(response, "content", ""), rag_context)
        if content != getattr(response, "content", ""):
            response = AIMessage(content=content)

    if not str(getattr(response, "content", "") or "").strip():
        messages = state.get("messages", [])
        last_human_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break
        last_tool = None
        if last_human_idx is not None:
            for msg in reversed(messages[last_human_idx + 1:]):
                if isinstance(msg, ToolMessage):
                    last_tool = msg
                    break
        if last_tool and str(getattr(last_tool, "content", "") or "").strip():
            response = AIMessage(content=str(getattr(last_tool, "content", "")).strip())

    return {"messages": [response], "allow_tools": allow_tools, "tools_called_in_turn": state.get("tools_called_in_turn", 0)}


def route_after_chat(state: ChatState):
    """Route to tools, end, or wait for human based on state and latest message."""
    if state.get("awaiting_approval"):
        return "wait_for_human"
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

        if _ensure_memory_store():
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
                value = str(memory.get(key) or "").strip()

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

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat_node")

builder.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {
        "wait_for_human": END,
        "__end__": END,
    }
)

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


def get_user_memory(user_id: str) -> list[str]:
    """Return stored memory entries for a user."""
    if not _ensure_memory_store():
        return []

    user_key = _safe_thread_id(user_id)
    ns = _memory_namespace(user_key)
    with _open_memory_store() as store:
        if store is None:
            return []
        try:
            items = store.search(ns)
        except Exception:
            return []

    structured = _structured_memory_from_items(items)
    structured_lines = _structured_memory_to_sentences(structured)
    unstructured = [
        item.value.get("data", "")
        for item in items
        if getattr(item, "value", None)
        and item.value.get("data")
        and item.value.get("kind") != "structured"
    ]
    return _dedupe_memory_list(structured_lines + unstructured)


def get_user_memory_count(user_id: str) -> int:
    if not _ensure_memory_store():
        return 0

    user_key = _safe_thread_id(user_id)
    ns = _memory_namespace(user_key)
    with _open_memory_store() as store:
        if store is None:
            return 0
        try:
            items = store.search(ns)
        except Exception:
            return 0

    return len(items)


def cleanup_user_memory(user_id: str) -> int:
    """Remove duplicate memory entries for a user. Returns removed count."""
    if not _ensure_memory_store():
        return 0

    user_key = _safe_thread_id(user_id)
    ns = _memory_namespace(user_key)
    with _open_memory_store() as store:
        if store is None:
            return 0
        try:
            items = store.search(ns)
        except Exception:
            return 0

        seen = set()
        removed = 0
        for item in items:
            value = getattr(item, "value", {}) or {}
            if value.get("kind") == "structured":
                continue
            text = value.get("data", "")
            norm = _normalize_memory_text(text)
            if not norm or norm in seen:
                key = getattr(item, "key", None)
                if key is not None:
                    store.delete(ns, key)
                    removed += 1
                continue
            seen.add(norm)

    return removed


def clear_user_memory(user_id: str) -> int:
    """Delete all LTM entries for a user. Returns removed count."""
    if not _ensure_memory_store():
        return 0

    user_key = _safe_thread_id(user_id)
    ns = _memory_namespace(user_key)
    with _open_memory_store() as store:
        if store is None:
            return 0
        try:
            items = store.search(ns)
        except Exception:
            return 0

        removed = 0
        for item in items:
            key = getattr(item, "key", None)
            if key is not None:
                store.delete(ns, key)
                removed += 1

    return removed


def get_memory_status() -> dict:
    return {
        "available": _ensure_memory_store(),
        "last_error": memory_store_last_error,
        "db_uri": LTM_DB_URI,
    }