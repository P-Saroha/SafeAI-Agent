from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langgraph.store.postgres import PostgresStore
except Exception:
    PostgresStore = None

from chatbot_rag import _embedding_from_backend, _safe_thread_id

load_dotenv(find_dotenv())

# ==============================
# LTM CONFIG (Postgres + Memory)
# ==============================
LTM_DB_URI = os.getenv(
    "LTM_POSTGRES_URI",
    "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable",
)
LTM_TOP_K = int(os.getenv("LTM_TOP_K", "4"))
LTM_STM_MAX_MESSAGES = int(os.getenv("LTM_STM_MAX_MESSAGES", "12"))
LTM_EMBEDDING_BACKEND = os.getenv("LTM_EMBEDDING_BACKEND", "hash").lower()
LTM_MAX_ENTRIES = int(os.getenv("LTM_MAX_ENTRIES", "25"))

STRUCTURED_MEMORY_PROMPT = """You extract user memory into a strict JSON structure.

EXISTING USER MEMORY:
{user_details_content}

TASK:
- Read the latest user message.
- Extract ONLY new or updated facts that are explicitly stated.
- If nothing new is present, set should_write=false and all fields empty.
- Keep values short, clean, and precise.
- Capture ALL degrees and ALL institutions mentioned. Use arrays when multiple items exist.
- Normalize abbreviations like B.E., E.C.E., ML, AI, DL.
- If the user explicitly asks to remember something, set should_write=true and capture any relevant fields.

Return ONLY valid JSON in this exact format:
{
    "should_write": true|false,
    "memory": {
        "name": "",
        "age": "",
        "education": [],
        "university": [],
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


SUMMARY_PROMPT = """Summarize the following conversation history into short, factual memory notes.
Focus on stable facts about the user, preferences, goals, and ongoing work.
Return 3-6 short bullet-like sentences, each on its own line. No speculation.

Conversation:
{history}
"""

memory_store_last_error = ""


def _init_memory_store() -> bool:
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


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    return [text] if text else []


def _structured_memory_summary(memory: dict) -> str:
    if not memory:
        return "(empty)"

    parts = []
    for key in ["name", "age", "favorite_language", "favorite_color", "favorite_laptop"]:
        value = str(memory.get(key) or "").strip()
        if value:
            parts.append(f"{key}: {value}")

    for key in ["education", "university"]:
        values = _normalize_list(_as_list(memory.get(key)))
        if values:
            parts.append(f"{key}: {', '.join(values)}")

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
        "education": _normalize_list(_as_list(memory.get("education"))),
        "university": _normalize_list(_as_list(memory.get("university"))),
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
    education_list = [_normalize_degree_text(v).rstrip(".") for v in _as_list(memory.get("education"))]
    education_list = _normalize_list([v for v in education_list if v])
    university_list = _normalize_list(_as_list(memory.get("university")))
    favorite_language = _clean_favorite_language(memory.get("favorite_language") or "")
    favorite_color = _clean_favorite_color(memory.get("favorite_color") or "")
    favorite_laptop = _clean_favorite_device(memory.get("favorite_laptop") or "")

    if name:
        sentences.append(f"Your name is {name}.")
    if age:
        sentences.append(f"You are {age} years old.")
    if education_list:
        sentences.append(f"You are studying {'; '.join(education_list)}.")
    if university_list:
        sentences.append(f"You study at {'; '.join(university_list)}.")
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
            data_value = value.get("data", "")
            if field in {"education", "university"}:
                memory[field] = _normalize_list(_as_list(data_value))
            else:
                memory[field] = str(data_value).strip()

    memory["name"] = _clean_name_value(memory.get("name", ""))
    memory["age"] = _clean_age_value(memory.get("age", ""))
    education_list = [_normalize_degree_text(v).rstrip(".") for v in _as_list(memory.get("education"))]
    memory["education"] = _normalize_list([v for v in education_list if v])
    memory["favorite_language"] = _clean_favorite_language(memory.get("favorite_language", ""))
    memory["favorite_color"] = _clean_favorite_color(memory.get("favorite_color", ""))
    memory["favorite_laptop"] = _clean_favorite_device(memory.get("favorite_laptop", ""))

    return memory


def _merge_structured_memory(base: dict, incoming: dict) -> dict:
    result = dict(base or {})
    for key in ["name", "age", "education", "university", "favorite_language", "favorite_color", "favorite_laptop"]:
        value = incoming.get(key)
        if key == "name":
            value = _clean_name_value(value)
        if key == "age":
            value = _clean_age_value(value)
        if key == "education":
            incoming_list = _normalize_list(
                [_normalize_degree_text(v).rstrip(".") for v in _as_list(value)]
            )
            existing_list = _normalize_list(_as_list(result.get("education")))
            merged_list = _normalize_list(existing_list + incoming_list)
            if merged_list:
                result["education"] = merged_list
            continue
        if key == "university":
            incoming_list = _normalize_list(_as_list(value))
            existing_list = _normalize_list(_as_list(result.get("university")))
            merged_list = _normalize_list(existing_list + incoming_list)
            if merged_list:
                result["university"] = merged_list
            continue
        if key == "favorite_language":
            value = _clean_favorite_language(value)
        if key == "favorite_color":
            value = _clean_favorite_color(value)
        if key == "favorite_laptop":
            value = _clean_favorite_device(value)
        existing = str(result.get(key, "") or "").strip()
        value = str(value or "").strip()
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
        for key in ["name", "age", "education", "university", "favorite_language", "favorite_color", "favorite_laptop"]:
            raw_value = memory.get(key)
            if key == "name":
                value = _clean_name_value(raw_value)
            if key == "age":
                value = _clean_age_value(raw_value)
            if key == "education":
                values = _normalize_list(
                    [_normalize_degree_text(v).rstrip(".") for v in _as_list(raw_value)]
                )
                if values:
                    sentence = f"You are studying {'; '.join(values)}."
                    store.put(
                        ns,
                        "structured:education",
                        {
                            "data": values,
                            "list": values,
                            "sentence": sentence,
                            "embedding": _embed_text(sentence),
                            "kind": "structured",
                            "field": "education",
                            "ts": timestamp,
                        },
                    )
                continue
            if key == "university":
                values = _normalize_list(_as_list(raw_value))
                if values:
                    sentence = f"You study at {'; '.join(values)}."
                    store.put(
                        ns,
                        "structured:university",
                        {
                            "data": values,
                            "list": values,
                            "sentence": sentence,
                            "embedding": _embed_text(sentence),
                            "kind": "structured",
                            "field": "university",
                            "ts": timestamp,
                        },
                    )
                continue
            if key == "favorite_language":
                value = _clean_favorite_language(raw_value)
            if key == "favorite_color":
                value = _clean_favorite_color(raw_value)
            if key == "favorite_laptop":
                value = _clean_favorite_device(raw_value)
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

        for key in ["interests", "likes", "projects", "travel_plans"]:
            values = _normalize_list(memory.get(key) or [])
            if not values:
                continue
            if key == "interests":
                sentence = f"You are interested in {', '.join(values)}."
            elif key == "likes":
                sentence = f"You like {', '.join(values)}."
            elif key == "travel_plans":
                sentence = f"You plan to go to {', '.join(values)}."
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
        return "plan to go" in t or "plans to go" in t
    if "name" in q:
        return "name is" in t
    if "study" in q or "education" in q:
        return "stud" in t or "education" in t

    return True


def _normalize_degree_text(text: str) -> str:
    value = text
    value = re.sub(r"\b(studying|doing)\s+B\.\b", r"\1 B.E.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bB\.E\.E\b", "B.E.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bB\.?S\.?\b", "B.S.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bB\.?Sc\.?\b", "B.Sc.", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
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


def _latest_user_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


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


def _extract_stm_facts(messages: list[BaseMessage], exclude_latest: bool = True) -> list[str]:
    if not messages:
        return []

    human_texts = [
        msg.content
        for msg in messages
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
    ]

    if exclude_latest and human_texts:
        human_texts = human_texts[:-1]

    if not human_texts:
        return []

    if len(human_texts) <= LTM_STM_MAX_MESSAGES:
        return _dedupe_memory_list([text.strip() for text in human_texts if text.strip()])

    older = human_texts[:-LTM_STM_MAX_MESSAGES]
    recent = human_texts[-LTM_STM_MAX_MESSAGES:]

    summary_text = ""
    if older:
        history = "\n".join(older)
        try:
            summary_text = memory_llm.invoke(
                [SystemMessage(content=SUMMARY_PROMPT.format(history=history))]
            ).content.strip()
        except Exception:
            summary_text = ""

    stm_items = [text.strip() for text in recent if text.strip()]
    if summary_text:
        stm_items.insert(0, f"Summary of earlier chat: {summary_text}")

    return _dedupe_memory_list(stm_items)


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
    unstructured_texts = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        kind = value.get("kind")
        data = value.get("data")
        if not data:
            continue
        if kind not in {"structured", "summary"}:
            unstructured_texts.append(str(data))

    unstructured_texts = _dedupe_memory_list([t for t in unstructured_texts if t])

    if _is_self_query(query):
        about_me = "about myself" in query.lower() or "about me" in query.lower()
        if structured_texts:
            if about_me:
                return _dedupe_memory_list(structured_texts)
            filtered = [item for item in structured_texts if _memory_matches_query(item, query)]
            return _dedupe_memory_list(filtered or structured_texts)
        return []

    candidates = structured_texts + unstructured_texts
    if not candidates:
        return []
    if query:
        return _search_memory_texts(candidates, query, LTM_TOP_K)
    return candidates


def _get_user_memory_context(user_id: str, query: str) -> str:
    items = _get_user_memory_items(user_id, query)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def remember_node(state: dict):
    tools_count = 0

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
        explicit_text = _extract_explicit_remember_text(last_text) or last_text

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

    should_write = bool(parsed.get("should_write")) or explicit_requested
    incoming = parsed.get("memory", {}) if should_write else {}
    merged = _merge_structured_memory(existing_structured, incoming)

    if merged and merged != existing_structured:
        _store_structured_memory(user_id, merged)
        _prune_memory(user_id)

    if explicit_requested and explicit_text:
        _store_explicit_memory(user_id, explicit_text)
        _prune_memory(user_id)

    _prune_memory(user_id)
    return {"messages": [], "tools_called_in_turn": tools_count}


def get_user_memory(user_id: str) -> list[str]:
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
