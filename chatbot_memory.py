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
LTM_SUMMARY_EVERY_N = int(os.getenv("LTM_SUMMARY_EVERY_N", "10"))
LTM_SUMMARY_KEEP_LAST = int(os.getenv("LTM_SUMMARY_KEEP_LAST", "6"))
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
    if structured.get("age"):
        skip_terms.extend(["years old", "age"])
    if structured.get("education"):
        skip_terms.extend(["studying", "education", "b.e", "b.s", "b.sc"])
    if structured.get("university"):
        skip_terms.append("university")
    if structured.get("favorite_language"):
        skip_terms.append("favorite programming language")
    if structured.get("favorite_color"):
        skip_terms.append("favorite color")
    if structured.get("favorite_laptop"):
        skip_terms.append("favorite laptop")
    if structured.get("interests"):
        skip_terms.append("interested in")
    if structured.get("likes"):
        skip_terms.append("you like")
    if structured.get("projects"):
        skip_terms.append("worked on")
    if structured.get("travel_plans"):
        skip_terms.append("plan to go")

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
            value = str(memory.get(key) or "").strip()
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

    summary_text = memory_llm.invoke(
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


def _heuristic_structured_memory(text: str) -> dict:
    if not text:
        return {}

    clean = re.sub(r"\s+", " ", text.strip())
    memory = {
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
        "travel_plans": [],
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

    fav_color_match = re.search(
        r"\bfavorite color is\s+([A-Za-z\s-]{2,30})(?=[\.,]|$)",
        clean,
        re.IGNORECASE,
    )
    if fav_color_match:
        memory["favorite_color"] = _clean_favorite_color(fav_color_match.group(1).strip())

    age_match = re.search(
        r"\b(?:my age is|i am)\s+(\d{1,2})\s*(?:years? old|yo)?\b",
        clean,
        re.IGNORECASE,
    )
    if age_match:
        memory["age"] = age_match.group(1).strip()

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

    plan_match = re.search(
        r"\b(?:planning to|plan to|going to)\s+go\s+to\s+([A-Za-z\s'-]{2,60})\b",
        clean,
        re.IGNORECASE,
    )
    if plan_match:
        memory["travel_plans"].append(plan_match.group(1).strip())

    memory["interests"] = _normalize_interest_list(memory["interests"])
    memory["likes"] = _normalize_list(memory["likes"])
    memory["projects"] = _normalize_list(memory["projects"])
    memory["travel_plans"] = _normalize_list(memory["travel_plans"])

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


def _extract_stm_facts(messages: list[BaseMessage], exclude_latest: bool = True) -> list[str]:
    if not messages:
        return []
    human_texts = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            human_texts.append(msg.content)

    if exclude_latest and human_texts:
        human_texts = human_texts[:-1]

    combined = " ".join(human_texts).strip()
    if not combined:
        return []

    stm_structured = _heuristic_structured_memory(combined)
    stm_facts = _structured_memory_to_sentences(stm_structured)
    stm_facts.extend(_heuristic_auto_facts(combined))
    return _dedupe_memory_list([f for f in stm_facts if f])


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
        return _dedupe_memory_list(auto_texts)

    candidates = structured_texts + auto_texts + unstructured_texts
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
