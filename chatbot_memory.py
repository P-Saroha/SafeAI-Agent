from __future__ import annotations

import json
import math
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
LTM_IMPORTANCE_THRESHOLD = float(os.getenv("LTM_IMPORTANCE_THRESHOLD", "0.6"))
LTM_DECAY_DAYS = int(os.getenv("LTM_DECAY_DAYS", "30"))
LTM_WEIGHT_BASE = float(os.getenv("LTM_WEIGHT_BASE", "0.6"))
LTM_WEIGHT_FREQ = float(os.getenv("LTM_WEIGHT_FREQ", "0.2"))
LTM_WEIGHT_RECENCY = float(os.getenv("LTM_WEIGHT_RECENCY", "0.2"))
LTM_USAGE_BOOST = float(os.getenv("LTM_USAGE_BOOST", "0.2"))

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
        "location": "",
        "role": "",
        "education": [],
        "university": [],
        "favorite_language": "",
        "favorite_color": "",
        "favorite_laptop": "",
        "favorite_tools": [],
        "response_style": "",
        "tone": "",
        "likes_code_examples": false,
        "learning_preferences": [],
        "interests": [],
        "likes": [],
        "projects": [],
        "travel_plans": [],
        "recent_topics": [],
        "goals": [],
        "current_project": "",
        "issues": [],
        "level": "",
        "skills": []
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
    # Initialize Postgres-backed LTM storage and capture connection errors.
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
    # Lazily re-initialize the LTM store if it was unavailable.
    global memory_store_available
    if memory_store_available:
        return True
    memory_store_available = _init_memory_store()
    return memory_store_available


@contextmanager
def _open_memory_store():
    # Context manager to safely open and close the LTM store.
    if PostgresStore is None:
        yield None
        return
    if not _ensure_memory_store():
        yield None
        return
    with PostgresStore.from_conn_string(LTM_DB_URI) as store:
        yield store


def _memory_namespace(user_id: str) -> tuple:
    # Build the per-user namespace used for LTM keys.
    return ("user", user_id, "details")


def _normalize_list(values: list[str]) -> list[str]:
    # Clean, de-duplicate, and normalize a list of strings.
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


def _coerce_list(value) -> list[str]:
    # Coerce scalars into list form for uniform downstream processing.
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return _split_list_terms(value)
    return []


def _as_list(value) -> list[str]:
    # Wrap a single value in a list when needed.
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    return [text] if text else []


def _structured_memory_summary(memory: dict) -> str:
    # Summarize structured memory into a compact, readable block.
    if not memory:
        return "(empty)"

    parts = []
    for key in [
        "name",
        "age",
        "location",
        "role",
        "favorite_language",
        "favorite_color",
        "favorite_laptop",
        "response_style",
        "tone",
        "current_project",
        "level",
        "likes_code_examples",
    ]:
        value = str(memory.get(key) or "").strip()
        if value:
            parts.append(f"{key}: {value}")

    for key in ["education", "university"]:
        values = _normalize_list(_as_list(memory.get(key)))
        if values:
            parts.append(f"{key}: {', '.join(values)}")

    for key in [
        "favorite_tools",
        "learning_preferences",
        "interests",
        "likes",
        "projects",
        "travel_plans",
        "recent_topics",
        "goals",
        "issues",
        "skills",
    ]:
        values = (
            _normalize_interest_list(memory.get(key) or [])
            if key == "interests"
            else _normalize_list(memory.get(key) or [])
        )
        if values:
            parts.append(f"{key}: {', '.join(values)}")

    return "\n".join(parts) if parts else "(empty)"


def _classify_memory_intent(query: str) -> str:
    # Map a user query to a memory intent bucket.
    q = query.lower()
    if any(term in q for term in ["about me", "about myself", "who am i", "my profile"]):
        return "about_user"
    if any(term in q for term in ["project", "projects", "build", "built", "debug", "bug", "issue"]):
        return "technical"
    if any(term in q for term in ["skill", "skills", "experience", "level"]):
        return "skills"
    if any(term in q for term in ["goal", "goals", "dream", "career", "job", "internship"]):
        return "goals"
    if any(term in q for term in ["prefer", "preference", "tone", "style", "detailed", "short"]):
        return "preferences"
    return "general"


def _select_structured_fields(memory: dict, fields: list[str]) -> dict:
    # Return only the requested fields from structured memory.
    if not memory:
        return {}
    return {key: memory.get(key) for key in fields if key in memory}


def _structured_by_intent(memory: dict, intent: str) -> dict:
    # Filter structured memory based on the detected intent.
    if intent == "about_user":
        return memory
    if intent == "technical":
        return _select_structured_fields(
            memory,
            ["projects", "current_project", "issues", "skills", "education", "university"],
        )
    if intent == "skills":
        return _select_structured_fields(memory, ["level", "skills", "education", "university"])
    if intent == "goals":
        return _select_structured_fields(memory, ["goals", "current_project", "issues"])
    if intent == "preferences":
        return _select_structured_fields(
            memory,
            [
                "response_style",
                "tone",
                "likes_code_examples",
                "favorite_tools",
                "learning_preferences",
                "likes",
                "favorite_language",
                "favorite_color",
                "favorite_laptop",
            ],
        )
    return memory


def _fields_for_intent(intent: str) -> list[str]:
    # Return the list of structured fields relevant to an intent.
    if intent == "technical":
        return ["projects", "current_project", "issues", "skills", "education", "university"]
    if intent == "skills":
        return ["level", "skills", "education", "university"]
    if intent == "goals":
        return ["goals", "current_project", "issues"]
    if intent == "preferences":
        return [
            "response_style",
            "tone",
            "likes_code_examples",
            "favorite_tools",
            "learning_preferences",
            "likes",
            "favorite_language",
            "favorite_color",
            "favorite_laptop",
        ]
    return []


def _try_load_json(text: str) -> dict | None:
    # Parse JSON from model output, with a fallback to substring extraction.
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _parse_structured_memory_json(raw_text: str) -> dict:
    # Normalize the LLM extraction output into a stable structured schema.
    data = _try_load_json(raw_text)
    if not isinstance(data, dict):
        return {"should_write": False, "memory": {}}

    memory = data.get("memory") or {}
    if not isinstance(memory, dict):
        memory = {}

    favorite_tools_raw = _coerce_list(memory.get("favorite_tools") or [])
    learning_raw = _coerce_list(memory.get("learning_preferences") or [])
    interests_raw = _coerce_list(memory.get("interests") or [])
    likes_raw = _coerce_list(memory.get("likes") or [])
    projects_raw = _coerce_list(memory.get("projects") or [])
    travel_raw = _coerce_list(memory.get("travel_plans") or [])
    topics_raw = _coerce_list(memory.get("recent_topics") or [])
    goals_raw = _coerce_list(memory.get("goals") or [])
    issues_raw = _coerce_list(memory.get("issues") or [])
    skills_raw = _coerce_list(memory.get("skills") or [])

    likes_code_examples = memory.get("likes_code_examples")
    if isinstance(likes_code_examples, bool):
        likes_code_examples = "true" if likes_code_examples else "false"
    elif likes_code_examples is None:
        likes_code_examples = ""

    parsed = {
        "name": str(memory.get("name", "")).strip(),
        "age": str(memory.get("age", "")).strip(),
        "location": str(memory.get("location", "")).strip(),
        "role": str(memory.get("role", "")).strip(),
        "education": _normalize_list(_as_list(memory.get("education"))),
        "university": _normalize_list(_as_list(memory.get("university"))),
        "favorite_language": str(memory.get("favorite_language", "")).strip(),
        "favorite_color": str(memory.get("favorite_color", "")).strip(),
        "favorite_laptop": str(memory.get("favorite_laptop", "")).strip(),
        "favorite_tools": _normalize_list(favorite_tools_raw),
        "response_style": str(memory.get("response_style", "")).strip(),
        "tone": str(memory.get("tone", "")).strip(),
        "likes_code_examples": str(likes_code_examples or "").strip(),
        "learning_preferences": _normalize_list(learning_raw),
        "interests": _normalize_interest_list(interests_raw),
        "likes": _normalize_list(likes_raw),
        "projects": _normalize_list(projects_raw),
        "travel_plans": _normalize_list(travel_raw),
        "recent_topics": _normalize_list(topics_raw),
        "goals": _normalize_list(goals_raw),
        "current_project": str(memory.get("current_project", "")).strip(),
        "issues": _normalize_list(issues_raw),
        "level": str(memory.get("level", "")).strip(),
        "skills": _normalize_list(skills_raw),
    }

    return {
        "should_write": bool(data.get("should_write")),
        "memory": parsed,
    }


def _memory_has_facts(memory: dict) -> bool:
    # Check if any usable memory fields exist.
    if not memory:
        return False
    for value in memory.values():
        if isinstance(value, list) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _importance_score_for_field(field: str) -> float:
    # Assign a static importance weight per memory field.
    high = {"name", "education", "university", "skills", "goals", "current_project"}
    medium = {
        "role",
        "location",
        "response_style",
        "tone",
        "learning_preferences",
        "favorite_tools",
        "level",
        "issues",
    }
    low = {"likes", "favorite_color", "favorite_laptop", "travel_plans"}

    if field in high:
        return 0.9
    if field in medium:
        return 0.7
    if field in low:
        return 0.4
    if field in {"interests", "projects", "favorite_language"}:
        return 0.6
    return 0.5


def _value_is_repeated(existing: dict, field: str, value: str) -> bool:
    # Detect whether a value is already stored for a field.
    if not existing:
        return False
    if field in {"education", "university", "favorite_tools", "learning_preferences", "interests", "likes", "projects", "travel_plans", "recent_topics", "goals", "issues", "skills"}:
        existing_list = _normalize_list(existing.get(field) or [])
        return _normalize_memory_text(value) in {_normalize_memory_text(v) for v in existing_list}
    existing_value = str(existing.get(field, "") or "").strip()
    return _normalize_memory_text(existing_value) == _normalize_memory_text(value)


def _filter_by_importance(existing: dict, incoming: dict, force: bool) -> dict:
    # Filter incoming memory by importance score and repetition.
    if force or not incoming:
        return incoming or {}

    filtered: dict = {}
    for field, value in incoming.items():
        score = _importance_score_for_field(field)
        if isinstance(value, list):
            kept = []
            for item in value:
                if not item:
                    continue
                repeated = _value_is_repeated(existing, field, str(item))
                if score >= LTM_IMPORTANCE_THRESHOLD or repeated:
                    kept.append(item)
            if kept:
                filtered[field] = kept
        else:
            text_value = str(value or "").strip()
            if not text_value:
                continue
            repeated = _value_is_repeated(existing, field, text_value)
            if score >= LTM_IMPORTANCE_THRESHOLD or repeated:
                filtered[field] = text_value

    return filtered


def _clean_institution_value(text: str) -> str:
    # Clean school/institution names extracted from free text.
    if not text:
        return ""
    cleaned = re.split(
        r"\b(and|also|i am|i'm|im|doing|studying|my|interest)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;")
    return cleaned


def _fallback_extract_from_text(text: str) -> dict:
    # Regex-based backup extractor for key memory facts.
    result = {
        "name": "",
        "age": "",
        "location": "",
        "role": "",
        "education": [],
        "university": [],
        "favorite_language": "",
        "favorite_color": "",
        "favorite_laptop": "",
        "favorite_tools": [],
        "response_style": "",
        "tone": "",
        "likes_code_examples": "",
        "learning_preferences": [],
        "interests": [],
        "likes": [],
        "projects": [],
        "travel_plans": [],
        "recent_topics": [],
        "goals": [],
        "current_project": "",
        "issues": [],
        "level": "",
        "skills": [],
    }
    if not text:
        return result

    name_match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s'\-]{1,60})", text, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r"\bi'm\s+([A-Za-z][A-Za-z\s'\-]{1,60})", text, re.IGNORECASE)
    if name_match:
        result["name"] = _clean_name_value(name_match.group(1))

    role_match = re.search(r"\b(role|profession)\s*(?:is|:)?\s*([^\.;,]+)", text, re.IGNORECASE)
    if role_match:
        result["role"] = _clean_simple_value(role_match.group(2))

    location_match = re.search(r"\b(?:i am in|i live in|based in|location is)\s+([^\.;,]+)", text, re.IGNORECASE)
    if location_match:
        result["location"] = _clean_simple_value(location_match.group(1))

    interest_match = re.search(r"\bmy interest is\s+([^\.;,]+)", text, re.IGNORECASE)
    if not interest_match:
        interest_match = re.search(r"\bi am interested in\s+([^\.;,]+)", text, re.IGNORECASE)
    if interest_match:
        result["interests"] = _normalize_interest_list([interest_match.group(1).strip()])

    fav_lang_match = re.search(
        r"\bmy favorite\s+(?:programming\s+)?language\s*(?:is|:)?\s*([^\.;,]+)",
        text,
        re.IGNORECASE,
    )
    if fav_lang_match:
        result["favorite_language"] = _clean_favorite_language(fav_lang_match.group(1))

    goal_match = re.search(r"\bgoal\s*(?:is|:)?\s*([^\.;,]+)", text, re.IGNORECASE)
    if goal_match:
        result["goals"] = _normalize_list([goal_match.group(1).strip()])

    edu_matches = re.findall(
        r"\b(B\.?E\.?|B\.?S\.?|B\.?Sc\.?)\b\s*(?:in\s+([A-Za-z0-9&.\s]+?))?(?=\bfrom\b|\band\b|\bmy\b|,|;|$)",
        text,
        flags=re.IGNORECASE,
    )
    education = []
    for degree, field in edu_matches:
        degree_text = _normalize_degree_text(degree)
        field_text = re.sub(r"\.{2,}", ".", str(field or ""))
        field_text = re.sub(r"\s+", " ", field_text.strip(" ."))
        entry = f"{degree_text} in {field_text}".strip() if field_text else degree_text
        if entry:
            education.append(entry)
    result["education"] = _normalize_list(education)

    uni_matches = re.findall(r"\bfrom\s+([A-Za-z][A-Za-z\s.&'\-]{2,60})", text, re.IGNORECASE)
    uni_matches += re.findall(r"\bat\s+([A-Za-z][A-Za-z\s.&'\-]{2,60})", text, re.IGNORECASE)
    universities = []
    for item in uni_matches:
        cleaned = _clean_institution_value(item)
        if cleaned:
            universities.append(cleaned)
    result["university"] = _normalize_list(universities)

    project_match = re.search(r"\bproject\s*(?:is|was|:)?\s*([^\.;,]+)", text, re.IGNORECASE)
    if not project_match:
        project_match = re.search(r"\bbuilt\s+(?:an|a)?\s*([^\.;,]+)", text, re.IGNORECASE)
    if project_match:
        project_text = project_match.group(1).strip()
        project_text = re.sub(r"\bremember\s+(this|it)\b", "", project_text, flags=re.IGNORECASE).strip(" .,:;")
        if project_text:
            result["projects"] = _normalize_list([project_text])

    current_project_match = re.search(r"\bcurrent project\s*(?:is|:)?\s*([^\.;,]+)", text, re.IGNORECASE)
    if current_project_match:
        result["current_project"] = _clean_simple_value(current_project_match.group(1))

    issue_match = re.search(r"\b(issue|bug|problem)\s*(?:is|:)?\s*([^\.;,]+)", text, re.IGNORECASE)
    if issue_match:
        result["issues"] = _normalize_list([issue_match.group(2).strip()])

    skills_match = re.search(r"\bskills?\s*(?:are|:)?\s*([^\.;,]+)", text, re.IGNORECASE)
    if skills_match:
        result["skills"] = _normalize_list(_split_list_terms(skills_match.group(1)))

    style_match = re.search(r"\b(response style|style)\s*(?:is|:)?\s*(short|detailed|concise|brief)\b", text, re.IGNORECASE)
    if style_match:
        result["response_style"] = _clean_simple_value(style_match.group(2))

    tone_match = re.search(r"\btone\s*(?:is|:)?\s*(formal|casual|friendly)\b", text, re.IGNORECASE)
    if tone_match:
        result["tone"] = _clean_simple_value(tone_match.group(1))

    if re.search(r"\blike\s+code\s+examples\b", text, re.IGNORECASE):
        result["likes_code_examples"] = "true"
    if re.search(r"\bprefer\s+fewer\s+code\s+examples\b", text, re.IGNORECASE):
        result["likes_code_examples"] = "false"

    return result


def _structured_memory_to_sentences(memory: dict) -> list[str]:
    # Convert structured memory into readable sentences.
    if not memory:
        return []

    sentences = []
    name = _clean_name_value(memory.get("name") or "")
    age = _clean_age_value(memory.get("age") or "")
    location = _clean_simple_value(memory.get("location") or "")
    role = _clean_simple_value(memory.get("role") or "")
    education_list = [_normalize_degree_text(v).rstrip(".") for v in _as_list(memory.get("education"))]
    education_list = _normalize_list([v for v in education_list if v])
    university_list = _normalize_list(_as_list(memory.get("university")))
    favorite_language = _clean_favorite_language(memory.get("favorite_language") or "")
    favorite_color = _clean_favorite_color(memory.get("favorite_color") or "")
    favorite_laptop = _clean_favorite_device(memory.get("favorite_laptop") or "")
    response_style = _clean_simple_value(memory.get("response_style") or "")
    tone = _clean_simple_value(memory.get("tone") or "")
    likes_code_examples = _normalize_bool_preference(memory.get("likes_code_examples") or "")
    current_project = _clean_simple_value(memory.get("current_project") or "")
    level = _clean_simple_value(memory.get("level") or "")

    if name:
        sentences.append(f"Your name is {name}.")
    if age:
        sentences.append(f"You are {age} years old.")
    if location:
        sentences.append(f"You are based in {location}.")
    if role:
        sentences.append(f"Your role is {role}.")
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
    if response_style:
        sentences.append(f"You prefer {response_style} responses.")
    if tone:
        sentences.append(f"You prefer a {tone} tone.")
    if likes_code_examples == "true":
        sentences.append("You like code examples.")
    if likes_code_examples == "false":
        sentences.append("You prefer fewer code examples.")
    if current_project:
        sentences.append(f"You are working on {current_project}.")
    if level:
        sentences.append(f"Your skill level is {level}.")

    for tool in _normalize_list(memory.get("favorite_tools") or []):
        sentences.append(f"You prefer using {tool}.")
    for pref in _normalize_list(memory.get("learning_preferences") or []):
        sentences.append(f"You prefer {pref} for learning.")

    for interest in _normalize_list(memory.get("interests") or []):
        sentences.append(f"You are interested in {interest}.")
    for like in _normalize_list(memory.get("likes") or []):
        sentences.append(f"You like {like}.")
    for project in _normalize_list(memory.get("projects") or []):
        sentences.append(f"You worked on a {project}.")
    for plan in _normalize_list(memory.get("travel_plans") or []):
        sentences.append(f"You plan to go to {plan}.")
    for topic in _normalize_list(memory.get("recent_topics") or []):
        sentences.append(f"You recently discussed {topic}.")
    for goal in _normalize_list(memory.get("goals") or []):
        sentences.append(f"Your goal is {goal}.")
    for issue in _normalize_list(memory.get("issues") or []):
        sentences.append(f"You are dealing with {issue}.")
    for skill in _normalize_list(memory.get("skills") or []):
        sentences.append(f"You have experience with {skill}.")

    return sentences


def _structured_memory_from_items(items: list) -> dict:
    # Rebuild a structured memory dict from stored LTM items.
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

        if field in {
            "favorite_tools",
            "learning_preferences",
            "interests",
            "likes",
            "projects",
            "travel_plans",
            "recent_topics",
            "goals",
            "issues",
            "skills",
        }:
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
    memory["location"] = _clean_simple_value(memory.get("location", ""))
    memory["role"] = _clean_simple_value(memory.get("role", ""))
    education_list = [_normalize_degree_text(v).rstrip(".") for v in _as_list(memory.get("education"))]
    memory["education"] = _normalize_list([v for v in education_list if v])
    memory["favorite_language"] = _clean_favorite_language(memory.get("favorite_language", ""))
    memory["favorite_color"] = _clean_favorite_color(memory.get("favorite_color", ""))
    memory["favorite_laptop"] = _clean_favorite_device(memory.get("favorite_laptop", ""))
    memory["response_style"] = _clean_simple_value(memory.get("response_style", ""))
    memory["tone"] = _clean_simple_value(memory.get("tone", ""))
    memory["likes_code_examples"] = _normalize_bool_preference(memory.get("likes_code_examples", ""))
    memory["current_project"] = _clean_simple_value(memory.get("current_project", ""))
    memory["level"] = _clean_simple_value(memory.get("level", ""))

    return memory


def _merge_structured_memory(base: dict, incoming: dict) -> dict:
    # Merge new structured facts into existing memory with normalization.
    result = dict(base or {})
    for key in [
        "name",
        "age",
        "location",
        "role",
        "education",
        "university",
        "favorite_language",
        "favorite_color",
        "favorite_laptop",
        "response_style",
        "tone",
        "likes_code_examples",
        "current_project",
        "level",
    ]:
        value = incoming.get(key)
        if key == "name":
            value = _clean_name_value(value)
        if key == "age":
            value = _clean_age_value(value)
        if key in {"location", "role", "response_style", "tone", "current_project", "level"}:
            value = _clean_simple_value(value)
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
        if key == "likes_code_examples":
            value = _normalize_bool_preference(value)
        existing = str(result.get(key, "") or "").strip()
        value = str(value or "").strip()
        if value and value.lower() != existing.lower():
            result[key] = value

    for key in [
        "favorite_tools",
        "learning_preferences",
        "interests",
        "likes",
        "projects",
        "travel_plans",
        "recent_topics",
        "goals",
        "issues",
        "skills",
    ]:
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
    # Persist structured memory fields into the LTM store.
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
        for key in [
            "name",
            "age",
            "location",
            "role",
            "education",
            "university",
            "favorite_language",
            "favorite_color",
            "favorite_laptop",
            "response_style",
            "tone",
            "likes_code_examples",
            "current_project",
            "level",
        ]:
            raw_value = memory.get(key)
            if key == "name":
                value = _clean_name_value(raw_value)
            if key == "age":
                value = _clean_age_value(raw_value)
            if key in {"location", "role", "response_style", "tone", "current_project", "level"}:
                value = _clean_simple_value(raw_value)
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
                            "access_count": 0,
                            "last_accessed": timestamp,
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
                            "access_count": 0,
                            "last_accessed": timestamp,
                        },
                    )
                continue
            if key == "favorite_language":
                value = _clean_favorite_language(raw_value)
            if key == "favorite_color":
                value = _clean_favorite_color(raw_value)
            if key == "favorite_laptop":
                value = _clean_favorite_device(raw_value)
            if key == "likes_code_examples":
                value = _normalize_bool_preference(raw_value)
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
                    "access_count": 0,
                    "last_accessed": timestamp,
                },
            )

        for key in [
            "favorite_tools",
            "learning_preferences",
            "interests",
            "likes",
            "projects",
            "travel_plans",
            "recent_topics",
            "goals",
            "issues",
            "skills",
        ]:
            values = _normalize_list(memory.get(key) or [])
            if not values:
                continue
            if key == "favorite_tools":
                sentence = f"You prefer using {', '.join(values)}."
            elif key == "learning_preferences":
                sentence = f"You prefer {', '.join(values)} for learning."
            elif key == "interests":
                sentence = f"You are interested in {', '.join(values)}."
            elif key == "likes":
                sentence = f"You like {', '.join(values)}."
            elif key == "travel_plans":
                sentence = f"You plan to go to {', '.join(values)}."
            elif key == "recent_topics":
                sentence = f"You recently discussed {', '.join(values)}."
            elif key == "goals":
                sentence = f"Your goals include {', '.join(values)}."
            elif key == "issues":
                sentence = f"You are dealing with {', '.join(values)}."
            elif key == "skills":
                sentence = f"You have experience with {', '.join(values)}."
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
                    "access_count": 0,
                    "last_accessed": timestamp,
                },
            )


def _store_explicit_memory(user_id: str, text: str) -> None:
    # Store explicit user notes as unstructured LTM.
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
                "access_count": 0,
                "last_accessed": datetime.utcnow().isoformat(),
            },
        )


def _search_memory_texts(texts: list[str], query: str, k: int) -> list[str]:
    # Rank raw memory texts by embedding similarity.
    if not texts:
        return []
    query_vec = _embed_text(query)
    scored = []
    for text in texts:
        score = _cosine_similarity(query_vec, _embed_text(text)) if query_vec else 0.0
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:k] if text]


def _structured_entries_from_items(items: list) -> list[dict]:
    # Convert stored structured items into retrieval-ready entries.
    entries = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        if value.get("kind") != "structured":
            continue
        field = value.get("field") or ""
        key = getattr(item, "key", "")
        if not field and isinstance(key, str) and key.startswith("structured:"):
            field = key.split(":", 1)[1]
        sentence = str(value.get("sentence") or "").strip()
        data = value.get("data")
        text = sentence or str(data or "").strip()
        if not text:
            continue
        entries.append({"item": item, "text": text, "field": field, "kind": "structured", "value": value})
    return entries


def _unstructured_entries_from_items(items: list) -> list[dict]:
    # Convert unstructured items into retrieval-ready entries.
    entries = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        kind = value.get("kind")
        if kind in {"structured", "summary"}:
            continue
        data = value.get("data")
        text = str(data or "").strip()
        if not text:
            continue
        entries.append({"item": item, "text": text, "field": "", "kind": kind or "", "value": value})
    return entries


def _rank_memory_entries(entries: list[dict], query: str, k: int) -> list[dict]:
    # Rank entries using similarity plus usage/decay weighting.
    if not entries:
        return []
    query_vec = _embed_text(query) if query else []
    scored = []
    for entry in entries:
        text = entry.get("text", "")
        if not text:
            continue
        sim = _cosine_similarity(query_vec, _embed_text(text)) if query_vec else 0.0
        weight = _memory_weight(entry.get("value", {}), entry.get("kind", ""), entry.get("field", ""), text)
        score = sim + (LTM_USAGE_BOOST * weight)
        scored.append((score, weight, entry))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [entry for _, __, entry in scored[:k]]


def _normalize_memory_text(text: str) -> str:
    # Normalize text for de-duplication comparisons.
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _dedupe_memory_list(items: list[str]) -> list[str]:
    # Remove near-duplicate strings from a memory list.
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
    # Heuristic filter to select memory relevant to a self-query.
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
    if "role" in q or "profession" in q:
        return "role" in t
    if "location" in q or "where" in q or "city" in q:
        return "based in" in t or "location" in t
    if "tone" in q or "style" in q:
        return "tone" in t or "responses" in t
    if "goal" in q:
        return "goal" in t
    if "skill" in q:
        return "experience" in t or "skill" in t
    if "project" in q:
        return "worked on" in t or "project" in t
    if "issue" in q or "bug" in q:
        return "dealing with" in t or "issue" in t

    return True


def _normalize_degree_text(text: str) -> str:
    # Normalize common degree abbreviations.
    value = text
    value = re.sub(r"\b(studying|doing)\s+B\.\b", r"\1 B.E.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bB\.E\.E\b", "B.E.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bB\.?S\.?\b", "B.S.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bB\.?Sc\.?\b", "B.Sc.", value, flags=re.IGNORECASE)
    value = re.sub(r"\.{2,}", ".", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clean_simple_value(value: str) -> str:
    # Trim and normalize a simple string field.
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,")


def _normalize_bool_preference(value) -> str:
    # Normalize boolean-like preferences to "true"/"false" strings.
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return "true"
    if text in {"false", "no", "n"}:
        return "false"
    return ""


def _clean_age_value(value: str) -> str:
    # Extract a clean numeric age value from text.
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\b(\d{1,2})\b", text)
    return match.group(1) if match else ""


def _clean_favorite_color(value: str) -> str:
    # Normalize a color preference string.
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,")


def _clean_name_value(value: str) -> str:
    # Clean and constrain a parsed name value.
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\.{2,}", ".", text).strip(" .")
    text = re.split(
        r"\b(i am|i'm|im|doing|studying|from|my|and|also)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return ""
    return " ".join(words[:4])


def _clean_favorite_language(value: str) -> str:
    # Normalize a programming language label.
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.match(r"[A-Za-z]+(?:\+\+|#)?", text)
    if match:
        return match.group(0)
    return text.split()[0] if text.split() else ""


def _clean_favorite_device(value: str) -> str:
    # Normalize a device or laptop preference string.
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,")


def _is_explicit_remember(text: str) -> bool:
    # Detect whether the user explicitly asked to remember something.
    if not text:
        return False
    return bool(re.search(r"\b(remember|save)\b", text, re.IGNORECASE))


def _extract_explicit_remember_text(text: str) -> str:
    # Extract the payload after "remember" or "save" prompts.
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
            value = match.group(1).strip()
            value = re.sub(r"\bremember\s+(this|it)\b", "", value, flags=re.IGNORECASE).strip(" .,:;")
            return value
    return ""


def _format_memory_item(text: str, name: str) -> str:
    # Format memory text into user-facing sentences.
    value = _normalize_degree_text(str(text).strip())
    value = re.sub(r"\bgo to go\b", "go to", value, flags=re.IGNORECASE)
    value = value.replace("B.E..", "B.E.")
    value = re.sub(r"\.{2,}", ".", value)

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
    # Build a normalized key for de-duping display lines.
    value = text.lower().strip()
    if name:
        value = value.replace(name.lower(), "")
    value = re.sub(r"^user's\s+|^user\s+|^you\s+|^your\s+", "", value)
    value = value.replace("does gym", "gym")
    value = value.replace("likes gym", "gym")
    value = re.sub(r"\s+", " ", value)
    return value


def _memory_priority(text: str) -> int:
    # Assign a rough priority score for unstructured memory.
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
    # Extract the timestamp string from a stored memory item.
    ts = value.get("ts") if isinstance(value, dict) else None
    return str(ts or "")


def _parse_ts_to_datetime(value: str) -> datetime | None:
    # Parse ISO timestamps into datetime objects.
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", ""))
    except Exception:
        return None


def _days_since(ts_value: str) -> float | None:
    # Compute days since a timestamp string.
    dt = _parse_ts_to_datetime(ts_value)
    if dt is None:
        return None
    return (datetime.utcnow() - dt).total_seconds() / 86400.0


def _frequency_score(access_count: int) -> float:
    # Convert access count into a bounded frequency score.
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / math.log1p(10))


def _recency_score(ts_value: str) -> float:
    # Convert timestamp recency into an exponential decay score.
    if LTM_DECAY_DAYS <= 0:
        return 1.0
    days = _days_since(ts_value)
    if days is None:
        return 0.0
    return math.exp(-days / max(1.0, float(LTM_DECAY_DAYS)))


def _memory_weight(value: dict, kind: str, field: str, text: str) -> float:
    # Combine importance, frequency, and recency into a final weight.
    if kind == "structured" and field:
        base = _importance_score_for_field(field)
    else:
        base = min(1.0, max(0.0, _memory_priority(text) / 5.0))

    access_count = 0
    if isinstance(value, dict):
        access_count = int(value.get("access_count") or 0)
        last_accessed = str(value.get("last_accessed") or value.get("ts") or "")
    else:
        last_accessed = ""

    freq = _frequency_score(access_count)
    recency = _recency_score(last_accessed)
    weight = (LTM_WEIGHT_BASE * base) + (LTM_WEIGHT_FREQ * freq) + (LTM_WEIGHT_RECENCY * recency)
    return max(0.0, min(1.0, weight))


def _touch_memory_items(store, ns: tuple, items: list) -> None:
    # Update access_count and last_accessed for retrieved items.
    if not items:
        return
    now = datetime.utcnow().isoformat()
    for item in items:
        value = getattr(item, "value", {}) or {}
        if value.get("kind") == "summary":
            continue
        key = getattr(item, "key", None)
        if not key:
            continue
        updated = dict(value)
        updated["access_count"] = int(updated.get("access_count") or 0) + 1
        updated["last_accessed"] = now
        try:
            store.put(ns, key, updated)
        except Exception:
            continue


def _prune_memory(user_id: str) -> None:
    # Remove low-weight unstructured memories beyond the max limit.
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
            weight = _memory_weight(value, value.get("kind") or "", value.get("field") or "", str(text))
            ts = value.get("last_accessed") or _parse_memory_ts(value)
            scored.append((weight, ts, item))

        scored.sort(key=lambda x: (x[0], x[1]))
        to_remove = scored[: max(0, len(scored) - LTM_MAX_ENTRIES)]
        for _, __, item in to_remove:
            key = getattr(item, "key", None)
            if key is not None:
                store.delete(ns, key)


def _embed_text(text: str) -> list[float]:
    # Compute the embedding vector for a given text.
    try:
        return memory_embeddings.embed_query(text)
    except Exception:
        return []


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    # Compute cosine similarity between two vectors.
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _latest_user_query(messages: list[BaseMessage]) -> str:
    # Return the latest user message text from the message list.
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def _is_self_query(query: str) -> bool:
    # Check if the query is asking about the user.
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
    # Extract a name from the memory context summary.
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
    # Remove tool/function messages before sending to the LLM.
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
    # Split comma/and-separated strings into a list.
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.replace("/", ",")
    cleaned = cleaned.replace(" and ", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return _normalize_list(parts)


def _normalize_interest_list(values: list[str]) -> list[str]:
    # Normalize interest phrases and de-duplicate.
    cleaned = []
    for item in values or []:
        raw = str(item).strip()
        value = re.sub(r"^learning\s+", "", raw, flags=re.IGNORECASE)
        value = re.sub(r"^leaning\s+", "", value, flags=re.IGNORECASE)
        if value:
            cleaned.append(value)
    return _normalize_list(cleaned)


def _extract_stm_facts(messages: list[BaseMessage], exclude_latest: bool = True) -> list[str]:
    # Build STM from recent user turns, with an LLM summary for older history when needed.
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
    # Retrieve LTM items using intent-aware filtering, embedding similarity, and usage/decay weighting.
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

        structured_entries = _structured_entries_from_items(items)
        unstructured_entries = _unstructured_entries_from_items(items)

        intent = _classify_memory_intent(query)
        if intent != "general" and intent != "about_user":
            intent_fields = set(_fields_for_intent(intent))
            structured_entries = [e for e in structured_entries if e.get("field") in intent_fields]

        if _is_self_query(query):
            about_me = "about myself" in query.lower() or "about me" in query.lower()
            if not structured_entries:
                return []
            if about_me:
                selected_entries = structured_entries
            else:
                selected_entries = [
                    entry for entry in structured_entries if _memory_matches_query(entry.get("text", ""), query)
                ]
                if not selected_entries:
                    selected_entries = structured_entries
            _touch_memory_items(store, ns, [entry["item"] for entry in selected_entries])
            return _dedupe_memory_list([entry.get("text", "") for entry in selected_entries if entry.get("text")])

        candidates = structured_entries if intent != "general" else structured_entries + unstructured_entries
        if not candidates:
            return []

        if query:
            ranked = _rank_memory_entries(candidates, query, LTM_TOP_K)
            _touch_memory_items(store, ns, [entry["item"] for entry in ranked])
            return _dedupe_memory_list([entry.get("text", "") for entry in ranked if entry.get("text")])

        candidates_sorted = sorted(
            candidates,
            key=lambda entry: _memory_weight(
                entry.get("value", {}),
                entry.get("kind", ""),
                entry.get("field", ""),
                entry.get("text", ""),
            ),
            reverse=True,
        )
        _touch_memory_items(store, ns, [entry["item"] for entry in candidates_sorted])
        return _dedupe_memory_list([entry.get("text", "") for entry in candidates_sorted if entry.get("text")])


def _get_user_memory_context(user_id: str, query: str) -> str:
    # Convert retrieved memory items into a prompt-ready bullet list.
    items = _get_user_memory_items(user_id, query)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def remember_node(state: dict):
    # Extract structured facts from the latest user message and persist them into LTM.
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

    if explicit_requested and not parsed.get("memory") and explicit_text:
        try:
            explicit_raw = memory_llm.invoke(
                [
                    SystemMessage(content=STRUCTURED_MEMORY_PROMPT.format(user_details_content=structured_summary)),
                    HumanMessage(content=explicit_text),
                ]
            ).content
            parsed = _parse_structured_memory_json(explicit_raw)
        except Exception as err:
            print(f"LTM explicit extract warning: {err}")

    fallback = _fallback_extract_from_text(explicit_text or last_text)
    if _memory_has_facts(fallback):
        merged_fallback = _merge_structured_memory(parsed.get("memory", {}), fallback)
        parsed = {"should_write": True, "memory": merged_fallback}

    has_facts = _memory_has_facts(parsed.get("memory", {}))
    should_write = bool(parsed.get("should_write")) or explicit_requested or has_facts
    incoming = parsed.get("memory", {}) if should_write else {}
    incoming = _filter_by_importance(existing_structured, incoming, explicit_requested)
    merged = _merge_structured_memory(existing_structured, incoming)

    if merged and merged != existing_structured:
        _store_structured_memory(user_id, merged)
        _prune_memory(user_id)

    _prune_memory(user_id)
    return {"messages": [], "tools_called_in_turn": tools_count}


def get_user_memory(user_id: str) -> list[str]:
    # Return a combined list of structured and unstructured memory sentences.
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
    # Count total stored memory items for a user.
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
    # Remove duplicate unstructured memory entries.
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
    # Delete all memory items for a user.
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
    # Report LTM availability and connection details.
    return {
        "available": _ensure_memory_store(),
        "last_error": memory_store_last_error,
        "db_uri": LTM_DB_URI,
    }
