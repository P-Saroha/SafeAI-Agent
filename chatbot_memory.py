"""
chatbot_memory.py
-----------------
Handles two kinds of memory:

1. Short-Term Memory (STM)
   - The last few messages in the current conversation.
   - Passed directly to the LLM so it knows recent context.

2. Long-Term Memory (LTM)
   - Facts about the user stored in a Postgres database.
   - Things like name, education, interests, goals, etc.
   - Persists across different chat sessions.
   - Requires Docker Postgres to be running (see docker-compose.yml).

How it works:
  - Every time the user sends a message, the remember_node() function
    asks the LLM to extract any new facts from that message.
  - Those facts are saved to Postgres.
  - When the user asks "what do you know about me?", we read them back.
"""

import json
import os
import re
import warnings
import logging

# Suppress ZoeDepth and torchvision warnings BEFORE any imports
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load Chatbot/.env first, then fall back to root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(find_dotenv())

# ── Config ──────────────────────────────────────────────────────────────────
STM_MAX_MESSAGES = int(os.getenv("LTM_STM_MAX_MESSAGES", "12"))

LTM_DB_URI = os.getenv(
    "LTM_POSTGRES_URI",
    "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable",
)

# ── Optional import ─────────────────────────────────────────────────────────
try:
    from langgraph.store.postgres import PostgresStore
    _POSTGRES_AVAILABLE = True
except ImportError:
    PostgresStore = None
    _POSTGRES_AVAILABLE = False

# LLM for extracting memory facts — Groq via OpenAI-compatible API
_memory_llm = ChatOpenAI(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)

# Track whether Postgres connected successfully
_ltm_available = False
_ltm_error = ""


# ══════════════════════════════════════════════════════════════════════════
# LTM CONNECTION
# ══════════════════════════════════════════════════════════════════════════

def _init_ltm() -> bool:
    """Try to connect to Postgres and set up the LTM tables. Returns True on success."""
    global _ltm_available, _ltm_error
    if not _POSTGRES_AVAILABLE:
        _ltm_error = "langgraph.store.postgres is not installed."
        return False
    try:
        with PostgresStore.from_conn_string(LTM_DB_URI) as store:
            store.setup()
        _ltm_available = True
        _ltm_error = ""
        return True
    except Exception as e:
        _ltm_available = False
        _ltm_error = str(e)
        return False


_init_ltm()  # Try to connect when the module loads


def _ensure_ltm() -> bool:
    """Return True if LTM is available (tries to reconnect if not)."""
    global _ltm_available
    if _ltm_available:
        return True
    _ltm_available = _init_ltm()
    return _ltm_available


def _ltm_namespace(user_id: str) -> tuple:
    """Each user gets their own namespace in the Postgres store."""
    return ("user", user_id, "memory")


# ══════════════════════════════════════════════════════════════════════════
# LTM READ / WRITE
# ══════════════════════════════════════════════════════════════════════════

def _save_memory(user_id: str, facts: dict) -> None:
    """
    Save a dictionary of user facts to Postgres.
    Each fact is stored as a separate key so it can be updated independently.
    Example facts: {"name": "Alice", "interests": ["Python", "ML"]}
    """
    if not _ensure_ltm():
        return

    ns = _ltm_namespace(user_id)
    with PostgresStore.from_conn_string(LTM_DB_URI) as store:
        for field, value in facts.items():
            if not value:
                continue
            # Convert lists to a comma-separated string for simple storage
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value if v)
            if str(value).strip():
                store.put(ns, field, {"value": str(value).strip()})


def _load_memory(user_id: str) -> dict:
    """
    Load all saved facts for a user from Postgres.
    Returns a dict like {"name": "Alice", "interests": "Python, ML"}.
    """
    if not _ensure_ltm():
        return {}

    ns = _ltm_namespace(user_id)
    try:
        with PostgresStore.from_conn_string(LTM_DB_URI) as store:
            items = store.search(ns)
            return {
                getattr(item, "key", ""): (getattr(item, "value", {}) or {}).get("value", "")
                for item in items
                if getattr(item, "key", "")
            }
    except Exception as e:
        print(f"LTM load error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════
# FACT EXTRACTION
# Uses the LLM to pull structured facts out of the user's message.
# ══════════════════════════════════════════════════════════════════════════

_EXTRACT_PROMPT = """You extract facts about a user from their message.

Return ONLY valid JSON. If no new facts are present, return {}.

Example output:
{
  "name": "Alice",
  "age": "22",
  "location": "Delhi",
  "education": "B.E. in Computer Science",
  "university": "VIT",
  "interests": "Machine Learning, Python",
  "goals": "become a software engineer",
  "skills": "Python, SQL",
  "current_project": "AI chatbot",
  "favorite_language": "Python",
  "role": "student"
}

Only include fields that are explicitly mentioned. Do not guess.
"""


def _extract_facts(user_message: str) -> dict:
    """Ask the LLM to extract any user facts from a single message."""
    if not user_message.strip():
        return {}
    try:
        response = _memory_llm.invoke([
            SystemMessage(content=_EXTRACT_PROMPT),
            HumanMessage(content=user_message),
        ])
        raw = str(response.content).strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY (STM)
# ══════════════════════════════════════════════════════════════════════════

def get_recent_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Return the last STM_MAX_MESSAGES messages, keeping only Human and AI messages.
    Tool messages and function messages are excluded.
    """
    clean = [
        msg for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage))
    ]
    return clean[-STM_MAX_MESSAGES:]


def get_latest_user_message(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent Human message."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


# ══════════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY (LTM) — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def get_memory_as_text(user_id: str) -> str:
    """
    Return all stored user facts as a short bullet list for injection into
    the system prompt.

    Example:
      - name: Alice
      - interests: Python, ML
    """
    facts = _load_memory(user_id)
    if not facts:
        return ""
    lines = [f"- {k}: {v}" for k, v in facts.items() if v]
    return "\n".join(lines)


def get_memory_list(user_id: str) -> list[str]:
    """Return stored facts as a list of 'key: value' strings (for the UI)."""
    facts = _load_memory(user_id)
    return [f"{k}: {v}" for k, v in facts.items() if v]


def get_memory_count(user_id: str) -> int:
    """Return how many fact entries are stored for this user."""
    return len(_load_memory(user_id))


def clear_memory(user_id: str) -> int:
    """Delete all stored facts for a user. Returns the number of items removed."""
    if not _ensure_ltm():
        return 0
    ns = _ltm_namespace(user_id)
    removed = 0
    try:
        with PostgresStore.from_conn_string(LTM_DB_URI) as store:
            items = store.search(ns)
            for item in items:
                key = getattr(item, "key", None)
                if key:
                    store.delete(ns, key)
                    removed += 1
    except Exception as e:
        print(f"LTM clear error: {e}")
    return removed


def get_memory_status() -> dict:
    """Return a status dict showing whether LTM is connected."""
    return {
        "available": _ensure_ltm(),
        "last_error": _ltm_error,
    }


def generate_recap_greeting(user_id: str) -> str:
    """
    Generate a personalized welcome-back message using the user's stored LTM facts.

    Example output:
      "Welcome back, Sara! Last time you were working on your AI chatbot project.
       What are we working on today?"

    Returns an empty string if no facts are stored (so we can skip showing it
    for first-time users).
    """
    facts = _load_memory(user_id)
    if not facts:
        return ""  # First-time user — no recap to show

    # Build a short readable summary of what we know
    fact_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items() if v)

    try:
        response = _memory_llm.invoke([
            SystemMessage(content=(
                "You are a friendly AI assistant greeting a returning user. "
                "Using the stored facts below, write a short warm welcome-back message "
                "(2-3 sentences max). "
                "Mention their name if known, reference something specific like their "
                "current project, goals, or interests. "
                "End with a question like 'What are we working on today?' or "
                "'What can I help you with today?'. "
                "Do NOT list all facts — just pick 1-2 natural highlights. "
                "Keep it conversational, not robotic."
            )),
            HumanMessage(content=f"User facts:\n{fact_lines}"),
        ])
        greeting = str(response.content).strip()
        return greeting if greeting else ""
    except Exception:
        # Fallback to a simple name-based greeting if LLM call fails
        name = facts.get("name", "")
        if name:
            return f"Welcome back, {name}! What can I help you with today?"
        return "Welcome back! What can I help you with today?"


def is_self_query(query: str) -> bool:
    """Return True if the user is ASKING about themselves or their stored info.
    
    NOTE: Statements like "my interests are..." are NOT self-queries.
    Only questions like "what do you know about me?" are self-queries.
    Statements are handled by remember_node (LTM extraction).
    """
    q = query.lower().strip()
    markers = [
        "about me", "about myself", "who am i", "my details",
        "my profile", "what do you know about me", "what have you learned",
        "remember", "remind me", "tell me about me",
    ]
    return any(m in q for m in markers)


# ══════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE
# This is called automatically on every user message to save new facts.
# ══════════════════════════════════════════════════════════════════════════

def remember_node(state: dict) -> dict:
    """
    LangGraph node: extract facts from the latest user message and save them.
    OPTIMIZED: Skip extraction for non-informative queries (greetings, tools, etc).
    Returns empty messages so graph can continue normally.
    """
    if not _ensure_ltm():
        return {"messages": []}

    user_id = state.get("user_id") or state.get("thread_id", "default")
    latest = get_latest_user_message(state.get("messages", []))

    if not latest:
        return {"messages": []}

    # OPTIMIZATION: Skip fact extraction only for very short greetings
    # But allow extraction for messages with personal info
    query_lower = latest.lower()
    
    # Only skip if it's a PURE greeting (max 2-3 words, no personal info)
    pure_greetings = ["hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "yes", "no", "sure"]
    words = query_lower.split()
    
    # If message is ONLY a greeting (max 3 words, all are greetings), skip extraction
    is_pure_greeting = len(words) <= 3 and all(w.strip("!.?,") in pure_greetings for w in words)
    
    if is_pure_greeting:
        return {"messages": []}  # SKIP fact extraction for pure greetings - saves 600-800ms
    
    # Also skip if message is asking a question (what, how, why, tell me, etc)
    # These rarely contain personal facts
    question_keywords = ["what", "how", "why", "when", "where", "can you", "could you", "tell me", "show me", "explain"]
    is_question = any(q in query_lower for q in question_keywords)
    if is_question:
        return {"messages": []}  # SKIP fact extraction for questions - saves 600-800ms

    # Extract new facts from this message (only for potentially personal info)
    new_facts = _extract_facts(latest)

    if new_facts:
        # Merge with existing facts (new values overwrite old ones)
        existing = _load_memory(user_id)
        merged = {**existing, **new_facts}
        _save_memory(user_id, merged)

    return {"messages": []}
