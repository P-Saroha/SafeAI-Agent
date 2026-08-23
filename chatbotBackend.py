"""
chatbotBackend.py
-----------------
Main agent orchestrator. Chains together:
1. remember_node: Save facts to long-term memory
2. chat_node: Route intent and generate response
3. Graph checkpoint: Save state to SQLite

"""

from __future__ import annotations

import os
import shutil
import sqlite3
import warnings
import logging
import sys
from typing import Annotated, TypedDict

# Suppress transformers/torchvision warnings (harmless background module inspection)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

from dotenv import load_dotenv

# Load Chatbot/.env first so GROQ_API_KEY is available before any LLM imports
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chatbot_memory import (
    get_latest_user_message,
    get_memory_as_text,
    get_memory_count,
    get_memory_list,
    get_memory_status,
    get_recent_messages,
    is_self_query,
    remember_node,
)
from chatbot_rag import (
    _safe_id,
    get_docs_dir,
    get_index_dir,
    get_rag_context,
    has_documents,
    rebuild_rag_index,
)
from chatbot_query_rewriter import get_rag_context_with_rewriting
from chatbot_tools import (
    call_datetime,
    call_search,
    call_stock,
    call_weather,
    extract_stock_symbol,
    extract_weather_location,
    format_search_response,
    format_weather_response,
    is_greeting,
    is_news_query,
    is_stock_query,
    is_time_query,
    is_weather_query,
    llm,
)


def _cite(source: str) -> str:
    """Add citation footer: > 🔧 **Powered by:** {source}"""
    return f"> 🔧 **Powered by:** {source}"


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str        # unique ID for this conversation
    user_id: str          # unique ID for this user (for long-term memory)
    awaiting_hitl: bool   # True = bot paused, waiting for human decision
    hitl_question: str    # The original question that triggered HITL pause
    hitl_decision: str    # Human's answer: "approve" or "skip"


# ══════════════════════════════════════════════════════════════════════════
# CHAT NODE
# The main function that handles every user message.
# ══════════════════════════════════════════════════════════════════════════

def chat_node(state: ChatState, config: RunnableConfig) -> dict:
    """
    Decide how to respond to the user's latest message.

    The function follows a simple top-to-bottom routing order.
    The first matching condition wins and returns immediately.
    """
    query = get_latest_user_message(state["messages"])

    # Read thread_id and user_id from the LangGraph config (always reliable)
    # Fall back to state fields if config doesn't have them
    configurable = config.get("configurable", {})
    thread_id = _safe_id(
        configurable.get("thread_id")
        or state.get("thread_id")
        or "default"
    )
    user_id = _safe_id(
        configurable.get("user_id")
        or state.get("user_id")
        or thread_id
    )

    print(f"[CHAT_NODE] START | query='{query[:50]}...'")
    print(f"[CHAT_NODE] awaiting_hitl={state.get('awaiting_hitl')} | hitl_decision='{state.get('hitl_decision', '')}'")
    print(f"[RAG] thread_id={thread_id} | has_docs={has_documents(thread_id)}")

    # ── 1. Greeting ─────────────────────────────────────────────────────
    if is_greeting(query):
        return {"messages": [AIMessage(content="Hello! How can I help you today?")]}

    # ── 2. TOOLS FIRST (before RAG) - Weather, Stock, Time, News ─────────
    if is_weather_query(query):
        location = extract_weather_location(query)
        if not location:
            return {"messages": [AIMessage(content="Which city would you like the weather for?")]}
        raw = call_weather(location)
        content = format_weather_response(raw, location)
        content = f"{content}\n\n{_cite('OpenWeather API')}"
        return {"messages": [AIMessage(content=content)]}

    if is_time_query(query):
        now = call_datetime()
        content = (
            f"### Current Date & Time\n\n"
            f"**{now}**\n\n"
            f"{_cite('System Clock')}"
        )
        return {"messages": [AIMessage(content=content)]}

    if is_stock_query(query):
        print(f"[TOOL] Stock query detected: {query[:50]}...")
        symbol = extract_stock_symbol(query)
        if symbol:
            print(f"[TOOL] Extracted symbol: {symbol}")
            result = call_stock(symbol)
            content = (
                f"### Stock Price — {symbol}\n\n"
                f"**{result}**\n\n"
                f"**Data source:** https://finance.yahoo.com/quote/{symbol}\n\n"
                f"{_cite('Yahoo Finance via yfinance')}"
            )
            return {"messages": [AIMessage(content=content)]}
        print(f"[TOOL] No symbol extracted from query")
        return {"messages": [AIMessage(
            content="Please include a company name or ticker, e.g. *'stock price of NVDA'*"
        )]}

    if is_news_query(query):
        results = call_search(query)
        content = f"{format_search_response(results, query)}\n\n{_cite('DuckDuckGo Search + Groq')}"
        return {"messages": [AIMessage(content=content)]}

    # ── 3. HITL RESUME - human made a decision ─────────────────────────
    # Check BEFORE processing the question again (avoids re-triggering HITL)
    if state.get("awaiting_hitl") and state.get("hitl_decision"):
        print(f"[HITL_RESUME] User decision: '{state.get('hitl_decision')}'")
        decision = str(state["hitl_decision"]).lower().strip()
        original_question = str(state.get("hitl_question", query))
        print(f"[HITL_RESUME] Processing decision='{decision}' for question='{original_question[:50]}...'")

        if decision == "approve":
            print(f"[HITL_RESUME] APPROVE: Getting RAG context...")
            rag_context = get_rag_context(original_question, thread_id)
            
            print(f"[HITL_RESUME] Calling Groq LLM for formatting...")
            prompt = (
                "You are answering a question based ONLY on provided document chunks.\n\n"
                "INSTRUCTIONS:\n"
                "1. Use ONLY the chunk information provided\n"
                "2. Structure clearly: ### headers, **bold**, bullet points\n"
                "3. PRESERVE CITATIONS: Every fact must have [1] [2] or [3] with PDF name and page\n"
                "4. If chunks don't fully answer, say 'Limited information available: [cite]\n\n"
                f"QUESTION: {original_question}\n\n"
                f"DOCUMENT CHUNKS (with [1] [2] [3] citations):\n{rag_context}\n\n"
                "ANSWER (with citations preserved):"
            )
            response = llm.invoke(prompt)
            
            content = f"{response.content}\n\n{_cite('Document Retrieval with Citations')}"
            print(f"[HITL_RESUME] APPROVE complete, clearing awaiting_hitl")
            return {
                "messages": [AIMessage(content=content)],
                "awaiting_hitl": False,
                "hitl_question": "",
                "hitl_decision": "",
            }
        else:
            print(f"[HITL_RESUME] SKIP")
            return {
                "messages": [AIMessage(
                    content=(
                        "Understood. I don't have enough information in the "
                        "uploaded document to answer confidently. "
                        "Try uploading a more relevant document or rephrase your question."
                    )
                )],
                "awaiting_hitl": False,
                "hitl_question": "",
                "hitl_decision": "",
            }

    # ── 3. RAG (PRIORITIZE DOCUMENTS) ────────────────────────────────────
    if has_documents(thread_id):
        rewritten, rag_context = get_rag_context_with_rewriting(query, thread_id)
        
        # No context found - ask for clarification if ambiguous
        if not rag_context and len(query) < 10:
            from chatbot_query_rewriter import is_ambiguous_query
            if is_ambiguous_query(query):
                return {"messages": [AIMessage(content="Your question is unclear. Could you provide more details?")]}
        
        # Low confidence context - trigger HITL ONLY ONCE
        if rag_context and len(rag_context) < 200:
            print(f"[HITL_TRIGGER] Low confidence! rag_context length={len(rag_context)} < 200")
            print(f"[HITL_TRIGGER] Current awaiting_hitl={state.get('awaiting_hitl')}")
            if not state.get("awaiting_hitl"):
                print(f"[HITL_TRIGGER] Setting awaiting_hitl=True, storing hitl_question='{query[:50]}...'")
                return {
                    "messages": [AIMessage(content="Found limited context. Do you want me to answer with this, or rephrase?")],
                    "awaiting_hitl": True,
                    "hitl_question": rewritten or query,
                    "hitl_decision": "",
                }
            else:
                print(f"[HITL_TRIGGER] Already awaiting HITL, skipping (avoid loop)")

        
        # Good context - pass through LLM for better formatting
        if rag_context:
            print(f"[RAG] Good context found ({len(rag_context)} chars), calling LLM for formatting...")
            prompt = (
                "You are formatting retrieved document chunks into a well-structured response.\n\n"
                "CRITICAL: Preserve ALL citations [1], [2], [3] with PDF filenames and page numbers.\n\n"
                "INSTRUCTIONS:\n"
                "1. Answer the question using ONLY the provided chunks\n"
                "2. Use clear formatting: ### headers, **bold**, bullet points\n"
                "3. PRESERVE citations exactly as shown: [1] filename.pdf (Page X)\n"
                "4. Every fact must have a citation [1] or [2] or [3] inline\n"
                "5. Keep answer concise but complete\n"
                "6. Do NOT add information beyond what's in chunks\n"
                "7. At the end, repeat the Sources list from chunks\n\n"
                f"QUESTION: {query}\n\n"
                f"DOCUMENT CHUNKS (already formatted with proper citations):\n{rag_context}\n\n"
                "FORMATTED RESPONSE (preserve all citations [1] [2] [3] with filenames and pages!):"
            )
            formatted = llm.invoke(prompt)
            
            content = f"{formatted.content}\n\n{_cite('Document Chunks + LLM Formatting')}"
            print(f"[RAG] RAG answer complete")
            return {"messages": [AIMessage(content=content)]}

    # ── 4. Self-query ────────────────────────────────────────────────────
    if is_self_query(query):
        facts = get_memory_as_text(user_id)
        if facts:
            # Use Groq to format facts beautifully and structurally
            format_prompt = f"""
You are a personal profile formatter. Given user facts, create a clean, structured markdown profile.

Format ONLY as:
👤 **Profile Overview**
- **Name:** [name]
- **Education:** [education]
- **University:** [university]
- **Interests:** [interests in bullet list]

That's it. No learning path. No resources. Just the profile.

User Facts:
{facts}

Create the profile now:
"""
            formatted_profile = llm.invoke(format_prompt)
            response = f"{formatted_profile.content}\n\n{_cite('PostgreSQL Long-Term Memory + Groq Formatting')}"
        else:
            response = (
                "I don't have any saved details about you yet. "
                "Tell me your name, interests, or goals and I'll remember them!"
            )
        return {"messages": [AIMessage(content=response)]}

    # ── 5. Default LLM answer ────────────────────────────────────────────
    # But check if this looks like a personal statement (not a question)
    if not any(q in query.lower() for q in ["?", "what", "how", "why", "when", "where", "can you", "could you", "summary", "recap", "tell me", "explain"]):
        # This is likely a personal statement ("my interests are...", "I love...", etc)
        # Just acknowledge it briefly — LTM will extract the facts
        return {"messages": [AIMessage(
            content="Got it! I've noted that. 📝"
        )]}
    
    memory_text = get_memory_as_text(user_id)

    system_parts = ["You are a helpful AI assistant. Answer clearly and concisely."]
    if memory_text:
        system_parts.append(
            f"\n🎯 About the user (personalize your response using this context):\n{memory_text}\n\n"
            f"💡 Tip: When answering, relate your response to their interests and experience level. "
            f"For example, if they're interested in RAG/ML, mention relevant frameworks or techniques."
        )

    system_prompt = "\n".join(system_parts)
    recent = get_recent_messages(state["messages"])
    response = llm.invoke([SystemMessage(content=system_prompt)] + recent)
    content = f"{response.content}\n\n{_cite('Groq (gpt-oss-120b)')}"
    return {"messages": [AIMessage(content=content)]}


# ══════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# After chat_node runs, this decides which node comes next.
# If we are waiting for human input, we go to END (graph pauses).
# Otherwise, we also go to END (graph is done for this turn).
#
# The key difference: when awaiting_hitl=True, the graph state is SAVED
# but the conversation is considered "incomplete". The frontend detects
# this and shows the Approve/Skip buttons to the user.
# ══════════════════════════════════════════════════════════════════════════

def route_after_chat(state: ChatState) -> str:
    """Return 'end' always — the graph pauses naturally via checkpointing."""
    return END


# ══════════════════════════════════════════════════════════════════════════
# SQLITE CHECKPOINTER
# Saves conversation state to a local SQLite file so chats persist
# across app restarts. This is also what makes HITL possible —
# the graph state (including awaiting_hitl=True) is saved here,
# so the frontend can read it and show the approval buttons.
# ══════════════════════════════════════════════════════════════════════════

def _init_checkpointer():
    """Create (or recover) the SQLite checkpointer."""
    db_path = os.path.join(os.path.dirname(__file__), "chatbot_db")
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("SELECT 1")
        return SqliteSaver(conn=conn), conn
    except Exception as e:
        print(f"Checkpoint DB error: {e}. Creating a fresh one.")
        if os.path.exists(db_path):
            os.rename(db_path, f"{db_path}.bak")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn=conn), conn


checkpointer, _db_conn = _init_checkpointer()


# ══════════════════════════════════════════════════════════════════════════
# LANGGRAPH — BUILD THE AGENT GRAPH
#
# Flow:  START → remember → chat → END
#
#   remember : extract facts from the user message and save to LTM
#   chat     : decide the response (may set awaiting_hitl=True to pause)
# ══════════════════════════════════════════════════════════════════════════

_builder = StateGraph(ChatState)
_builder.add_node("remember", remember_node)
_builder.add_node("chat", chat_node)
_builder.add_edge(START, "remember")
_builder.add_edge("remember", "chat")
_builder.add_edge("chat", END)

chatbot = _builder.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════
# THREAD UTILITIES
# Helper functions used by the Streamlit frontend.
# ══════════════════════════════════════════════════════════════════════════

def list_all_threads() -> list[str]:
    """Return all thread IDs that have saved checkpoints."""
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])

    docs_root = get_docs_dir("_placeholder_").parent
    if docs_root.exists():
        for path in docs_root.iterdir():
            if path.is_dir():
                threads.add(path.name)

    return list(threads)


def get_thread_hitl_state(thread_id: str, user_id: str) -> dict:
    """
    Read the current graph state for a thread and return the HITL fields.
    The frontend calls this on page load to check if an approval is pending.

    Returns a dict like:
      {"awaiting": True, "question": "What does the PDF say about X?"}
    or
      {"awaiting": False}
    """
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id, "user_id": user_id}}
        )
        values = state.values or {}
        if values.get("awaiting_hitl"):
            return {
                "awaiting": True,
                "question": values.get("hitl_question", ""),
            }
    except Exception:
        pass
    return {"awaiting": False}


def delete_thread(thread_id: str) -> str:
    """
    Delete a conversation thread:
    - Remove its checkpoints from SQLite.
    - Delete its uploaded documents.
    - Delete its FAISS index.
    """
    if not thread_id or not thread_id.strip():
        return "Delete failed: invalid thread ID."

    tid = _safe_id(thread_id)
    deleted_rows = 0

    try:
        cur = _db_conn.cursor()
        tables = [
            r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for table in tables:
            columns = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")').fetchall()]
            if "thread_id" not in columns:
                continue
            cur.execute(f'DELETE FROM "{table}" WHERE thread_id = ?', (tid,))
            deleted_rows += cur.rowcount or 0
        _db_conn.commit()
    except Exception as e:
        return f"Delete failed: {e}"

    for folder in [get_docs_dir(tid), get_index_dir(tid)]:
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    return f"Chat deleted ({deleted_rows} checkpoint rows removed)."


def generate_chat_title(first_message: str) -> str:
    """Generate a short title for a chat from the user's first message."""
    if not first_message or not first_message.strip():
        return "New Chat"
    try:
        prompt = (
            "Generate a very short chat title (2-5 words, no punctuation) "
            "for this user message. Return only the title.\n\n"
            f"Message: {first_message[:200]}"
        )
        title = llm.invoke(prompt).content.strip().strip('"\'.,!?')
        return title[:50] if title else " ".join(first_message.split()[:5])
    except Exception:
        return " ".join(first_message.split()[:5]) or "New Chat"


# ══════════════════════════════════════════════════════════════════════════
# RAG METRICS INTEGRATION
# Retrieve evaluation metrics for a thread (cached results from latest eval)
# ══════════════════════════════════════════════════════════════════════════

def get_rag_metrics(thread_id: str) -> dict:
    """
    Get cached RAG metrics for a thread (if an evaluation has been run).
    
    This returns the results from the most recent evaluate_retriever() call.
    
    Returns:
        Dict with metrics (hit_rate@5, hit_rate@10, mrr, latency_ms, etc.)
        or empty dict {} if no evaluation has been run yet.
    
    Example:
      >>> metrics = get_rag_metrics("thread-123")
      >>> if metrics:
      ...     print(f"Hit Rate@5: {metrics['hit_rate@5']:.2%}")
      ... else:
      ...     print("No evaluation data yet")
    """
    return get_cached_metrics(thread_id)
