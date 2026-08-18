"""
chatbotBackend.py
-----------------
Main agent orchestrator. Chains together:
1. remember_node: Save facts to long-term memory
2. chat_node: Route intent and generate response
3. Graph checkpoint: Save state to SQLite

See prep/HITL_EXPLANATION.md for details on Human-In-The-Loop pausing.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from typing import Annotated, TypedDict

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
from chatbot_rag_metrics import (
    get_cached_metrics,
)
from chatbot_query_rewriter import (
    get_rag_context_with_rewriting,
)
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


# Config: RAG context below this length triggers HITL approval request
HITL_MIN_CONTEXT_LENGTH = 200

class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str        # unique ID for this conversation
    user_id: str          # unique ID for this user (for long-term memory)

    # ── HITL fields ──────────────────────────────────────────────────────
    # These three fields control the Human-In-The-Loop approval flow.
    awaiting_hitl: bool   # True = bot is paused, waiting for human decision
    hitl_question: str    # The original question that triggered the pause
    hitl_decision: str    # Human's answer: "approve" or "skip"


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _is_document_question(query: str) -> bool:
    """
    Return True only for specific document questions where HITL makes sense.
    Broad queries like 'summarize' or 'what does it say' should ALWAYS go
    straight to RAG — never trigger HITL — because the user clearly wants
    the bot to try regardless of context length.
    """
    q = query.lower()

    # These broad queries mean "try with whatever you have" — never pause
    broad_queries = ["summarize", "summary", "overview", "what does it say",
                     "what is in", "tell me about", "explain this", "describe"]
    if any(kw in q for kw in broad_queries):
        return False

    # Only trigger HITL for specific targeted questions about document content
    specific_keywords = ["according to", "in this pdf", "in this document",
                         "from this file", "the report says", "the paper says"]
    return any(kw in q for kw in specific_keywords)


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

    print(f"[RAG] thread_id={thread_id} | has_docs={has_documents(thread_id)}")

    # ── 1. Greeting ─────────────────────────────────────────────────────
    if is_greeting(query):
        return {"messages": [AIMessage(content="Hello! How can I help you today?")]}

    # ── 2. HITL resume: human has made a decision ────────────────────────
    if state.get("awaiting_hitl") and state.get("hitl_decision"):
        decision = str(state["hitl_decision"]).lower().strip()
        original_question = str(state.get("hitl_question", query))

        if decision == "approve":
            rag_context = get_rag_context(original_question, thread_id)
            prompt = (
                "You are answering from limited document context — the user approved this.\n"
                "Do your best with the context below. Be honest if the answer is unclear.\n"
                "Cite sources like [1], [2] where possible.\n\n"
                f"Document context:\n{rag_context if rag_context else 'No context found.'}\n\n"
                f"Question: {original_question}"
            )
            response = llm.invoke(prompt)
            content = f"{response.content}\n\n{_cite('FAISS + BM25 Hybrid Retriever + Groq (low-confidence approval)')}"
            return {
                "messages": [AIMessage(content=content)],
                "awaiting_hitl": False,
                "hitl_question": "",
                "hitl_decision": "",
            }
        else:
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

    # ── 3. Self-query ────────────────────────────────────────────────────
    if is_self_query(query):
        facts = get_memory_as_text(user_id)
        if facts:
            response = f"Here is what I remember about you:\n\n{facts}\n\n{_cite('PostgreSQL Long-Term Memory')}"
        else:
            response = (
                "I don't have any saved details about you yet. "
                "Tell me your name, interests, or goals and I'll remember them!"
            )
        return {"messages": [AIMessage(content=response)]}

    # ── 4. Weather ───────────────────────────────────────────────────────
    if is_weather_query(query):
        location = extract_weather_location(query)
        if not location:
            return {"messages": [AIMessage(content="Which city would you like the weather for?")]}
        raw = call_weather(location)
        content = format_weather_response(raw, location)
        # format_weather_response already adds a Sources line — append citation below it
        content = f"{content}\n\n{_cite('OpenWeather API')}"
        return {"messages": [AIMessage(content=content)]}

    # ── 5. Date / Time ───────────────────────────────────────────────────
    if is_time_query(query):
        now = call_datetime()
        content = (
            f"### Current Date & Time\n\n"
            f"**{now}**\n\n"
            f"{_cite('System Clock')}"
        )
        return {"messages": [AIMessage(content=content)]}

    # ── 6. News ──────────────────────────────────────────────────────────
    if is_news_query(query):
        results = call_search(query)
        # format_search_response already appends source URLs — add powered-by below
        content = f"{format_search_response(results, query)}\n\n{_cite('DuckDuckGo Search + Groq (gpt-oss-120b)')}"
        return {"messages": [AIMessage(content=content)]}

    # ── 7. Stock price ───────────────────────────────────────────────────
    if is_stock_query(query):
        symbol = extract_stock_symbol(query)
        if symbol:
            result = call_stock(symbol)
            content = (
                f"### Stock Price — {symbol}\n\n"
                f"**{result}**\n\n"
                f"**Data source:** https://finance.yahoo.com/quote/{symbol}\n\n"
                f"{_cite('Yahoo Finance via yfinance')}"
            )
            return {"messages": [AIMessage(content=content)]}
        return {"messages": [AIMessage(
            content="Please include a company name or ticker, e.g. *'stock price of ORCL'*"
        )]}

    # ── 8. RAG ───────────────────────────────────────────────────────────
    if has_documents(thread_id):
        rewritten, rag_context = get_rag_context_with_rewriting(query, thread_id)
        if rewritten != query:
            print(f"[Query Rewrite Fallback] {query} → {rewritten}")

        # Ask user to clarify if query is ambiguous with no context
        if not rag_context and len(query) < 10:
            from chatbot_query_rewriter import is_ambiguous_query
            if is_ambiguous_query(query):
                clarify_message = (
                    "❓ Your question is unclear. Could you provide more details?\n\n"
                    "Example: Instead of 'What is it?', try 'What is LoRA in fine-tuning?'"
                )
                return {"messages": [AIMessage(content=clarify_message)]}

        # HITL: Low confidence retrieval
        if _is_document_question(query) and len(rag_context) < HITL_MIN_CONTEXT_LENGTH:
            pause_message = (
                "⚠️ I found very little relevant content in your uploaded document "
                "for this question.\n\n"
                "Do you want me to try answering with what I found, "
                "or should I skip and let you rephrase / upload a better document?"
            )
            return {
                "messages": [AIMessage(content=pause_message)],
                "awaiting_hitl": True,
                "hitl_question": query,
                "hitl_decision": "",
            }

        if rag_context:
            # Check if context is good quality (coherent, substantial chunks)
            chunk_count = rag_context.count("[")  # Count citations
            context_length = len(rag_context)
            avg_chunk_size = context_length / max(chunk_count, 1) if chunk_count > 0 else 0
            
            # If chunks are substantial (avg > 300 chars each), return with minimal LLM touch
            if avg_chunk_size > 300 and context_length > 1000:
                print(f"[RAG] High-quality context ({context_length} chars, {chunk_count} chunks) - returning directly")
                # Return chunks directly - they already include **Sources:** section
                content = rag_context
            else:
                # Fragments - use LLM to organize
                print(f"[RAG] Fragmented context ({context_length} chars, {chunk_count} chunks) - using LLM to organize")
                system_prompt = (
                    "You are answering based on provided document content.\n"
                    "**CRITICAL RULES:**\n"
                    "1. Use EXACT quotes from the context - do NOT paraphrase or rewrite\n"
                    "2. If the context contains bullet points, preserve them exactly\n"
                    "3. Only combine or organize, NEVER change the wording\n"
                    "4. Use citations [1], [2], etc. for each fact\n"
                    "5. IMPORTANT: Include the **Sources:** section from the context at the end\n\n"
                    "Format: Use bullet points or short paragraphs with exact quotes.\n"
                    "Example: 'Functional Monitoring includes request volume, response times, and error rates [1].'\n\n"
                    f"{rag_context}\n\n"
                    f"Question: {rewritten if rewritten != query else query}\n"
                    "Answer (use exact text with [1], [2] citations, and include **Sources:** section):"
                )
                recent = get_recent_messages(state["messages"])
                response = llm.invoke([SystemMessage(content=system_prompt)] + recent)
                content = response.content
            
            content = f"{content}\n\n{_cite('FAISS + BM25 Hybrid Retriever + Groq (gpt-oss-120b)')}"
            return {"messages": [AIMessage(content=content)]}

    # ── 9. Default LLM answer ────────────────────────────────────────────
    memory_text = get_memory_as_text(user_id)

    system_parts = ["You are a helpful AI assistant. Answer clearly and concisely."]
    if memory_text:
        system_parts.append(f"\nFacts about the user (use only if relevant):\n{memory_text}")

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
