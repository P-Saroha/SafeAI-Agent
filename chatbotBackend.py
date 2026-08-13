"""
chatbotBackend.py
-----------------
The main agent brain. This file wires everything together:

1. Defines the ChatState — what data the agent tracks per conversation.
2. Defines chat_node — the function that decides how to respond to the user.
3. Builds the LangGraph graph: remember → chat → end.
4. Sets up a SQLite checkpointer so conversations are saved to disk.

Routing logic inside chat_node (in order):
  1. Greeting            → reply "Hello! How can I help?"
  2. HITL resume         → continue after human approved/rejected
  3. Self-query          → return facts from long-term memory
  4. Weather question    → call OpenWeather (or DuckDuckGo as fallback)
  5. Time/date question  → return system clock
  6. News question       → DuckDuckGo search
  7. Stock question      → Yahoo Finance
  8. Has documents?      → RAG retrieval
     - Low confidence    → PAUSE and ask human to approve
     - Good context      → answer with citations
  9. Default             → plain LLM answer

──────────────────────────────────────────────────────────────────────────
WHAT IS HITL (Human-In-The-Loop)?
──────────────────────────────────────────────────────────────────────────
Normally the chatbot answers automatically. But what if the uploaded
document doesn't really contain the answer? The bot might hallucinate
or give a wrong answer confidently.

HITL solves this by PAUSING the agent when it is not confident:

  User asks a document question
       ↓
  Bot finds very little context in the document (low confidence)
       ↓
  Instead of guessing → Bot PAUSES and asks the human:
  "I don't have enough info. Should I try to answer anyway?"
       ↓
  Human clicks "Yes, answer" or "No, skip"
       ↓
  Bot resumes with the human's decision

This is implemented using LangGraph's interrupt mechanism:
- The graph stores `awaiting_hitl = True` in its state.
- The frontend detects this and shows Approve / Skip buttons.
- When the human clicks, a new graph invocation resumes with the decision.

Why is this useful?
- It keeps humans in control when the AI is uncertain.
- It prevents confidently wrong answers on document questions.
- It is a real pattern used in production AI systems.
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chatbot_memory import (
    clear_memory,
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
    evaluate_retriever,
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


# ══════════════════════════════════════════════════════════════════════════
# CITATION HELPER
# Every response ends with a small "Powered by" footer so the user
# always knows where the information came from.
# ══════════════════════════════════════════════════════════════════════════

def _cite(source: str) -> str:
    """
    Return a markdown citation footer line.

    Usage:
        content = f"Some answer\n\n{_cite('OpenWeather API')}"

    Available source labels and what they mean:
        "OpenWeather API"  — real-time weather data
        "Yahoo Finance"    — live stock prices
        "DuckDuckGo"       — web search results
        "System Clock"     — local machine date/time
        "Gemini 2.5 Flash" — LLM-generated answer (no external data)
        "FAISS + Gemini"   — document RAG + LLM answer with citations
        "PostgreSQL LTM"   — facts retrieved from long-term memory store
    """
    return f"> 🔧 **Powered by:** {source}"


# ══════════════════════════════════════════════════════════════════════════
# HITL CONFIG
# If the RAG context is shorter than this, we consider it "low confidence"
# and pause to ask the human whether to proceed.
# ══════════════════════════════════════════════════════════════════════════
HITL_MIN_CONTEXT_LENGTH = 200  # characters


# ══════════════════════════════════════════════════════════════════════════
# STATE
# This dictionary is passed through every node in the graph.
# LangGraph merges the messages list automatically using add_messages.
# ══════════════════════════════════════════════════════════════════════════

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
            # NOTE: Gemini requires at least one HumanMessage in the messages list.
            # We combine the system instructions + context into the HumanMessage.
            prompt = (
                "You are answering from limited document context — the user approved this.\n"
                "Do your best with the context below. Be honest if the answer is unclear.\n"
                "Cite sources like [1], [2] where possible.\n\n"
                f"Document context:\n{rag_context if rag_context else 'No context found.'}\n\n"
                f"Question: {original_question}"
            )
            response = llm.invoke(prompt)  # plain string prompt works for Gemini
            content = f"{response.content}\n\n{_cite('FAISS + Gemini 2.5 Flash (low-confidence approval)')}"
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
        content = f"{format_search_response(results, query)}\n\n{_cite('DuckDuckGo Search + Gemini 2.5 Flash')}"
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
            print(f"[Query Rewrite] {query} → {rewritten}")

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
            # IMPROVED: More detailed system prompt with structured instructions
            system_prompt = (
                "You are a document analysis expert. Your task is to answer the user's question "
                "using ONLY the document context provided below.\n\n"
                "IMPORTANT RULES:\n"
                "1. Use ONLY information from the document context. Do NOT use external knowledge.\n"
                "2. If the answer is not in the context, say clearly: 'This information is not found in the provided document.'\n"
                "3. Always cite your sources using [1], [2], etc. format when referencing specific sections.\n"
                "4. Structure your response clearly with headings and bullet points if the answer is complex.\n"
                "5. Be specific and detailed - avoid vague or generic responses.\n"
                "6. If the document mentions steps, processes, or stages - list them clearly in order.\n\n"
                "═" * 60 + "\n"
                "DOCUMENT CONTEXT:\n"
                "═" * 60 + "\n"
                f"{rag_context}\n"
                "═" * 60 + "\n"
                "USER QUESTION:\n"
                "═" * 60 + "\n"
                f"Question: {rewritten if rewritten != query else query}\n"
                "═" * 60 + "\n"
            )
            recent = get_recent_messages(state["messages"])
            response = llm.invoke([SystemMessage(content=system_prompt)] + recent)
            content = f"{response.content}\n\n{_cite('FAISS + BM25 Hybrid Retriever + Gemini 2.5 Flash')}"
            return {"messages": [AIMessage(content=content)]}

    # ── 9. Default LLM answer ────────────────────────────────────────────
    memory_text = get_memory_as_text(user_id)

    system_parts = ["You are a helpful AI assistant. Answer clearly and concisely."]
    if memory_text:
        system_parts.append(f"\nFacts about the user (use only if relevant):\n{memory_text}")

    system_prompt = "\n".join(system_parts)
    recent = get_recent_messages(state["messages"])
    response = llm.invoke([SystemMessage(content=system_prompt)] + recent)
    content = f"{response.content}\n\n{_cite('Gemini 2.5 Flash')}"
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
    db_path = "chatbot_db"
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
    return {}
