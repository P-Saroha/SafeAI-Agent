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
    Return True if the user is asking about an uploaded document.
    We check for keywords that suggest the user means "from the file I uploaded".
    """
    q = query.lower()
    doc_keywords = [
        "pdf", "document", "doc", "file", "upload",
        "in this", "from this", "according to", "summarize",
        "what does it say", "the report", "the paper",
    ]
    return any(kw in q for kw in doc_keywords)


# ══════════════════════════════════════════════════════════════════════════
# CHAT NODE
# The main function that handles every user message.
# ══════════════════════════════════════════════════════════════════════════

def chat_node(state: ChatState) -> dict:
    """
    Decide how to respond to the user's latest message.

    The function follows a simple top-to-bottom routing order.
    The first matching condition wins and returns immediately.
    """
    query = get_latest_user_message(state["messages"])
    thread_id = _safe_id(state.get("thread_id", "default"))
    user_id = _safe_id(state.get("user_id") or state.get("thread_id", "default"))

    # ── 1. Greeting ─────────────────────────────────────────────────────
    if is_greeting(query):
        return {"messages": [AIMessage(content="Hello! How can I help you today?")]}

    # ── 2. HITL resume: human has made a decision ────────────────────────
    # When awaiting_hitl is True and hitl_decision is set, the human has
    # responded to our pause. We now act on their choice.
    if state.get("awaiting_hitl") and state.get("hitl_decision"):
        decision = str(state["hitl_decision"]).lower().strip()
        original_question = str(state.get("hitl_question", query))

        if decision == "approve":
            # Human said "yes, try to answer even with weak context"
            rag_context = get_rag_context(original_question, thread_id)
            prompt = (
                "The user approved answering from limited document context. "
                "Do your best using the context below. "
                "Be honest if the answer is not clearly in the document. "
                "Cite sources like [1], [2] where possible.\n\n"
                f"Context:\n{rag_context}\n\n"
                f"Question: {original_question}"
            )
            response = llm.invoke([SystemMessage(content=prompt)])
            return {
                "messages": [response],
                "awaiting_hitl": False,
                "hitl_question": "",
                "hitl_decision": "",
            }
        else:
            # Human said "no, skip" — give a polite decline
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

    # ── 3. Self-query: user asking about their stored info ───────────────
    if is_self_query(query):
        facts = get_memory_as_text(user_id)
        if facts:
            response = f"Here is what I remember about you:\n{facts}"
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
        return {"messages": [AIMessage(content=format_weather_response(raw, location))]}

    # ── 5. Date / Time ───────────────────────────────────────────────────
    if is_time_query(query):
        now = call_datetime()
        return {"messages": [AIMessage(content=f"Current date and time:\n- {now}\n\nSources:\n- System clock")]}

    # ── 6. News ──────────────────────────────────────────────────────────
    if is_news_query(query):
        raw = call_search(query)
        return {"messages": [AIMessage(content=format_search_response(raw))]}

    # ── 7. Stock price ───────────────────────────────────────────────────
    if is_stock_query(query):
        symbol = extract_stock_symbol(query)
        if symbol:
            result = call_stock(symbol)
            return {"messages": [AIMessage(content=f"Stock price:\n- {result}\n\nSources:\n- Yahoo Finance")]}
        return {"messages": [AIMessage(content="Please include a company name or ticker, e.g. 'stock price of ORCL'.")]}

    # ── 8. RAG: user has uploaded documents ─────────────────────────────
    if has_documents(thread_id):
        rag_context = get_rag_context(query, thread_id)

        # ── HITL: low-confidence check ───────────────────────────────────
        # If the retrieved context is very short, the document probably
        # doesn't contain a good answer. Instead of guessing, we PAUSE
        # and ask the human whether they want us to try anyway.
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

        # Good context — answer with citations
        if rag_context:
            system_prompt = (
                "Answer the question using ONLY the document context below. "
                "Cite sources like [1], [2]. "
                "If the answer is not in the context, say so clearly.\n\n"
                f"Document context:\n{rag_context}"
            )
            recent = get_recent_messages(state["messages"])
            response = llm.invoke([SystemMessage(content=system_prompt)] + recent)
            return {"messages": [response]}

    # ── 9. Default LLM answer (with memory context if available) ─────────
    memory_text = get_memory_as_text(user_id)

    system_parts = ["You are a helpful AI assistant. Answer clearly and concisely."]
    if memory_text:
        system_parts.append(f"\nFacts about the user (use only if relevant):\n{memory_text}")

    system_prompt = "\n".join(system_parts)
    recent = get_recent_messages(state["messages"])
    response = llm.invoke([SystemMessage(content=system_prompt)] + recent)
    return {"messages": [response]}


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
