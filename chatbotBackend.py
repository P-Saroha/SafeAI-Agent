from __future__ import annotations

import os
import re
import sqlite3
import shutil
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chatbot_memory import (
    LTM_STM_MAX_MESSAGES,
    _extract_name_from_memory,
    _extract_stm_facts,
    _filter_messages_for_llm,
    _format_memory_item,
    _get_user_memory_context,
    _get_user_memory_items,
    _is_self_query,
    _memory_display_key,
    _memory_namespace,
    _open_memory_store,
    _safe_thread_id,
    _ensure_memory_store,
    cleanup_user_memory,
    clear_user_memory,
    get_memory_status,
    get_user_memory,
    get_user_memory_count,
    remember_node,
)
from chatbot_rag import (
    active_thread_id,
    get_thread_rag_docs_dir,
    get_thread_rag_index_dir,
    rebuild_rag_index,
    rag_retriever_cache,
    rag_status_cache,
    _collect_rag_files,
    _ensure_rag_citations,
    _rag_context_for_query,
)
from chatbot_tools import (
    llm,
    _call_search_tool,
    _call_stock_tool,
    _call_time_tool,
    _call_weather_tool,
    _collapse_duplicate_phrase,
    _extract_first_url,
    _extract_stock_symbol,
    _extract_urls_from_text,
    _extract_weather_location,
    _format_search_results,
    _format_sources,
    _is_greeting,
    _is_news_query,
    _is_simple_question,
    _is_stock_query,
    _is_time_query,
    _is_weather_query,
    _llm_needs_tool,
    _needs_external_tools,
    _parse_weather_payload,
    _route_tool_with_llm,
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
    tools_called_in_turn: int


# HITL settings: keep only low-confidence RAG approval.
LOW_CONFIDENCE_RAG_MIN_CHARS = 220


def _is_document_intent(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False

    doc_markers = [
        "pdf",
        "document",
        "doc",
        "file",
        "upload",
        "resume",
        "letter",
        "in this",
        "from this",
        "according to",
        "provided",
        "page",
    ]
    return any(marker in q for marker in doc_markers)


def _approval_reset_state() -> dict:
    return {
        "awaiting_approval": False,
        "approval_request": "",
        "approval_type": "",
        "approval_tool_calls": [],
        "approval_decision": "",
    }


def _resolve_hitl_decision(state: ChatState, latest_query: str, rag_context: str) -> dict | None:
    decision = str(state.get("approval_decision", "")).lower().strip()
    if not decision:
        return None

    approval_type = str(state.get("approval_type", ""))
    if approval_type == "low_confidence_rag":
        if decision == "approve":
            forced_prompt = [
                SystemMessage(
                    content=(
                        "Human approved answering from weak document context. "
                        "Answer only from the provided context and cite sources like [1], [2]."
                    )
                ),
                HumanMessage(content=f"Question: {latest_query}\n\nContext:\n{rag_context}"),
            ]
            forced_answer = llm.invoke(forced_prompt)
            content = _ensure_rag_citations(getattr(forced_answer, "content", ""), rag_context)
            if content != getattr(forced_answer, "content", ""):
                forced_answer = AIMessage(content=content)
            return {**_approval_reset_state(), "messages": [forced_answer]}

        clarify = AIMessage(
            content=(
                "I need a little more direction before answering. "
                "Please refine your question or upload a clearer/more relevant document."
            )
        )
        return {**_approval_reset_state(), "messages": [clarify]}

    return _approval_reset_state()


# ==============================
# CHAT NODE
# ==============================
def chat_node(state: ChatState):
    latest_query = _latest_user_query(state["messages"])
    thread_id = _safe_thread_id(state.get("thread_id", "default"))
    user_id = _safe_thread_id(state.get("user_id") or state.get("thread_id", "default"))
    memory_context = "" if _is_greeting(latest_query) else _get_user_memory_context(user_id, latest_query)
    last_weather_link = str(state.get("last_weather_link") or "")

    if _is_greeting(latest_query):
        return {"messages": [AIMessage(content="Hello! How can I help you today?")], "allow_tools": False}

    if latest_query and _link_request(latest_query):
        if last_weather_link:
            return {
                "messages": [AIMessage(content=last_weather_link)],
                "allow_tools": False,
            }

    if state.get("pending_weather") and latest_query:
        raw_weather = _call_weather_tool(latest_query)
        payload = _parse_weather_payload(raw_weather)
        if payload:
            response_text = (
                "Weather:\n"
                f"- Location: {payload.get('location', latest_query).strip()}\n"
                f"- Condition: {payload.get('description', '').strip() or 'Unavailable'}\n"
                f"- Temperature: {payload.get('temp_c')} C\n"
                f"- Feels like: {payload.get('feels_like_c')} C\n"
                f"- Humidity: {payload.get('humidity')}%\n"
                f"- Wind: {payload.get('wind_mps')} m/s\n\n"
                f"{_format_sources(['https://openweathermap.org/'], fallback='OpenWeather')}"
            )
            return {
                "messages": [AIMessage(content=response_text)],
                "allow_tools": False,
                "pending_weather": False,
                "last_weather_link": "https://openweathermap.org/",
            }

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

        return {
            "messages": [
                AIMessage(
                    content="Weather:\n- Summary: Could not fetch weather data.\n\n"
                    + _format_sources([], fallback="Web search")
                )
            ],
            "allow_tools": False,
            "pending_weather": False,
        }

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

    decision_result = _resolve_hitl_decision(state, latest_query, rag_context)
    if decision_result is not None:
        return decision_result

    route_hint = "none"
    if latest_query:
        has_rule_match = any(
            [
                _is_weather_query(latest_query),
                _is_time_query(latest_query),
                _is_news_query(latest_query),
                _is_stock_query(latest_query),
            ]
        )
        if not has_rule_match and not _is_simple_question(latest_query):
            if _llm_needs_tool(latest_query):
                route_hint = _route_tool_with_llm(latest_query)

    if _is_self_query(latest_query):
        memory_items = _get_user_memory_items(user_id, latest_query)
        stm_items = _extract_stm_facts(state.get("messages", []), exclude_latest=True)
        combined_items = memory_items + stm_items
        if combined_items:
            greeting = "Sure, "
            seen = set()
            lines = []
            for item in combined_items:
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

    if _is_weather_query(latest_query) or route_hint == "weather":
        location = _extract_weather_location(latest_query)
        if not location:
            return {
                "messages": [AIMessage(content="Which city are you asking about?")],
                "allow_tools": False,
                "pending_weather": True,
            }
        location = location or latest_query
        raw_weather = _call_weather_tool(location)
        payload = _parse_weather_payload(raw_weather)
        if payload:
            response_text = (
                "Weather:\n"
                f"- Location: {payload.get('location', location).strip()}\n"
                f"- Condition: {payload.get('description', '').strip() or 'Unavailable'}\n"
                f"- Temperature: {payload.get('temp_c')} C\n"
                f"- Feels like: {payload.get('feels_like_c')} C\n"
                f"- Humidity: {payload.get('humidity')}%\n"
                f"- Wind: {payload.get('wind_mps')} m/s\n\n"
                f"{_format_sources(['https://openweathermap.org/'], fallback='OpenWeather')}"
            )
            return {
                "messages": [AIMessage(content=response_text)],
                "allow_tools": False,
                "pending_weather": False,
                "last_weather_link": "https://openweathermap.org/",
            }

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

    if _is_time_query(latest_query) or route_hint == "time":
        now_text = _call_time_tool()
        response_text = f"Date/Time:\n- {now_text}\n\n" + _format_sources([], fallback="System clock")
        return {"messages": [AIMessage(content=response_text)], "allow_tools": False}

    if _is_news_query(latest_query) or route_hint == "news":
        raw_news = _call_search_tool(latest_query)
        response_text = "Top results:\n" + _format_search_results(raw_news)
        return {"messages": [AIMessage(content=response_text)], "allow_tools": False}

    if _is_stock_query(latest_query) or route_hint == "stock":
        symbol = _extract_stock_symbol(latest_query)
        if symbol:
            stock_text = _call_stock_tool(symbol)
            response_text = f"Stock:\n- {stock_text}\n\n" + _format_sources([], fallback="Yahoo Finance")
            return {
                "messages": [AIMessage(content=response_text)],
                "allow_tools": False,
            }
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Stock:\n- I need a ticker or company name I can match. "
                        "Try something like 'stock price of Oracle (ORCL)'.\n\n"
                        + _format_sources([], fallback="Market data")
                    )
                )
            ],
            "allow_tools": False,
        }

    if _needs_external_tools(latest_query) or route_hint == "search":
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


def _latest_user_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def _link_request(text: str) -> bool:
    return bool(re.search(r"\b(link|that link|the link|share link|give me that link)\b", text.lower()))


def route_after_chat(state: ChatState):
    if state.get("awaiting_approval"):
        return "wait_for_human"
    return "__end__"


# ==============================
# DATABASE (MEMORY)
# ==============================
def init_checkpointer():
    db_path = "chatbot_db"

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("SELECT 1")
        checkpointer = SqliteSaver(conn=conn)
        return checkpointer, conn
    except Exception as err:
        print(f"Checkpoint error detected: {err}. Recovering...")

        if os.path.exists(db_path):
            backup_path = f"{db_path}.backup_{os.getpid()}"
            try:
                os.rename(db_path, backup_path)
                print(f"Backed up corrupted database to {backup_path}")
            except Exception as backup_err:
                print(f"Backup failed: {backup_err}")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        print("Fresh checkpoint database created")
        return checkpointer, conn


checkpointer, conn = init_checkpointer()


def delete_thread_history(thread_id: str) -> str:
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
            columns = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")').fetchall()]
            if "thread_id" not in columns:
                continue

            cur.execute(f'DELETE FROM "{table}" WHERE thread_id = ?', (thread_key,))
            if cur.rowcount and cur.rowcount > 0:
                total_deleted += cur.rowcount

        conn.commit()

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
            except Exception:
                pass

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
    },
)

chatbot = builder.compile(checkpointer=checkpointer)


# ==============================
# CHAT TITLE GENERATION
# ==============================
def generate_chat_title(user_message: str) -> str:
    if not user_message or not user_message.strip():
        return "New Chat"

    msg = user_message.strip()[:200]

    try:
        prompt = (
            "Given this user message, generate a very short chat title (2-5 words max, no punctuation).\n"
            "Only return the title, nothing else.\n\n"
            f"User message: {msg}\n\n"
            "Title:"
        )
        title = llm.invoke(prompt).content.strip()
        title = title.strip('"\'.!?,;:').strip()
        if title:
            return title[:50]
    except Exception as err:
        print(f"Title generation error (using fallback): {str(err)[:50]}")

    words = msg.split()
    if words:
        fallback_title = " ".join(words[:5])
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
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
