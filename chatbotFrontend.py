"""
chatbotFrontend.py
------------------
The Streamlit UI for the chatbot.

What this file does:
1. Shows a chat interface where the user can type messages.
2. Displays the conversation history.
3. Lets the user manage multiple chat threads from the sidebar.
4. Lets the user upload documents (PDF, TXT, MD) for RAG.
5. Shows long-term memory info in the sidebar.

Run with:
    streamlit run chatbotFrontend.py
"""

#  SUPPRESS WARNINGS ONLY
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load Chatbot/.env first so all keys are available before any imports
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbotBackend import (
    chatbot,
    delete_thread,
    get_thread_hitl_state,
    list_all_threads,
)
from chatbot_memory import (
    clear_memory,
    generate_recap_greeting,
    get_memory_count,
    get_memory_list,
    get_memory_status,
)
from chatbot_rag import get_docs_dir, rebuild_rag_index

# LangSmith configuration (optional for production monitoring)
# Automatically enabled if LANGSMITH_API_KEY is set in .env
# Traces all LLM calls, tool execution, and graph state transitions


# ══════════════════════════════════════════════════════════════════════════
# SESSION HELPERS
# Small utilities for managing user/thread IDs across page reloads.
# ══════════════════════════════════════════════════════════════════════════

def _user_id_file() -> Path:
    """Path to a file that stores this user's stable ID."""
    return Path(__file__).resolve().parent / ".user_id"


def get_or_create_user_id() -> str:
    """
    Load the user ID from disk, or create a new one.
    This keeps the same user ID even after restarting the app.
    """
    if "user_id" in st.session_state:
        return st.session_state["user_id"]

    path = _user_id_file()
    if path.exists():
        uid = path.read_text(encoding="utf-8").strip()
        if uid:
            st.session_state["user_id"] = uid
            return uid

    uid = str(uuid.uuid4())
    path.write_text(uid, encoding="utf-8")
    st.session_state["user_id"] = uid
    return uid


def new_thread_id() -> str:
    """Generate a fresh unique thread ID."""
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════
# CONVERSATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_thread_messages(thread_id: str) -> list[dict]:
    """
    Load the saved messages for a thread from the LangGraph checkpointer.
    Returns a list of {"role": "user"/"assistant", "content": "..."} dicts
    that Streamlit's st.chat_message can display.
    
    CACHED for 5 minutes to avoid re-querying state on every page rerun.
    """
    user_id = get_or_create_user_id()
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id, "user_id": user_id}}
    )
    messages = state.values.get("messages", [])

    result = []
    seen = set()
    for msg in messages:
        # Skip non-human/AI messages
        if not isinstance(msg, (HumanMessage, AIMessage)):
            continue
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = str(msg.content or "").strip()
        # Deduplicate by content
        key = f"{role}:{content[:200]}"
        if key in seen or not content:
            continue
        seen.add(key)
        result.append({"role": role, "content": content})

    return result


def start_new_chat():
    """Reset session state and start a fresh chat thread."""
    tid = new_thread_id()
    st.session_state["thread_id"] = tid
    st.session_state["messages"] = []
    # Add to thread list if not already there
    if tid not in st.session_state["thread_ids"]:
        st.session_state["thread_ids"].append(tid)
    # Clear thread title so it gets generated after first message
    st.session_state["thread_titles"].pop(tid, None)


def save_uploaded_files(files, thread_id: str) -> tuple[int, str]:
    """
    Save uploaded files to the thread's knowledge_base folder,
    then rebuild the FAISS index.
    Returns (number_of_files_saved, status_message).
    """
    if not files:
        return 0, ""

    docs_dir = get_docs_dir(thread_id)
    docs_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = docs_dir / f.name
        dest.write_bytes(f.getbuffer())

    status = rebuild_rag_index(thread_id)
    return len(files), status


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# Runs once when the app first loads.
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_threads() -> list:
    """Load all existing threads from checkpointer, cached for 5 minutes."""
    return list_all_threads()


user_id = get_or_create_user_id()

if "thread_ids" not in st.session_state:
    # Load all existing threads from checkpointer on first run
    st.session_state["thread_ids"] = get_cached_threads()

if "thread_id" not in st.session_state:
    if st.session_state["thread_ids"]:
        st.session_state["thread_id"] = st.session_state["thread_ids"][-1]
    else:
        st.session_state["thread_id"] = new_thread_id()
        st.session_state["thread_ids"].append(st.session_state["thread_id"])

if "messages" not in st.session_state:
    st.session_state["messages"] = load_thread_messages(st.session_state["thread_id"])

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "hitl_pending" not in st.session_state:
    st.session_state["hitl_pending"] = False

# Check if an HITL approval was already pending when the page loads
# (e.g. user refreshed the browser mid-approval)
_hitl = get_thread_hitl_state(st.session_state["thread_id"], user_id)
if _hitl["awaiting"] and not st.session_state["hitl_pending"]:
    st.session_state["hitl_pending"] = True


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

st.sidebar.title("LangGraph AI Agent")

# ── New Chat button ─────────────────────────────────────────────────────
if st.sidebar.button("New Chat"):
    # Clear message cache for the old thread
    load_thread_messages.clear()
    start_new_chat()
    st.rerun()

# ── RAG section ─────────────────────────────────────────────────────────
st.sidebar.header("Documents (RAG)")
current_docs_dir = get_docs_dir(st.session_state["thread_id"])
doc_files = list(current_docs_dir.rglob("*")) if current_docs_dir.exists() else []
doc_count = sum(1 for p in doc_files if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"})
st.sidebar.caption(f"Uploaded documents in this chat: {doc_count}")

if st.sidebar.button("Rebuild RAG Index"):
    with st.spinner("Rebuilding..."):
        msg = rebuild_rag_index(st.session_state["thread_id"])
    st.sidebar.success(msg)

# ── Export Chat ──────────────────────────────────────────────────────────
# Converts the current conversation into a Markdown file the user can download.
# Each message is formatted as "**user:** ..." or "**assistant:** ..."
st.sidebar.header("Export")
if st.session_state.get("messages"):
    # Build the markdown content
    title = st.session_state["thread_titles"].get(
        st.session_state["thread_id"], "Chat Export"
    )
    lines = [f"# {title}\n"]
    for msg in st.session_state["messages"]:
        role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{role_label}:\n{msg['content']}\n")
    export_text = "\n---\n".join(lines)

    st.sidebar.download_button(
        label="Download chat as .md",
        data=export_text,
        file_name=f"{title.replace(' ', '_')}.md",
        mime="text/markdown",
    )
else:
    st.sidebar.caption("No messages to export yet.")

# ── Memory section ──────────────────────────────────────────────────────
st.sidebar.header("Long-Term Memory")
mem_status = get_memory_status()
if mem_status["available"]:
    st.sidebar.success("Connected to Postgres")
    st.sidebar.caption(f"Stored facts: {get_memory_count(user_id)}")
else:
    st.sidebar.warning("Postgres not connected (LTM disabled)")
    if mem_status.get("last_error"):
        st.sidebar.caption(f"Error: {mem_status['last_error']}")

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_memory(user_id: str):
    """Get memory list cached for 5 minutes."""
    return get_memory_list(user_id)

if st.sidebar.button("Show Memory"):
    facts = get_cached_memory(user_id)
    if facts:
        for fact in facts:
            st.sidebar.write(f"- {fact}")
    else:
        st.sidebar.info("No memory stored yet.")

if st.sidebar.button("Clear All Memory"):
    removed = clear_memory(user_id)
    st.sidebar.success(f"Cleared {removed} memory entries.")

# ── Conversation list ────────────────────────────────────────────────────
st.sidebar.header("My Conversations")

for tid in reversed(st.session_state["thread_ids"]):
    # Use cached title or extract first message as preview (NO LLM CALL!)
    if tid not in st.session_state["thread_titles"]:
        msgs = load_thread_messages(tid)
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        # Just use first 50 chars of first message as title - NO LLM CALL
        title = (first_user[:50] + "..." if len(first_user) > 50 else first_user) if first_user else "New Chat"
        st.session_state["thread_titles"][tid] = title

    title = st.session_state["thread_titles"][tid]
    col1, col2 = st.sidebar.columns([5, 1], gap="small")

    with col1:
        # Highlight the active thread
        label = f"**{title}**" if tid == st.session_state["thread_id"] else title
        if st.button(label, key=f"load_{tid}", use_container_width=True):
            st.session_state["thread_id"] = tid
            # Don't use cached messages when switching threads - bypass cache
            load_thread_messages.clear()
            st.session_state["messages"] = load_thread_messages(tid)
            st.rerun()

    with col2:
        if st.button("X", key=f"del_{tid}", use_container_width=True):
            result = delete_thread(tid)
            st.session_state["thread_ids"].remove(tid)
            st.session_state["thread_titles"].pop(tid, None)
            # Clear the cache so get_cached_threads() is called again
            get_cached_threads.clear()
            # If we deleted the active thread, start a new one
            if st.session_state["thread_id"] == tid:
                start_new_chat()
            st.sidebar.success(result)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# MAIN CHAT UI
# ══════════════════════════════════════════════════════════════════════════

st.title("AI Agent Chatbot")
st.caption("Upload PDF, TXT, or MD files using the paperclip icon in the chat input.")

# Display all messages in the current thread
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Memory recap greeting ────────────────────────────────────────────────
# When a chat is empty (new chat) and the user has stored LTM facts,
# show a personalized welcome-back message so they know the bot remembers them.
# We store it in session_state so it only generates once per new chat,
# not on every page re-render.
if not st.session_state["messages"]:
    if "recap_shown_for" not in st.session_state:
        st.session_state["recap_shown_for"] = None

    current_tid = st.session_state["thread_id"]

    # Only generate once per thread (avoid re-running on every Streamlit rerun)
    if st.session_state["recap_shown_for"] != current_tid:
        recap = generate_recap_greeting(user_id)
        st.session_state["recap_shown_for"] = current_tid
        st.session_state["recap_greeting"] = recap
    else:
        recap = st.session_state.get("recap_greeting", "")

    if recap:
        with st.chat_message("assistant"):
            st.markdown(recap)

# ── HITL Approval UI ────────────────────────────────────────────────────
# When the bot is not confident about a document answer, it pauses and
# stores awaiting_hitl=True in the graph state. We detect that here and
# show Approve / Skip buttons instead of the normal chat input.
if st.session_state.get("hitl_pending"):
    st.warning(
        "Approval needed: The bot found limited context in your document. "
        "Choose what to do:"
    )
    col_approve, col_skip = st.columns(2)

    if col_approve.button("Yes, try to answer", key="hitl_approve"):
        config = {
            "configurable": {
                "thread_id": st.session_state["thread_id"],
                "user_id": user_id,
            },
            "recursion_limit": 25,
        }
        with st.chat_message("assistant"):
            with st.spinner("Answering with available context..."):
                ai_response = ""
                try:
                    for state_snapshot in chatbot.stream(
                        {
                            "messages": [],
                            "thread_id": st.session_state["thread_id"],
                            "user_id": user_id,
                            "hitl_decision": "approve",
                            "awaiting_hitl": True,
                        },
                        config=config,
                        stream_mode="values",
                    ):
                        messages = state_snapshot.get("messages", [])
                        for msg in reversed(messages):
                            if (
                                isinstance(msg, AIMessage)
                                and isinstance(msg.content, str)
                                and msg.content.strip()
                            ):
                                ai_response = msg.content.strip()
                                break
                except Exception as e:
                    ai_response = f"Error: {e}"
                ai_response = ai_response or "No response generated."
                st.markdown(ai_response)

        st.session_state["messages"].append({"role": "assistant", "content": ai_response})
        st.session_state["hitl_pending"] = False
        st.rerun()

    if col_skip.button("No, skip", key="hitl_skip"):
        config = {
            "configurable": {
                "thread_id": st.session_state["thread_id"],
                "user_id": user_id,
            },
            "recursion_limit": 25,
        }
        with st.chat_message("assistant"):
            with st.spinner("Skipping..."):
                ai_response = ""
                try:
                    for state_snapshot in chatbot.stream(
                        {
                            "messages": [],
                            "thread_id": st.session_state["thread_id"],
                            "user_id": user_id,
                            "hitl_decision": "skip",
                            "awaiting_hitl": True,
                        },
                        config=config,
                        stream_mode="values",
                    ):
                        messages = state_snapshot.get("messages", [])
                        for msg in reversed(messages):
                            if (
                                isinstance(msg, AIMessage)
                                and isinstance(msg.content, str)
                                and msg.content.strip()
                            ):
                                ai_response = msg.content.strip()
                                break
                except Exception as e:
                    ai_response = f"Error: {e}"
                ai_response = ai_response or "No response generated."
                st.markdown(ai_response)

        st.session_state["messages"].append({"role": "assistant", "content": ai_response})
        st.session_state["hitl_pending"] = False
        st.rerun()

    st.stop()  # Don't show the chat input while HITL is pending

# ── Chat input (supports file attachments) ───────────────────────────────
chat_input = st.chat_input(
    "Type your message...",
    accept_file="multiple",
    file_type=["pdf", "txt", "md"],
)

# Parse the chat input (can be a string or an object with .text and .files)
user_text = ""
uploaded_files = []

if chat_input is not None:
    if isinstance(chat_input, str):
        user_text = chat_input
    else:
        user_text = getattr(chat_input, "text", "") or getattr(chat_input, "message", "") or ""
        uploaded_files = list(getattr(chat_input, "files", []) or [])

# ── Handle file uploads ──────────────────────────────────────────────────
if uploaded_files:
    with st.spinner("Saving files and building RAG index..."):
        count, status = save_uploaded_files(uploaded_files, st.session_state["thread_id"])
    st.success(f"Uploaded {count} file(s). {status}")

    # If no text was typed, just confirm the upload and stop here
    if not user_text:
        names = ", ".join(f.name for f in uploaded_files)
        confirmation = (
            f"Files uploaded: **{names}**\n\n"
            f"{status}\n\n"
            "You can now ask questions about these documents."
        )
        with st.chat_message("assistant"):
            st.markdown(confirmation)
        st.session_state["messages"].append({"role": "assistant", "content": confirmation})
        st.stop()

# ── Handle user message ──────────────────────────────────────────────────
if user_text:
    # Show the user message immediately
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Call the chatbot and stream the response
    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "user_id": user_id,
        },
        "recursion_limit": 25,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use stream_mode="values" to get the final state after all nodes run.
            # This avoids the {} from remember_node and double-response issues.
            ai_response = ""
            try:
                for state_snapshot in chatbot.stream(
                    {
                        "messages": [HumanMessage(content=user_text)],
                        "thread_id": st.session_state["thread_id"],
                        "user_id": user_id,
                    },
                    config=config,
                    stream_mode="values",
                ):
                    # "values" mode gives us the full state after each node.
                    # We want the last AIMessage from the final state.
                    messages = state_snapshot.get("messages", [])
                    for msg in reversed(messages):
                        if (
                            isinstance(msg, AIMessage)
                            and isinstance(msg.content, str)
                            and msg.content.strip()
                        ):
                            ai_response = msg.content.strip()
                            break
            except Exception as e:
                ai_response = f"Sorry, something went wrong: {e}"

            if not ai_response:
                ai_response = "I couldn't generate a response. Please try again."
            st.markdown(ai_response)

    # Save the assistant response to history
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})

    # Check if this response triggered an HITL pause
    hitl = get_thread_hitl_state(st.session_state["thread_id"], user_id)
    if hitl["awaiting"]:
        st.session_state["hitl_pending"] = True

    # Generate a title for this thread after the first message
    if st.session_state["thread_id"] not in st.session_state["thread_titles"]:
        # Use first 50 chars of user text as title - NO LLM CALL
        title = user_text[:50] + "..." if len(user_text) > 50 else user_text
        st.session_state["thread_titles"][st.session_state["thread_id"]] = title
        st.rerun()
