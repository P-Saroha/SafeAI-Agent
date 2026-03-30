import streamlit as st
from chatbotBackend import (
    chatbot,
    unique_thread_pointer,
    generate_chat_title,
    rebuild_rag_index,
    delete_thread_history,
    get_thread_rag_docs_dir,
    get_user_memory,
    get_memory_status,
    get_user_memory_count,
    cleanup_user_memory,
    clear_user_memory,
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os
from pathlib import Path

os.environ['LANGSMITH_PROJECT'] = 'ChatBot-Project'


# ==============================
# UTIL FUNCTIONS
# ==============================
def generate_thread_id():
    return str(uuid.uuid4())


def deduplicate_message_history(messages):
    """Remove exact duplicate messages from history."""
    seen = set()
    deduped = []
    for msg in messages:
        key = f"{msg['role']}:{msg['content'][:200]}"
        if key not in seen:
            seen.add(key)
            deduped.append(msg)
    return deduped


def get_or_create_user_id():
    if 'user_id' in st.session_state:
        return st.session_state['user_id']

    user_id_path = Path(__file__).resolve().parent / ".user_id"
    if user_id_path.exists():
        stored = user_id_path.read_text(encoding="utf-8").strip()
        if stored:
            st.session_state['user_id'] = stored
            return stored

    user_id = str(uuid.uuid4())
    user_id_path.write_text(user_id, encoding="utf-8")
    st.session_state['user_id'] = user_id
    return user_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def delete_thread(thread_id):
    delete_status = delete_thread_history(thread_id)

    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
        if st.session_state['thread_id'] == thread_id:
            reset_chat()

    if thread_id in st.session_state.get('thread_titles', {}):
        del st.session_state['thread_titles'][thread_id]

    if st.session_state.get('thread_id') == thread_id:
        st.session_state['pending_approval'] = None

    return delete_status


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id, 'user_id': get_or_create_user_id()}}
    )
    messages = state.values.get('messages', [])
    
    # Deduplicate messages by content to prevent duplicates from appearing
    seen = set()
    deduped = []
    for msg in messages:
        # Skip ToolMessage and FunctionMessage - only show user and AI messages
        if hasattr(msg, '__class__'):
            msg_type = msg.__class__.__name__
            if msg_type in {'ToolMessage', 'FunctionMessage'}:
                continue
        
        # Create unique key for deduplication
        msg_content = str(getattr(msg, 'content', '')).strip()
        msg_type = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        key = f"{msg_type}:{msg_content[:200]}"  # Use first 200 chars as key
        
        if key not in seen:
            seen.add(key)
            deduped.append(msg)
    
    return deduped


def refresh_pending_approval(thread_id):
    """Load HITL pending request from graph state for the active thread."""
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id, 'user_id': get_or_create_user_id()}}
    )
    values = state.values or {}

    if values.get('awaiting_approval'):
        st.session_state['pending_approval'] = {
            'request': values.get('approval_request', 'Approval required.'),
            'type': values.get('approval_type', ''),
        }
    else:
        st.session_state['pending_approval'] = None


def save_uploaded_docs(uploaded_files, thread_id):
    """Save uploaded docs to knowledge folder and rebuild index."""
    if not uploaded_files:
        return 0, ""

    docs_dir = get_thread_rag_docs_dir(thread_id)
    Path(docs_dir).mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for uploaded in uploaded_files:
        file_path = Path(docs_dir) / uploaded.name
        with open(file_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        saved_count += 1

    status = rebuild_rag_index(thread_id)
    return saved_count, status


def extract_tool_names(message_chunk):
    """Extract tool names from model/tool message chunks."""
    tool_names = set()

    # AI tool calls (common for function-calling models)
    tool_calls = getattr(message_chunk, 'tool_calls', None)
    if tool_calls:
        for call in tool_calls:
            name = call.get('name') if isinstance(call, dict) else None
            if name:
                tool_names.add(name)

    # Tool message may carry tool name directly
    name = getattr(message_chunk, 'name', None)
    if isinstance(name, str) and name:
        tool_names.add(name)

    return tool_names


# ==============================
# SESSION INIT
# ==============================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

get_or_create_user_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = unique_thread_pointer()

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

if 'pending_approval' not in st.session_state:
    st.session_state['pending_approval'] = None

add_thread(st.session_state['thread_id'])
refresh_pending_approval(st.session_state['thread_id'])


# ==============================
# SIDEBAR
# ==============================
st.sidebar.title('LangGraph AI Agent')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('RAG Controls')
current_docs_dir = get_thread_rag_docs_dir(st.session_state['thread_id'])
knowledge_files = [
    p for p in Path(current_docs_dir).rglob('*')
    if p.is_file() and p.suffix.lower() in {'.pdf', '.txt', '.md'}
]
auto_mode_label = 'Smart Auto (Documents + Tools)' if knowledge_files else 'Agent Only (No RAG files found)'
st.sidebar.caption(f'Auto Mode: {auto_mode_label}')
st.sidebar.caption(f'Current chat documents: {len(knowledge_files)}')

if st.sidebar.button('Rebuild RAG Index'):
    with st.sidebar:
        with st.spinner('Rebuilding RAG index...'):
            status = rebuild_rag_index(st.session_state['thread_id'])
    st.sidebar.success(status)

st.sidebar.caption("Upload PDF, TXT, or MD files here to add chatbot knowledge.")
st.sidebar.caption('Attach files directly from chat input (paperclip) for GPT-like flow.')

st.sidebar.header('My Conversations')

st.sidebar.header('Memory (Debug)')
st.sidebar.caption(f"User ID: {get_or_create_user_id()}")
status = get_memory_status()
if status.get("available"):
    st.sidebar.success("LTM connected")
    st.sidebar.caption(f"Entries: {get_user_memory_count(get_or_create_user_id())}")
else:
    st.sidebar.warning("LTM not connected")
    if status.get("last_error"):
        st.sidebar.caption(status["last_error"])
if st.sidebar.button('Show Memory'):
    memories = get_user_memory(get_or_create_user_id())
    if memories:
        st.sidebar.write("\n".join(f"- {m}" for m in memories))
    else:
        st.sidebar.info('No LTM entries found for this user.')

if st.sidebar.button('Clean Memory Duplicates'):
    removed = cleanup_user_memory(get_or_create_user_id())
    st.sidebar.success(f"Removed {removed} duplicate entries.")

if st.sidebar.button('Clear All LTM (This User)'):
    removed = clear_user_memory(get_or_create_user_id())
    st.sidebar.success(f"Cleared {removed} memory entries.")

for thread_id in st.session_state['chat_threads'][::-1]:
    col1, col2 = st.sidebar.columns([4,1])

    # Get or generate title
    if thread_id not in st.session_state['thread_titles']:
        messages = load_conversation(thread_id)
        first_user_msg = None
        
        # Find first human message in conversation
        for msg in messages:
            if isinstance(msg, HumanMessage):
                first_user_msg = msg.content
                break
        
        if first_user_msg:
            generated_title = generate_chat_title(first_user_msg)
            st.session_state['thread_titles'][thread_id] = generated_title
        else:
            st.session_state['thread_titles'][thread_id] = "New Chat"
    
    title = st.session_state['thread_titles'][thread_id]

    with col1:
        if st.button(title, key=f"load_{thread_id}"):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages
            refresh_pending_approval(thread_id)

    with col2:
        if st.button("🗑", key=f"delete_{thread_id}"):
            status = delete_thread(thread_id)
            if status.lower().startswith('delete failed'):
                st.sidebar.warning(status)
            else:
                st.sidebar.success(status)


# ==============================
# MAIN CHAT UI
# ==============================
st.title(" AI Agent Chatbot")
st.caption('Attach PDF/TXT/MD with the chat input paperclip. Uploaded docs are indexed automatically.')

# Deduplicate and display messages
deduped_messages = deduplicate_message_history(st.session_state['message_history'])
for message in deduped_messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


pending = st.session_state.get('pending_approval')
if pending:
    st.warning(f"HITL: {pending['request']}")
    col_a, col_b = st.columns(2)

    if col_a.button('Approve', key=f"approve_{st.session_state['thread_id']}"):
        CONFIG = {
            'configurable': {
                'thread_id': st.session_state['thread_id'],
                'user_id': get_or_create_user_id(),
            }
        }
        with st.chat_message('user'):
            st.markdown('[HITL] Approve')

        with st.chat_message('assistant'):
            with st.spinner('Resuming after approval...'):
                ai_response = st.write_stream(
                    (
                        chunk.content
                        for chunk, _ in chatbot.stream(
                            {
                                'messages': [],
                                'mode': 'auto',
                                'thread_id': st.session_state['thread_id'],
                                'user_id': get_or_create_user_id(),
                                'approval_decision': 'approve',
                            },
                            config={**CONFIG, "recursion_limit": 50},
                            stream_mode='messages',
                        )
                        if isinstance(chunk, AIMessage) and chunk.content
                    )
                )

        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_response})
        refresh_pending_approval(st.session_state['thread_id'])
        st.rerun()

    if col_b.button('Regenerate', key=f"regen_{st.session_state['thread_id']}"):
        CONFIG = {
            'configurable': {
                'thread_id': st.session_state['thread_id'],
                'user_id': get_or_create_user_id(),
            }
        }
        with st.chat_message('user'):
            st.markdown('[HITL] Regenerate')

        with st.chat_message('assistant'):
            with st.spinner('Regenerating...'):
                ai_response = st.write_stream(
                    (
                        chunk.content
                        for chunk, _ in chatbot.stream(
                            {
                                'messages': [],
                                'mode': 'auto',
                                'thread_id': st.session_state['thread_id'],
                                'user_id': get_or_create_user_id(),
                                'approval_decision': 'regenerate',
                            },
                            config={**CONFIG, "recursion_limit": 50},
                            stream_mode='messages',
                        )
                        if isinstance(chunk, AIMessage) and chunk.content
                    )
                )

        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_response})
        refresh_pending_approval(st.session_state['thread_id'])
        st.rerun()


chat_payload = st.chat_input(
    "Type your message...",
    accept_file='multiple',
    file_type=['pdf', 'txt', 'md'],
    max_upload_size=200,
)

user_input = None
uploaded_files = []

if chat_payload is not None:
    if isinstance(chat_payload, str):
        user_input = chat_payload
    else:
        user_input = getattr(chat_payload, 'text', None) or getattr(chat_payload, 'message', None) or ""
        uploaded_files = list(getattr(chat_payload, 'files', []) or [])

if user_input or uploaded_files:
    if st.session_state.get('pending_approval'):
        st.warning('Resolve pending HITL request first using Approve or Regenerate.')
        st.stop()

    saved_count = 0
    upload_status = ""
    if uploaded_files:
        with st.spinner('Saving files and rebuilding RAG index...'):
            saved_count, upload_status = save_uploaded_docs(uploaded_files, st.session_state['thread_id'])

        if saved_count > 0:
            st.success(f'Uploaded {saved_count} file(s). {upload_status}')

    if not user_input:
        uploaded_names = ', '.join(f.name for f in uploaded_files) if uploaded_files else 'No files'
        confirmation = (
            f"Uploaded files: {uploaded_names}\n\n"
            f"{upload_status}\n\n"
            "You can now ask questions from these documents in this chat."
        )
        with st.chat_message('assistant'):
            st.markdown(confirmation)

        st.session_state['message_history'].append(
            {'role': 'assistant', 'content': confirmation}
        )
        refresh_pending_approval(st.session_state['thread_id'])
        st.stop()

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown(user_input)
        if uploaded_files:
            file_names = ', '.join(f.name for f in uploaded_files)
            st.caption(f'Attached files: {file_names}')

    CONFIG = {
        'configurable': {
            'thread_id': st.session_state['thread_id'],
            'user_id': get_or_create_user_id(),
        }
    }
    selected_mode = 'auto'

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            used_tools = set()
            tool_trace = st.empty()

            def stream_response():
                try:
                    for message_chunk, metadata in chatbot.stream(
                        {
                            "messages": [HumanMessage(content=user_input)],
                            "mode": selected_mode,
                            "thread_id": st.session_state['thread_id'],
                            "user_id": get_or_create_user_id(),
                        },
                        config={**CONFIG, "recursion_limit": 50},
                        stream_mode="messages"
                    ):
                        detected_tools = extract_tool_names(message_chunk)
                        if detected_tools:
                            used_tools.update(detected_tools)
                            tool_trace.info(f"Tools used: {', '.join(sorted(used_tools))}")

                        if metadata.get("langgraph_node") == "tools" and not used_tools:
                            tool_trace.info("Using tool...")

                        if isinstance(message_chunk, AIMessage) and message_chunk.content:
                            yield message_chunk.content

                except Exception as e:
                    error_msg = str(e).lower()
                    if "quota" in error_msg:
                        yield " API quota exceeded. Please wait."
                    elif "recursion limit" in error_msg:
                        yield " The response required too many tool calls. Please try a simpler query or rephrase your question."
                    else:
                        yield f" Error: {str(e)}"

            ai_response = st.write_stream(stream_response())

            if used_tools:
                tool_trace.success(f"Final tools used: {', '.join(sorted(used_tools))}")
            else:
                q = user_input.lower()
                if "weather" in q or "aqi" in q or "temperature" in q or "temp" in q:
                    tool_trace.success("Final tools used: get_weather")
                elif "stock" in q or "price" in q:
                    tool_trace.success("Final tools used: get_stock_price")
                elif "news" in q or "headline" in q or "trending" in q or "latest" in q:
                    tool_trace.success("Final tools used: search_tool")
                elif "time" in q or "date" in q or "today" in q:
                    tool_trace.success("Final tools used: get_current_date_time")
                else:
                    tool_trace.caption("No external tool was needed for this response.")

    refresh_pending_approval(st.session_state['thread_id'])

    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_response}
    )
    
    # Generate title for new conversation if not already set
    if st.session_state['thread_id'] not in st.session_state['thread_titles']:
        title = generate_chat_title(user_input)
        st.session_state['thread_titles'][st.session_state['thread_id']] = title
        st.rerun()