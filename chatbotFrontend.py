import streamlit as st
from chatbotBackend import chatbot, unique_thread_pointer, generate_chat_title, rebuild_rag_index, RAG_DOCS_DIR
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


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def delete_thread(thread_id):
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
        if st.session_state['thread_id'] == thread_id:
            reset_chat()


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# ==============================
# SESSION INIT
# ==============================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = unique_thread_pointer()

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

add_thread(st.session_state['thread_id'])


# ==============================
# SIDEBAR
# ==============================
st.sidebar.title('LangGraph AI Agent')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('RAG Controls')
if st.sidebar.button('Rebuild RAG Index'):
    with st.sidebar:
        with st.spinner('Rebuilding RAG index...'):
            status = rebuild_rag_index()
    st.sidebar.success(status)

st.sidebar.caption(f"RAG Document Folder: {RAG_DOCS_DIR}")
st.sidebar.caption("Upload PDF, TXT, or MD files here to add chatbot knowledge.")
uploaded_files = st.sidebar.file_uploader(
    "Upload RAG files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Supported: PDF, TXT, MD"
)

if st.sidebar.button('Upload Files and Rebuild Index'):
    if not uploaded_files:
        st.sidebar.warning('Please select at least one file to upload.')
    else:
        Path(RAG_DOCS_DIR).mkdir(parents=True, exist_ok=True)
        saved_count = 0

        for uploaded in uploaded_files:
            file_path = Path(RAG_DOCS_DIR) / uploaded.name
            with open(file_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            saved_count += 1

        with st.sidebar:
            with st.spinner('Rebuilding RAG index...'):
                status = rebuild_rag_index()

        st.sidebar.success(f'Saved {saved_count} file(s). {status}')

st.sidebar.header('My Conversations')

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

    with col2:
        if st.button("🗑", key=f"delete_{thread_id}"):
            delete_thread(thread_id)
            if thread_id in st.session_state['thread_titles']:
                del st.session_state['thread_titles'][thread_id]


# ==============================
# MAIN CHAT UI
# ==============================
st.title(" AI Agent Chatbot")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            def stream_response():
                try:
                    for message_chunk, metadata in chatbot.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages"
                    ):

                        # Show tool usage
                        if metadata.get("langgraph_node") == "tools":
                            st.info(" Using tool...")

                        if isinstance(message_chunk, AIMessage) and message_chunk.content:
                            yield message_chunk.content

                except Exception as e:
                    if "quota" in str(e).lower():
                        yield " API quota exceeded. Please wait."
                    else:
                        yield f" Error: {str(e)}"

            ai_response = st.write_stream(stream_response())

    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_response}
    )
    
    # Generate title for new conversation if not already set
    if st.session_state['thread_id'] not in st.session_state['thread_titles']:
        title = generate_chat_title(user_input)
        st.session_state['thread_titles'][st.session_state['thread_id']] = title
        st.rerun()