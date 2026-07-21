# AI Agent Chatbot

**Stack:** Python · LangGraph · LangChain · Gemini 2.5 Flash · FAISS · PostgreSQL · Streamlit

A beginner-friendly AI agent that combines deterministic tool routing, document-based RAG, short/long-term memory, and Human-In-The-Loop (HITL) approval — all wired together with LangGraph.

---

## What it can do

| Feature | Description |
|---|---|
| **Weather** | Real-time conditions via OpenWeather API, falls back to web search |
| **News** | Latest headlines via DuckDuckGo |
| **Stock price** | Live prices via Yahoo Finance |
| **Date / Time** | Current system time |
| **Document Q&A (RAG)** | Ask questions about your uploaded PDFs, TXT, or MD files |
| **Long-term memory** | Remembers your name, skills, goals across sessions (Postgres) |
| **Short-term memory** | Keeps the last 12 messages as conversation context |
| **HITL approval** | Pauses and asks you before answering with low-confidence document context |
| **Multi-thread chats** | Each conversation is isolated with its own documents and history |

---

## How the agent works

Every user message flows through this graph:

```
START → remember_node → chat_node → END
```

- **remember_node** — runs first on every message. Asks the LLM to extract any new facts (name, skills, goals, etc.) and saves them to Postgres.
- **chat_node** — decides how to respond, following this routing order:

```
1. Greeting?           → "Hello! How can I help?"
2. HITL resume?        → act on human's approve/skip decision
3. Self-query?         → return stored memory facts
4. Weather question?   → call OpenWeather API
5. Time question?      → return system clock
6. News question?      → DuckDuckGo search
7. Stock question?     → Yahoo Finance
8. Has documents?
   └─ Low confidence?  → PAUSE (HITL) and ask human to approve
   └─ Good context?    → LLM answer with [1][2] citations
9. Default             → plain LLM answer
```

---

## What is HITL (Human-In-The-Loop)?

Normally the bot answers automatically. But what if your uploaded document doesn't actually contain the answer? The bot could hallucinate a wrong answer confidently.

**HITL prevents this:**

```
User asks a document question
        ↓
Bot searches the document → finds very little context (< 200 chars)
        ↓
Instead of guessing → Bot PAUSES and asks:
"I found very little in your document. Should I try anyway?"
        ↓
Human clicks "Yes, try to answer" or "No, skip"
        ↓
Bot resumes with the human's decision
```

**How it works technically:**
- `chat_node` sets `awaiting_hitl = True` in the LangGraph state.
- The SqliteSaver checkpointer saves this state to disk.
- The Streamlit frontend reads the state, hides the chat input, and shows approval buttons.
- When the human clicks, the graph resumes with `hitl_decision = "approve"` or `"skip"`.

---

## Memory explained

### Short-Term Memory (STM)
The last 12 messages in the current conversation. Passed directly to the LLM so it remembers what was said earlier in the same chat. Gone when the session ends.

### Long-Term Memory (LTM)
User facts (name, education, interests, goals, skills, etc.) stored in Postgres. The `remember_node` runs on every message — it asks the LLM to extract any facts from the message and saves them. Persists across app restarts and different chat sessions.

### SqliteSaver (conversation checkpointing)
Every conversation's full state (messages, HITL flags, thread ID) is saved to a local SQLite file. This is what makes HITL possible — the `awaiting_hitl` flag survives page reloads.

---

## RAG (Retrieval-Augmented Generation) explained

Instead of the LLM making up an answer, the bot first searches your uploaded document for relevant text, then passes that text to the LLM as context.

- Each chat thread has its own `knowledge_base/<thread_id>/` folder.
- Documents are chunked (1000 chars, 150 overlap) and indexed into a FAISS vector store.
- Top-4 most relevant chunks are retrieved and formatted as `[1] filename.pdf (page 1): ...`
- The LLM is instructed to cite `[1]`, `[2]`, etc. in its answer.
- Embedding backend: Google `text-embedding-004` (or local hash embeddings as fallback — no API key needed).

---

## Project structure

```
Chatbot/
├── chatbotBackend.py     # Agent graph, chat_node, HITL logic, thread utilities
├── chatbotFrontend.py    # Streamlit UI — chat interface, sidebar, HITL buttons
├── chatbot_memory.py     # STM + LTM memory — remember_node, Postgres store
├── chatbot_rag.py        # FAISS index building, document loading, RAG retrieval
├── chatbot_tools.py      # Tool functions (weather, search, stock, time) + intent detectors
├── docker-compose.yml    # Postgres container for long-term memory
├── knowledge_base/       # Uploaded documents, one subfolder per thread
└── faiss_index/          # FAISS indexes, one subfolder per thread
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

# Optional — only needed for long-term memory
LTM_POSTGRES_URI=postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable

# Optional — use "google" for better RAG quality, "hash" works offline (default)
RAG_EMBEDDING_BACKEND=hash
```

### 3. Start Postgres (for long-term memory)

Long-term memory requires Docker. If you skip this step, the app still works — LTM is just disabled.

```bash
docker compose up -d
```

### 4. Run the app

```bash
streamlit run chatbotFrontend.py
```

---

## Example queries

```
"hello"                          → greeting
"weather in Mumbai"              → OpenWeather tool
"what time is it"                → system clock
"latest AI news"                 → DuckDuckGo search
"stock price of Apple"           → Yahoo Finance
"what do you know about me"      → reads your stored LTM facts
"my name is Sara, I like Python" → saves to LTM automatically
"summarize the PDF I uploaded"   → RAG over your document
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | Gemini 2.5 Flash (Google) |
| UI | Streamlit |
| Vector search | FAISS |
| Embeddings | Google text-embedding-004 / Hash fallback |
| Long-term memory | PostgreSQL via `langgraph.store.postgres` |
| Short-term memory | Last-N messages (in-context) |
| Conversation state | SqliteSaver (LangGraph) |
| Web search | DuckDuckGo (`ddgs`) |
| Stock data | Yahoo Finance (`yfinance`) |
| Weather | OpenWeather API |

---

## Skills demonstrated

- **LangGraph agent design** — multi-node graph with stateful checkpointing
- **Deterministic routing** — keyword-based intent detection before any LLM call
- **RAG pipeline** — per-thread FAISS indexes, chunking, citation-aware retrieval
- **Memory architecture** — STM vs LTM design, auto-extraction via LLM
- **HITL pattern** — graph interruption, state persistence, human approval flow
- **Error handling** — API fallbacks (OpenWeather → DuckDuckGo), graceful degradation
- **Streamlit UI** — streaming responses, file upload, multi-thread management
