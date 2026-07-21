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
| **Chat export** | Download any conversation as a Markdown file |
| **Memory recap** | Personalized welcome-back greeting on every new chat using your stored facts |
| **Query Rewriting** | Detects vague queries ("what about that") and rewrites them to be specific ("What is the main topic?") using LLM |
| **RAG Metrics** | Tracks retrieval quality with Hit Rate@K and Mean Reciprocal Rank (MRR) for production monitoring |

---

## Agent Workflow

```mermaid
flowchart TD
    U[User Query] --> R[remember_node\nAuto-save facts to LTM]
    R --> C{chat_node\nRoute Intent}

    C -->|Greeting| G[Reply: Hello!]
    C -->|Self query| M[Read LTM from Postgres\nReturn stored facts]
    C -->|Weather / Time\nNews / Stock| T[Tool Call\nOpenWeather · DuckDuckGo\nYahoo Finance · Clock]
    C -->|Has documents| QR[Query Rewriting\nis_ambiguous_query?]

    QR -->|Ambiguous| RW[rewrite_query\nUse LLM to clarify]
    QR -->|Clear| D{RAG Retrieval\nFAISS top-4 chunks}
    RW --> D

    D -->|Low confidence| H[HITL Pause\nAsk human to Approve or Skip]
    H -->|Approved| L[LLM Answer with citations]
    H -->|Skipped| S[Reply: Not enough context]
    D -->|Good context| L

    C -->|No match| L[LLM Answer\nwith STM + LTM context]

    T --> F[Formatted Answer with Sources]
    L --> F
    G --> F
    M --> F
    S --> F
    F --> DB[(SqliteSaver\nSave state to SQLite)]
```

---

## What is HITL (Human-In-The-Loop)?

Normally the bot answers automatically. But what if the uploaded document doesn't actually contain the answer? The bot could give a confidently wrong answer — that's called hallucination.

**HITL prevents this by pausing the agent when it is not confident:**

1. User asks a question about an uploaded document
2. Bot searches the document with FAISS and finds very little context (under 200 characters)
3. Instead of guessing, the bot **pauses** and shows a warning in the UI
4. Human clicks **"Yes, try to answer"** or **"No, skip"**
5. Bot resumes with the human's decision

**How it works in the code:**

| Step | What happens |
|---|---|
| Low context detected | `chat_node` sets `awaiting_hitl = True` in `ChatState` |
| State saved | `SqliteSaver` writes the full state (including the flag) to SQLite on disk |
| UI detects the pause | `get_thread_hitl_state()` reads the state — frontend hides chat input, shows buttons |
| Human decides | Clicking a button sends `hitl_decision = "approve"` or `"skip"` back to the graph |
| Graph resumes | `chat_node` reads `hitl_decision` and acts accordingly |

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

## Query Rewriting explained

Raw user queries can be ambiguous or vague. The chatbot automatically detects and clarifies them before RAG retrieval.

**Examples:**
- "what about it" → "What is the main topic discussed in the document?"
- "tell me about that" → "Provide a detailed explanation of the key concepts"
- "what does it say" → "What information is available in the document?"

**How it works:**
1. User query comes in
2. `is_ambiguous_query()` checks for vague pronouns (it, that, this), vague verbs (tell, say, show), or very short queries
3. If ambiguous → `rewrite_query()` uses Gemini to clarify it
4. Rewritten query is used for RAG retrieval
5. Better retrieval = better answers

**Integration:**
- Happens transparently in `chatbotBackend.py` line 318 via `get_rag_context_with_rewriting()`
- Console shows: `[Rewrite] original → rewritten`

---

## RAG Metrics explained

The chatbot tracks retrieval quality using industry-standard metrics:

**Hit Rate@K** — What percentage of queries found the relevant document in the top-K results?
- Hit Rate@5: 87% means 87% of test queries found the answer in top-5 chunks
- Hit Rate@10: 95% includes larger result set
- Shows retriever health

**Mean Reciprocal Rank (MRR)** — On average, how high in the ranking does the relevant document appear?
- MRR = 1.0 → perfect (relevant doc always at rank 1)
- MRR = 0.5 → good (relevant doc typically at rank 2)
- MRR = 0.33 → okay (relevant doc typically at rank 3)

**Usage:**
```python
from chatbotBackend import run_rag_evaluation

metrics = run_rag_evaluation(
    thread_id="conversation-123",
    test_queries=["What is Python?", "Explain machine learning"],
    expected_sources=["python_guide.pdf", "ml_basics.pdf"]
)
# Outputs:
# Hit Rate@5:  100.0%
# Hit Rate@10: 100.0%
# MRR:         0.50
# Queries:     2
```

---

## Complete Chat Flow Diagram

```mermaid
flowchart TD
    U[User Query] --> R[remember_node<br/>Auto-save facts to LTM]
    R --> C{chat_node<br/>Route Intent}

    C -->|Greeting| G["Reply: Hello!"]
    C -->|Self query| M["Read LTM from Postgres<br/>Return stored facts"]
    C -->|Weather / Time<br/>News / Stock| T["Tool Call<br/>OpenWeather · DuckDuckGo<br/>Yahoo Finance · Clock"]
    C -->|Has documents| QR["Query Rewriting<br/>is_ambiguous_query?"]

    QR -->|Ambiguous| RW["rewrite_query<br/>Use LLM to clarify"]
    QR -->|Clear| D["D{RAG Retrieval<br/>FAISS top-4 chunks}"]
    RW --> D

    D -->|Low confidence| H["HITL Pause<br/>Ask human to Approve or Skip"]
    H -->|Approved| L["LLM Answer<br/>with citations"]
    H -->|Skipped| S["Reply: Not enough context"]
    D -->|Good context| L

    C -->|No match| L

    T --> F["Formatted Answer<br/>with Sources"]
    L --> F
    G --> F
    M --> F
    S --> F
    F --> DB[("SqliteSaver<br/>Save state to SQLite")]
```

---

## Project structure

```
Chatbot/
├── chatbotBackend.py            # Agent graph, chat_node, HITL logic, thread utilities
├── chatbotFrontend.py           # Streamlit UI — chat, sidebar, HITL buttons, export, recap
├── chatbot_memory.py            # STM + LTM memory — remember_node, recap greeting, Postgres store
├── chatbot_rag.py               # FAISS index building, document loading, RAG retrieval
├── chatbot_rag_metrics.py       # RAG evaluation metrics (Hit Rate@K, MRR)
├── chatbot_query_rewriter.py    # Query ambiguity detection and LLM-based rewriting
├── chatbot_tools.py             # Tool functions (weather, search, stock, time) + intent detectors
├── docker-compose.yml           # Postgres container for long-term memory
├── knowledge_base/              # Uploaded documents, one subfolder per thread
└── faiss_index/                 # FAISS indexes, one subfolder per thread
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
"hello"                          → greeting + memory recap if returning user
"weather in Mumbai"              → OpenWeather tool
"what time is it"                → system clock
"latest AI news"                 → DuckDuckGo search
"stock price of Apple"           → Yahoo Finance
"what do you know about me"      → reads your stored LTM facts
"my name is Sara, I like Python" → saves to LTM automatically
"summarize the PDF I uploaded"   → RAG over your document
```

Use the **⬇️ Download chat as .md** button in the sidebar to export any conversation.

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
- **Query rewriting** — ambiguity detection via heuristics, LLM-based clarification
- **RAG evaluation** — Hit Rate@K and MRR metrics for production monitoring
- **Memory architecture** — STM vs LTM design, auto-extraction via LLM
- **HITL pattern** — graph interruption, state persistence, human approval flow
- **Personalization** — LTM-powered recap greeting on every new session
- **User experience** — chat export to Markdown, streaming responses, file upload
- **Error handling** — API fallbacks (OpenWeather → DuckDuckGo), graceful degradation
- **Streamlit UI** — multi-thread management, sidebar controls, download button

---


