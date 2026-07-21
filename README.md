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

## Architecture Flowchart

```mermaid
flowchart TD
    Start([User sends message]) --> Graph[LangGraph Entry]
    Graph --> Remember[remember_node<br/>Extract facts with LLM]
    
    Remember --> Postgres{Postgres<br/>available?}
    Postgres -->|Yes| SaveFacts[Save facts to<br/>PostgresStore LTM]
    Postgres -->|No| Skip1[Skip LTM save]
    SaveFacts --> Chat
    Skip1 --> Chat
    
    Chat[chat_node<br/>Route & respond]
    
    Chat --> CheckGreeting{is_greeting?}
    CheckGreeting -->|Yes| ReturnGreeting[Return: Hello!]
    CheckGreeting -->|No| CheckHITL
    
    CheckHITL{HITL pending<br/>& decision set?}
    CheckHITL -->|Yes - Approve| HITLApprove[Retrieve weak context<br/>Answer anyway with LLM]
    CheckHITL -->|Yes - Skip| HITLSkip[Return: Not enough info]
    CheckHITL -->|No| CheckSelf
    
    CheckSelf{is_self_query?}
    CheckSelf -->|Yes| LoadLTM[Load facts from<br/>PostgresStore]
    LoadLTM --> ReturnMemory[Return: Here's what I know]
    CheckSelf -->|No| CheckWeather
    
    CheckWeather{is_weather_query?}
    CheckWeather -->|Yes| ExtractLoc[extract_weather_location]
    ExtractLoc --> HasLoc{Location<br/>found?}
    HasLoc -->|No| AskLoc[Return: Which city?]
    HasLoc -->|Yes| CallWeather[call_weather<br/>OpenWeather API]
    CallWeather --> WeatherFail{API success?}
    WeatherFail -->|No| FallbackSearch[call_search<br/>DuckDuckGo fallback]
    WeatherFail -->|Yes| FormatWeather[format_weather_response]
    FallbackSearch --> ReturnWeather[Return: Weather + Sources]
    FormatWeather --> ReturnWeather
    CheckWeather -->|No| CheckTime
    
    CheckTime{is_time_query?}
    CheckTime -->|Yes| CallTime[call_datetime<br/>System clock]
    CallTime --> ReturnTime[Return: Date/Time + Sources]
    CheckTime -->|No| CheckNews
    
    CheckNews{is_news_query?}
    CheckNews -->|Yes| CallNews[call_search<br/>DuckDuckGo]
    CallNews --> FormatNews[format_search_response]
    FormatNews --> ReturnNews[Return: Top results + Sources]
    CheckNews -->|No| CheckStock
    
    CheckStock{is_stock_query?}
    CheckStock -->|Yes| ExtractSymbol[extract_stock_symbol]
    ExtractSymbol --> HasSymbol{Symbol<br/>found?}
    HasSymbol -->|No| AskSymbol[Return: Need ticker/company]
    HasSymbol -->|Yes| CallStock[call_stock<br/>Yahoo Finance]
    CallStock --> ReturnStock[Return: Price + Sources]
    CheckStock -->|No| CheckDocs
    
    CheckDocs{has_documents<br/>in thread?}
    CheckDocs -->|Yes| GetRAG[get_rag_context<br/>FAISS retrieval]
    GetRAG --> CheckDocIntent{is_document_question<br/>& context < 200 chars?}
    CheckDocIntent -->|Yes - Low confidence| PauseHITL[Set awaiting_hitl=True<br/>Save to SqliteSaver]
    PauseHITL --> ReturnHITL[Return: Warning + Ask approval]
    CheckDocIntent -->|No - Good context| BuildRAGPrompt[Build system prompt<br/>with RAG context + citations]
    BuildRAGPrompt --> GetSTM1[get_recent_messages<br/>Last 12 messages STM]
    GetSTM1 --> InvokeLLM1[llm.invoke with<br/>RAG context]
    InvokeLLM1 --> ReturnRAG[Return: Answer with [1][2] cites]
    
    CheckDocs -->|No| GetLTM[get_memory_as_text<br/>Load user facts from Postgres]
    GetLTM --> HasMemory{Memory<br/>exists?}
    HasMemory -->|Yes| BuildMemPrompt[Build system prompt<br/>with LTM context]
    HasMemory -->|No| BuildBasicPrompt[Build basic<br/>system prompt]
    BuildMemPrompt --> GetSTM2[get_recent_messages<br/>Last 12 messages STM]
    BuildBasicPrompt --> GetSTM2
    GetSTM2 --> InvokeLLM2[llm.invoke<br/>Gemini 2.5 Flash]
    InvokeLLM2 --> ReturnLLM[Return: Plain LLM answer]
    
    ReturnGreeting --> End([END])
    HITLApprove --> End
    HITLSkip --> End
    ReturnMemory --> End
    AskLoc --> End
    ReturnWeather --> End
    ReturnTime --> End
    ReturnNews --> End
    AskSymbol --> End
    ReturnStock --> End
    ReturnHITL --> FrontendHITL[Streamlit Frontend<br/>Shows Approve/Skip buttons]
    FrontendHITL -.Human clicks.-> ResumeGraph[Graph resumes with<br/>hitl_decision]
    ResumeGraph --> CheckHITL
    ReturnRAG --> End
    ReturnLLM --> End
    
    End --> SaveCheckpoint[(SqliteSaver<br/>Save conversation state)]
    SaveCheckpoint --> Done([Response shown to user])
    
    style Remember fill:#e1f5ff
    style Chat fill:#fff4e1
    style PauseHITL fill:#ffe1e1
    style FrontendHITL fill:#ffe1e1
    style Postgres fill:#e8f5e9
    style SaveFacts fill:#e8f5e9
    style LoadLTM fill:#e8f5e9
    style GetLTM fill:#e8f5e9
    style GetRAG fill:#f3e5f5
    style SaveCheckpoint fill:#f3e5f5
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
