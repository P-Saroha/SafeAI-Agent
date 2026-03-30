# AI Chatbot (LangGraph + RAG + Tools)

A production-style chatbot that blends deterministic tool routing, LLM responses, and document-based RAG. It supports memory (short-term + long-term), real-time tools (weather, news, stock, time), and a Streamlit UI.

## Highlights

- Hybrid agent flow (rules + LLM tool gate) to choose the right tool at the right time
- RAG over local documents with per-thread knowledge bases
- Long-term memory backed by Postgres + short-term conversation memory
- Deterministic formatting with sources for tool-driven answers
- Real-time weather via OpenWeather API
- Built-in HITL gating for low-confidence RAG answers

## Features

- **Tool routing**: rules for weather/news/stock/time, plus LLM-based fallback
- **Weather tool**: OpenWeather current conditions, clean output format
- **Search tool**: DuckDuckGo search with structured results + sources
- **Stock tool**: Yahoo Finance pricing via `yfinance`
- **Time tool**: local system time
- **Memory**:
  - Structured long-term memory (Postgres)
  - Auto-memory for stable facts
  - Short-term memory for the active chat
- **RAG**:
  - Upload docs to `knowledge_base/`
  - Per-thread FAISS index
  - Citations for retrieved context
- **Streamlit UI**:
  - Conversation threads
  - Tool trace
  - RAG controls

## Agent Workflow (Exact)

The runtime flow matches the current `chat_node` logic:

1) **Read latest query** and set `thread_id` and `user_id`.
2) **Greeting short-circuit**: if greeting, respond immediately and skip memory.
3) **Link recall**: if user asks for the last weather link, return it.
4) **Pending weather**: if the bot asked for a city, call `get_weather` with the new location.
  - If OpenWeather succeeds, return structured weather.
  - If it fails, fallback to `search_tool`.
5) **Awaiting approval**: if HITL approval is pending, repeat the approval request.
6) **Mode resolution**: auto -> `agent_only` if no docs, `rag_only` for doc intent, else `hybrid`.
7) **RAG context**: load RAG context when in `hybrid` or `rag_only`.
8) **HITL decision handler**: resolve explicit Approve/Regenerate responses.
9) **Tool routing**:
  - If no rule match, and not a simple question, the LLM gate decides if a tool is needed.
  - If needed, the LLM router picks `weather`, `time`, `news`, `stock`, or `search`.
10) **Self-query**: if user asks about memory, return saved LTM + STM facts.
11) **Low-confidence RAG**: if RAG context is weak, ask for HITL approval.
12) **Tool handlers** in order:
   - Weather -> `get_weather` (fallback to `search_tool`).
   - Time -> `get_current_date_time`.
   - News -> `search_tool`.
   - Stock -> `get_stock_price` (or ask for ticker).
   - Generic tool-needed -> `search_tool`.
13) **LLM answer**: if no tool matched, answer using LLM with RAG context (if any) and memory context (if relevant).
14) **Fallback to tool output**: if LLM returns empty, use last tool output.

### Routing Logic (Exact Summary)

- Rules first (weather/time/news/stock), then LLM gate, then LLM router.
- One tool per turn; the answer is finalized from that tool output.
- RAG is only used in `hybrid` or `rag_only` modes and is gated by HITL when low confidence.

## Tools

- `get_weather`: OpenWeather current conditions
- `search_tool`: DuckDuckGo search for latest info
- `get_stock_price`: Yahoo Finance via `yfinance`
- `get_current_date_time`: local system time
- `rag_search`: local knowledge base retrieval

## Memory Flow

- **STM (short-term memory)**: recent messages in the active chat
- **LTM (long-term memory)**: Postgres-backed structured user facts
- **Auto-memory**: stable facts captured automatically

The agent merges STM + LTM as context, but avoids memory injection for greetings.

## Mini Diagram (Exact)

```mermaid
flowchart TD
  U[User Query] --> Greet{Greeting}
  Greet -->|Yes| Hello[Return greeting]
  Greet -->|No| Link{Asking for last link}
  Link -->|Yes| LinkOut[Return last weather link]
  Link -->|No| Pending{Pending weather}
  Pending -->|Yes| WeatherPending[Call get_weather and respond]
  Pending -->|No| Mode[Resolve mode]
  Mode --> RAG{RAG context needed}
  RAG -->|Yes| RAGCtx[Fetch RAG context]
  RAG -->|No| Route[Tool routing]
  RAGCtx --> Route
  Route -->|Rules match| Tool[Call tool]
  Route -->|No match| Gate[LLM tool gate]
  Gate -->|No tool| LLM[LLM answer]
  Gate -->|Tool| Router[LLM router]
  Router --> Tool
  Tool --> Answer[Formatted answer with sources]
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r ../requirements.txt
```

### 2) Configure environment

Create or update `.env` in the project root:

```
GOOGLE_API_KEY=your_gemini_key
OPENWEATHER_API_KEY=your_openweather_key
LTM_POSTGRES_URI=postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable
```

### 3) Start Postgres (for long-term memory)

```bash
docker compose up -d
```

### 4) Run the app

```bash
streamlit run chatbotFrontend.py
```

## Example Queries

- Weather: "today weather of Delhi"
- News: "latest tech news"
- Stock: "stock price of ORCL"
- Time: "what is the time now"
- RAG: "summarize the PDF I uploaded"
- Memory: "tell me about myself"

## Project Structure

```
Chatbot/
  chatbotBackend.py       # Agent logic, tools, memory, RAG
  chatbotFrontend.py      # Streamlit UI
  knowledge_base/         # Uploaded docs for RAG
  faiss_index/            # Per-thread FAISS indexes
  docker-compose.yml      # Postgres for long-term memory
```

## Notes for Recruiters

This project demonstrates:

- Applied LLM engineering (routing + tool usage + structured outputs)
- RAG design with local docs and citations
- Memory management (short-term vs long-term, auto-memory)
- Production concerns (tool errors, fallbacks, HITL gating)

## Tech Stack

- LangGraph, LangChain
- Streamlit UI
- FAISS for vector search
- Postgres for long-term memory
- OpenWeather API for real-time weather
- DuckDuckGo Search + Yahoo Finance

## Roadmap Ideas

- Add unit tests for tool routing and memory
- Add deployment recipe (Docker + Streamlit Cloud)
- Add structured evals for RAG correctness
