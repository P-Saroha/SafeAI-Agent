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

## Agent Workflow

The agent follows a deterministic flow with an LLM gate for tool usage:

1) **Input normalization**: extract latest user query and thread context.
2) **Memory prep**: load STM and LTM (skip memory injection for greetings).
3) **RAG routing**: if document intent is detected, fetch RAG context.
4) **Rule-based intent**: detect weather/news/stock/time quickly.
5) **LLM tool gate**: decide if external data is required.
6) **LLM router**: select the best tool when rules do not match.
7) **Tool execution**: call one tool and stop in the same turn.
8) **Fallbacks**: degrade gracefully if tool fails (weather -> search).
9) **Response formatting**: structured bullets + Sources.

### Routing Logic (Simplified)

- If weather/news/stock/time is detected -> call the specific tool.
- Else if LLM says tools are needed -> route to best tool.
- Else -> answer directly with the LLM.

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

## Mini Diagram

```mermaid
flowchart TD
  U[User Query] --> N[Normalize + Thread Context]
  N --> M[STM/LTM Memory Prep]
  M --> D{Document Intent?}
  D -->|Yes| RAG[RAG Retrieval]
  D -->|No| R[Rule-based Intent]
  R -->|Weather/News/Stock/Time| T[Tool Call]
  R -->|No match| G[LLM Tool Gate]
  G -->|No tool| L[LLM Answer]
  G -->|Tool needed| RT[LLM Router]
  RT --> T
  T --> F[Formatted Answer + Sources]
  RAG --> L
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
