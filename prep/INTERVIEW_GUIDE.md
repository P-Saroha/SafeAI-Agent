# AI Agent Chatbot — Complete Interview Preparation Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Key Concepts Explained](#key-concepts-explained)
4. [Interview Q&A](#interview-qa)
5. [Common Follow-up Questions](#common-follow-up-questions)
6. [What Interviewers Really Want to Know](#what-interviewers-really-want-to-know)
7. [Interview Preparation Checklist](#interview-preparation-checklist)

---

## Project Overview

### One-Line Pitch
"I built a production-grade AI agent that combines LangGraph for orchestration, FAISS + BM25 for hybrid document retrieval, and Human-In-The-Loop approval to prevent hallucinations — all with clean, beginner-friendly code."

### What Makes It Production-Grade
1. **Stateful orchestration** — LangGraph with SQLite checkpointing
2. **HITL (Human-In-The-Loop)** — Pauses on low-confidence answers
3. **Deterministic routing** — Tools before LLM calls
4. **Multi-thread isolation** — Each chat has separate docs + memory
5. **Quality metrics** — Hit Rate@K and MRR for monitoring
6. **Query clarity** — Automatic rewriting of ambiguous queries
7. **Hybrid retrieval** — 60% semantic (FAISS) + 40% keyword (BM25) = 90% accuracy

---

## Architecture Deep Dive

### System Design (High Level)

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│         (Chat UI, HITL buttons, Export)             │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│          LangGraph Agent (chatbotBackend)           │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ remember_node│  │  chat_node   │                │
│  │ (extract LTM)│  │ (route intent)                │
│  └──────────────┘  └──────────────┘                │
└────────────────┬────────────────────────────────────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
      ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────┐
  │ Tools  │ │ RAG    │ │ Memory │
  │(Weather│ │(FAISS) │ │(Postgres
  │ Stock) │ │+ BM25  │ │ SQLite)│
  └────────┘ └────────┘ └────────┘
```

### File Structure & Responsibility

| File | Lines | Responsibility |
|------|-------|---|
| **chatbotBackend.py** | 165 | Agent graph, routing logic, HITL state mgmt |
| **chatbotFrontend.py** | 200+ | Streamlit UI, thread mgmt, chat display |
| **chatbot_rag.py** | 280 | Hybrid retrieval (FAISS + BM25), chunking, indexing |
| **chatbot_query_rewriter.py** | 23 | Ambiguity detection, LLM clarification |
| **chatbot_rag_metrics.py** | 45 | Hit Rate, MRR calculation |
| **chatbot_memory.py** | 205 | STM/LTM extraction, Postgres storage |
| **chatbot_tools.py** | 170 | Weather, search, stock, time tools |

### Data Flow Example: "Summarize the PDF I uploaded"

```
1. Frontend captures query
   ▼
2. Streamlit calls: chatbot.invoke({"messages": [query]})
   ▼
3. LangGraph runs:
   - remember_node: Extract facts → save to Postgres LTM
   - chat_node: Route → detect "has documents" → RAG path
   ▼
4. Query Rewriting (if ambiguous):
   - is_ambiguous_query("summarize the pdf")? → False (specific)
   - Skip rewriting, use original query
   ▼
5. RAG Retrieval (Hybrid):
   - get_rag_context("summarize...", thread_id)
   - FAISS (60%) searches with embeddings: top-4 semantic chunks
   - BM25 (40%) searches with keywords: top-4 keyword chunks
   - Ensemble combines: 60% semantic score + 40% keyword score
   - Returns: "[1] doc.pdf (page 1): ...\n\n[2] doc.pdf (page 2): ..."
   ▼
6. Confidence Check (HITL):
   - Context length > 200 chars? → Yes
   - Proceed to LLM
   ▼
7. LLM Answer:
   - Prompt: "Answer using ONLY this context:\n[1]...[2]..."
   - Output: "Based on the document [1], the main points are..."
   ▼
8. Return to Frontend + Save state to SQLite
```

---

## Key Concepts Explained

### 1. LangGraph Agent
**What it is:** A state machine that defines how the agent thinks.

**Key parts:**
- **Nodes** — Functions that do work (remember_node, chat_node)
- **Edges** — How data flows between nodes (START → remember → chat → END)
- **State** — Shared dict that passes through all nodes (messages, thread_id, etc.)

**Why it's good:**
- Structured, predictable execution
- Easy to debug (trace each node)
- Supports checkpointing (save state, resume later)
- Supports interrupts (HITL pauses)

### 2. HITL (Human-In-The-Loop)
**What it is:** A mechanism to pause the agent and ask a human before acting.

**When it triggers:**
- User asks about a document
- RAG finds little context (< 200 chars)
- Instead of guessing, bot pauses

**How it works:**
```python
if _is_document_question(query) and len(rag_context) < HITL_MIN_CONTEXT_LENGTH:
    return {
        "messages": [AIMessage("Should I try to answer?")],
        "awaiting_hitl": True,
        "hitl_question": query,
    }
```

**Why it matters:**
- Prevents confident hallucinations
- Real production pattern (Anthropic, OpenAI)
- Shows maturity in handling AI uncertainty

### 3. RAG (Retrieval-Augmented Generation)
**What it is:** Search documents first, then use LLM to synthesize.

**Pipeline:**
```
1. Load PDF → 2. Chunk (1000 chars) → 3. Embed → 4. Index in FAISS
5. User query → 6. Embed query → 7. Search top-4 → 8. Pass to LLM
```

**Hybrid Retrieval (NEW):**
- FAISS (60%): Semantic search using embeddings
- BM25 (40%): Keyword search for exact matches
- Combined: 90% Hit Rate@5 (tested on 2 real PDFs, 30 queries)

**Why hybrid is better:**
- FAISS alone: 87% (good at semantic, misses exact keywords)
- BM25 alone: 75% (good at keywords, misses semantic meaning)
- Hybrid: 90% (balances both)

### 4. Query Rewriting
**What it is:** Detecting vague queries and clarifying them before RAG.

**Example:**
- User: "what about it" (ambiguous)
- System detects: "it" is vague pronoun
- LLM rewrites: "What is the main topic discussed?"
- Better retrieval happens

**Why it matters:**
- Garbage in, garbage out (GIGO)
- Query quality affects retrieval quality

### 5. Memory Architecture
**Short-Term (STM):**
- Last 12 messages in current conversation
- Passed to LLM in every call
- Disappears when chat ends

**Long-Term (LTM):**
- User facts: name, skills, interests, goals
- Stored in Postgres
- Persists across sessions

**Why separation?**
- LLMs have limited context windows
- STM keeps recent context fresh
- LTM keeps user knowledge across sessions

---

## Interview Q&A

### TIER 1: Expected Questions (Definitely Asked)

#### Q1: "Walk me through your project end-to-end"

**Good Answer (2-3 minutes):**

"I built an AI chatbot using LangGraph that does three things well:

1. **Intelligent Routing** — Before calling the LLM, I route to specific tools. Weather queries hit OpenWeather, stock queries hit Yahoo Finance. Saves latency and cost.

2. **Document Understanding** — Users upload PDFs. I index them in FAISS (vector database), retrieve relevant chunks, and pass to LLM with citations. No hallucinations because answers come from actual documents.

3. **Human Approval on Uncertainty** — Here's the key production feature: when RAG finds very little context (< 200 chars), I don't guess. The graph *pauses* and asks the human: 'Should I try to answer or skip?' This is HITL (Human-In-The-Loop), used by Anthropic.

**The hybrid retriever:** I combine FAISS (60% semantic) and BM25 (40% keyword) to get 90% accuracy instead of 87% (semantic-only) or 75% (keyword-only).

The backend is a LangGraph agent with two nodes:
- **remember_node**: Extracts facts from user messages and saves to Postgres
- **chat_node**: Routes the query to tools, RAG, or pure LLM based on intent

State is checkpointed to SQLite, so conversations survive page reloads. Frontend is Streamlit. Code is intentionally simple — no over-engineering."

---

#### Q2: "What's HITL and why does it matter?"

**Good Answer (1-2 minutes):**

"HITL stands for Human-In-The-Loop. It's a pattern where the AI system pauses and asks a human when it's uncertain.

In my chatbot:
- User asks about an uploaded document
- I search the document with FAISS
- If I find only 200 chars of context (very little), I know the answer is probably not in the document
- Instead of the LLM making something up (hallucination), the system **pauses**
- The UI shows: 'I don't have enough info. Should I try to answer anyway?'
- Human clicks 'Yes' or 'No'
- System resumes with that decision

**Why it matters:**
1. **Prevents hallucinations** — The LLM's #1 failure mode is confident wrong answers
2. **Keeps humans in control** — AI doesn't pretend to know things it doesn't
3. **Production pattern** — Anthropic, OpenAI use this
4. **Traceable** — You know when the AI paused and what the human decided

It's implemented using LangGraph's interrupt feature. State is saved to SQLite with `awaiting_hitl=True`."

---

#### Q3: "How does your RAG system work? What are the failure modes?"

**Good Answer (2-3 minutes):**

"RAG is Retrieval-Augmented Generation. I search the user's documents first, then the LLM synthesizes.

**My Pipeline:**
1. User uploads PDF
2. Chunk it (1000 chars per chunk, 150 char overlap)
3. Convert chunks to embeddings
4. Index into FAISS (fast similarity search)
5. User asks a question
6. Embed the question the same way
7. Search FAISS for top-4 most similar chunks
8. **NEW: Also search BM25 for top-4 keyword matches**
9. **Combine both (60% FAISS + 40% BM25) using EnsembleRetriever**
10. Pass combined chunks to LLM with instruction: 'Answer using ONLY this context'
11. LLM cites sources: '[1] doc.pdf (page 2): ...'

**Failure Modes & How I Handle Them:**

1. **Ambiguous query** — User asks 'what about it'
   - Solution: Query Rewriting detects vague pronouns, clarifies first

2. **Low confidence retrieval** — FAISS finds no relevant chunks
   - Solution: HITL pause. Show human the problem and ask before answering

3. **Outdated embeddings** — Old index is stale
   - Solution: 'Rebuild RAG Index' button deletes old, rebuilds fresh

4. **Hallucination on edge cases** — LLM might extrapolate beyond context
   - Solution: System prompt: 'Answer ONLY from context'

5. **Poor chunking** — Related info split at boundaries
   - Solution: 1000-char chunks with 150-char overlap = good coverage

6. **Embedding quality** — Bad embeddings = bad retrieval
   - Solution: Track Hit Rate@K and MRR. If metrics drop, investigate.

**Why my approach is solid:**
- Metrics-driven (I measure Hit Rate and MRR)
- Graceful degradation (fallbacks at each layer)
- Per-thread isolation (no cross-contamination)
- Hybrid approach (90% accuracy, better than either alone)"

---

#### Q4: "Why did you choose LangGraph over alternatives like CrewAI or LlamaIndex?"

**Good Answer (1.5 minutes):**

"I chose LangGraph for three reasons:

1. **State Management** — I needed structured way to track state (messages, memory, HITL flags). LangGraph's `StateGraph` is clean and predictable.

2. **Interrupts** — HITL requires pausing the graph, saving state to disk, and resuming later. LangGraph was designed for this. CrewAI is agent-only (no pause). LlamaIndex is query-focused, not agent-focused.

3. **Debugging** — LangGraph lets me trace execution through nodes. Others are more of a black box.

**Tradeoffs:**
- LangGraph is lower-level (I write more code)
- But I get more control (HITL, checkpointing, routing)
- For a production system needing human approval, this is the right tradeoff

If I just needed RAG without HITL, I'd use LlamaIndex. But HITL + interrupts = LangGraph."

---

#### Q5: "How do you prevent hallucinations in your RAG?"

**Good Answer (1.5 minutes):**

"Hallucinations happen when the LLM makes up information not in documents. I prevent this at multiple levels:

1. **System Prompt** — Tell the LLM explicitly: 'Answer ONLY from the provided context. If not in context, say so clearly.'

2. **Context Only** — Don't pass conversation history or LTM facts to the RAG answer. Only document chunks.

3. **HITL on Low Confidence** — If RAG retrieves < 200 chars, I pause and ask human. LLM only answers when sufficient ground truth exists.

4. **Query Clarity** — If query is vague ('what about it'), I rewrite it first. Better query = better retrieval = less guessing.

5. **Metrics** — I track Hit Rate@K. If it drops below 80%, something's wrong.

6. **Hybrid Retrieval** — FAISS + BM25 together (90%) catch more relevant docs than either alone (87% or 75%)

**Philosophy:**
Hallucinations are a symptom of low retrieval quality. Fix retrieval quality + add HITL, and hallucinations almost disappear."

---

#### Q6: "How do you evaluate your RAG system? How do you know it's working?"

**Good Answer (2 minutes):**

"I use two metrics:

**1. Hit Rate@K**
- What % of test queries found relevant document in top-K results?
- Hit Rate@5 = 87% means 87% of queries found answer in top-5 chunks
- Industry standard: Most companies target > 80%

**2. Mean Reciprocal Rank (MRR)**
- On average, at what rank does relevant document appear?
- MRR = 1.0 → perfect (always rank 1)
- MRR = 0.5 → good (typically rank 2)
- Rewards relevance ranking, not just presence

**My Results:**
- 2 real PDFs tested
- 30 diverse test queries
- Hybrid (FAISS 60% + BM25 40%): 90% Hit Rate@5
- RAG Interview PDF: 86.7%
- Fine-Tuning PDF: 93.3%

**If metrics drop:**
- Hit Rate drops → embedding quality problem or chunking issue
- MRR drops → relevant docs ranked lower, need reranking
- Signal: Time to retrain index or adjust chunk size

**Why these metrics:**
- Easy to calculate (no human judges needed)
- Track system health
- Production pattern (Databricks uses these)"

---

### TIER 2: Follow-up Questions

#### Q7: "What's the difference between STM and LTM? Why separate them?"

**Good Answer (1.5 minutes):**

"**Short-Term Memory (STM)** = Last 12 messages in current chat. Passed to LLM every time. Dies when session ends.

**Long-Term Memory (LTM)** = User facts (name, skills, interests) stored in Postgres. Persists across sessions.

**Why separate?**

LLMs have limited context windows. If I pass all 500 messages from all past chats, context gets bloated. Instead:
- STM = recent, high-signal info (last conversation)
- LTM = persistent, user-level info (who is this person?)

**Example:**
- Session 1: User says 'My name is Sara and I like Python'
- Remember_node extracts → saves to Postgres LTM
- Session 1 ends

- Session 2 (next day): User says 'Hi'
- Remember_node loads LTM → finds 'Sara likes Python'
- Greets: 'Hello Sara! Ready to discuss Python?'
- No old messages passed → context stays small

**Why it matters:**
- Context window efficiency
- User personalization across sessions
- Scalability (one user, many sessions)"

---

#### Q8: "How do you handle token limits? What if user uploads 100-page PDF?"

**Good Answer (1.5 minutes):**

"Good question. LLMs have context limits. Gemini 2.5 Flash has ~8K input tokens.

**How I handle large documents:**

1. **Chunking** — I split PDF into 1000-char chunks. 100-page PDF becomes ~400-500 chunks.

2. **Retrieval** — When user asks, I search FAISS for top-4 chunks (~4K chars), not all 100 pages.

3. **Relevance** — FAISS returns *most similar* chunks using embeddings, not random ones.

**Example:**
- 100-page PDF loaded and chunked
- User: 'Summarize page 50'
- I search: 'page 50' in FAISS
- Top-4 chunks are from page 50
- LLM gets only those (~2-3K chars)
- Stays under token limit

**Tradeoff:**
- Pro: Scales to any size PDF
- Con: Might miss context spanning many pages
- Solution: HITL pauses if confidence is low"

---

#### Q9: "Why use FAISS? What are the alternatives?"

**Good Answer (1.5 minutes):**

"FAISS is Facebook's vector search library. I chose it for:

1. **Speed** — Indexes in C++. Searches in milliseconds, even for 10K+ vectors.

2. **Simplicity** — No external service. Runs locally. No API calls, no cost.

3. **Beginner-friendly** — Easy to use. One command to save/load.

**Alternatives & tradeoffs:**

| Alternative | Pro | Con |
|---|---|---|
| **Pinecone** (cloud) | Managed, scales | Costs money, external dependency, latency |
| **Weaviate** (self-hosted) | Rich features | More complex setup, more RAM |
| **Milvus** (cloud/self) | Open source, scalable | Requires setup, overkill for small |
| **ChromaDB** (local) | Simple | Slow for large indexes |

**My choice:**
FAISS is right for this project because:
- Each chat has separate index (small, isolated)
- No external services (no single point of failure)
- Beginner-friendly code (clear goal)

**If I scale to 1M documents:**
I'd switch to Pinecone or Weaviate for distributed search."

---

#### Q10: "How would you deploy this to production?"

**Good Answer (2 minutes):**

"I'd do this in stages:

**Stage 1: Package It (30 mins)**
- Create Dockerfile with dependencies
- Mount volumes for knowledge_base/ and faiss_index/
- Expose port 8501 (Streamlit)

**Stage 2: Deploy to Cloud (1-2 hours)**
- Push to GitHub
- Deploy to Fly.io or Railway (free tier, easy)
- PostgreSQL in cloud (AWS RDS or Railway)

**Stage 3: Add Observability (2-3 hours)**
- Logging: Printed to stdout, collected by Fly.io
- Metrics: Track hit_rate, latency, error_rate
- Errors: Sentry for error tracking

**Stage 4: Scale (as needed)**
- FAISS too large? Switch to Pinecone
- Postgres bottleneck? Add read replicas
- Streamlit hitting limits? Deploy backend as FastAPI

**What I wouldn't do initially:**
- Kubernetes (overkill for MVP)
- Multi-region (not needed yet)
- Advanced caching (measure first)

**Risk management:**
- Automated backups of Postgres
- SQLite is single-file (easy to backup)
- FAISS indexes are re-buildable from source"

---

### TIER 3: Edge Cases & Hard Questions

#### Q11: "What if user uploads conflicting information in two PDFs?"

**Good Answer (1.5 minutes):**

"Example: Document A says 'Python created 1989', Document B says '1991'.

**My system's behavior:**
1. User: 'When was Python created?'
2. Hybrid search returns top-4 chunks from both docs
3. LLM sees both: '[1] doc_a.pdf: ...1989...' and '[2] doc_b.pdf: ...1991...'

**How I handle it:**
- LLM should point out conflict
- System doesn't try to resolve it (not its job)
- Cite both sources, let human decide

**If I wanted to handle better:**
1. **Reranking** — Use LLM to rank by reliability
2. **Source tagging** — Mark docs as 'official' vs 'draft'
3. **Conflict detection** — Explicitly identify contradictions

**Why I don't now:**
- Added complexity for rare edge case
- HITL catches it anyway (human sees both)
- Better to keep simple and scale if needed"

---

#### Q12: "What happens if FAISS index gets corrupted?"

**Good Answer (1 minute):**

"FAISS saves two files per thread:
- `index.faiss` — the actual index
- `index.pkl` — metadata

If corrupted:
1. User uploads docs
2. System tries to load index → error
3. Catches exception, rebuilds from source docs
4. Takes ~30 seconds
5. User sees: 'Rebuilding index, please wait...'

**Code:**
```python
try:
    vectorstore = FAISS.load_local(index_dir, embeddings)
except Exception as e:
    print(f\"Rebuilding: {e}\")
    docs = _load_documents(files)
    vectorstore = FAISS.from_documents(docs, embeddings)
```

**Safety:**
- Source documents immutable (saved in knowledge_base/)
- Can always rebuild
- No data loss

**For production:**
- Backup FAISS index to S3 monthly
- Monitor index size
- Version control index"

---

#### Q13: "How do you handle concurrent users? Race conditions?"

**Good Answer (2 minutes):**

"SQLite has built-in locking, so concurrent writes are serialized.

**My design:**
- Each user has unique `user_id`
- Each chat has unique `thread_id`
- Each thread has separate FAISS index folder

**Potential race condition:**
- User A and B upload files to same thread simultaneously
- Both trigger FAISS rebuild
- Could conflict

**Mitigation:**
- If different threads: no conflict (separate folders)
- If same thread (rare): SQLite lock serializes writes
- Worst case: one rebuild happens twice (idempotent)

**For production at scale:**
- SQLite ≈ light concurrency (< 100 users)
- At >100 concurrent: switch to PostgreSQL backend
- Redis for caching FAISS indexes

**Current assumption:**
- Few dozen concurrent users max
- If that changes, refactor to Redis + PostgreSQL"

---

#### Q14: "Design this chatbot to handle 1M users. What changes?"

**Good Answer (3-4 minutes):**

"Great scaling question. Let me identify bottlenecks:

**Current Bottlenecks:**

1. **SQLite** — Single file, one writer. OK for 100 users, fails at 10K+.
   - Solution: PostgreSQL or MongoDB

2. **FAISS** — Indexes live locally per thread. 1M users = 1M indexes (terabytes).
   - Solution: Centralized Pinecone or Weaviate

3. **Gemini API** — Rate limited, each call ~500ms.
   - Solution: Queue system (RabbitMQ, Kafka)

4. **Streamlit** — Single process, can't handle 1M concurrent users.
   - Solution: FastAPI backend + React frontend

**Scaled Architecture:**

```
Load Balancer (nginx)
    ├─→ API Server 1 (FastAPI) ─→ PostgreSQL
    ├─→ API Server 2 (FastAPI) ─→ Pinecone
    └─→ API Server 3 (FastAPI) ─→ Redis

React/Next.js Frontend ← → All servers
```

**Gradual Migration Path:**

1. **0-100 users** → Current (SQLite + FAISS)
2. **100-1K users** → Add Redis cache, monitoring
3. **1K-10K users** → PostgreSQL for state, Pinecone for vectors
4. **10K-100K users** → FastAPI backend, React frontend
5. **100K-1M users** → Distributed training, index optimization

**What I wouldn't change:**
- LangGraph (still good)
- HITL pattern (still valuable)
- Hybrid retrieval (still effective)"

---

#### Q15: "Debug scenario: Hit Rate suddenly drops from 90% to 40%"

**Good Answer (2-3 minutes):**

"Debugging systematically:

**Step 1: Confirm the drop**
```python
metrics = evaluate_retriever(thread_id, test_queries)
print(f\"Hit Rate@5: {metrics['hit_rate@5']}\")  # 0.40
```

**Step 2: Identify culprit (Checklist)**

**A. Embeddings changed?**
- Check `.env`: Is RAG_EMBEDDING_BACKEND still 'google'?
- API key expire?
- Test: `embeddings = _get_embeddings()` directly
- If hash fallback → embedding quality dropped
- Solution: Renew API key

**B. Documents changed?**
- User delete important docs?
- FAISS index corrupted?
- Test: Load index, search for known doc
- Check file timestamps
- Solution: Rebuild index

**C. Chunking changed?**
- Did chunk_size change from 1000 to something else?
- Chunks too small → semantic loss
- Chunks too large → less precise
- Solution: Revert params, rebuild

**D. Query distribution changed?**
- Are test queries different format?
- Were test sources updated?
- Solution: Run on old queries

**E. Index overwritten?**
- Rebuild with partial documents?
- Index size larger/smaller?
- Solution: Restore from backup

**Debug Script:**
```python
print(\"[Debug] Embedding quality...\")
embeddings = _get_embeddings()
print(f\"Embedding dim: {len(test_vec)}\")

print(\"[Debug] Index integrity...\")
retriever = _build_retriever(thread_id)
docs = retriever.invoke(\"test\")

print(\"[Debug] Source docs...\")
files = _get_supported_files(thread_id)
print(f\"Found {len(files)} files\")

print(\"[Debug] Running eval...\")
metrics = evaluate_retriever(thread_id)
print(f\"Hit Rate: {metrics['hit_rate@5']}\")
```

**80/20 Most Likely Causes:**
1. **50%** — Embedding backend failed (API key expired)
2. **20%** — Index corrupted or deleted
3. **15%** — Test set changed
4. **10%** — Documents deleted
5. **5%** — Other

**First thing to check:** Is Google API key still valid?"

---

## What Interviewers Really Want to Know

### 1. **Can You Think Like a Systems Engineer?**
They ask: "Walk me through your project end-to-end"
They really want: Can you see the whole system? Routing, storage, UI, error handling?

### 2. **Do You Understand Your Framework?**
They ask: "Why LangGraph? What are alternatives?"
They really want: Did you understand the decision or just copy-paste?

### 3. **Do You Think About Production?**
They ask: "How do you prevent hallucinations?"
They really want: Do you think failure modes? Or just happy path?

### 4. **Can You Communicate?**
They ask: Any technical question
They really want: Can you explain complex things simply? 2-3 min answers, not 10?

### 5. **Are You Humble?**
They ask: "What would you do differently?"
They really want: Do you see your code's limitations?

---

## Interview Preparation Checklist

Before the interview, make sure you can answer:

- [ ] What does each file do? (chatbotBackend, chatbotFrontend, chatbot_rag, etc.)
- [ ] Explain HITL in 30 seconds
- [ ] Draw the data flow for "user asks about PDF"
- [ ] What's the difference between STM and LTM?
- [ ] Why hybrid retrieval (FAISS 60% + BM25 40%) instead of pure semantic?
- [ ] Your 90% Hit Rate result and how you tested it (2 PDFs, 30 queries)
- [ ] Failure modes and how you mitigate them
- [ ] Tradeoff analysis (LangGraph vs alternatives, FAISS vs Pinecone)
- [ ] What you'd do differently if rebuilding today
- [ ] How you'd deploy to production
- [ ] How you measure success

---

## Final Talking Points

### The Pitch (30 seconds)
"I built a production-grade AI agent using LangGraph with FAISS + BM25 hybrid retrieval and Human-In-The-Loop approval to prevent hallucinations. I tested it on 2 real PDFs with 30 queries and achieved 90% Hit Rate@5."

### The Differentiator (Why you)
"Most chatbot projects don't think about hallucinations. I made HITL a core architecture, not an afterthought. I also chose hybrid retrieval (90% vs 87% for semantic-only), showing I optimize for results, not simplicity."

### The Reality (Honesty)
"It's not perfect—no unit tests yet, would switch to Pinecone if scaling. But it's complete, understandable, and ready to run. I optimized for clean code over complexity."

---

**Good luck! You've built something solid. Own it.** 

---

## Running Your Tests

To demonstrate your claims:

```bash
python RAG_TEST_CASES.py
```

**Output shows:**
- RAG Interview PDF: 86.7% Hit Rate
- Fine-Tuning PDF: 93.3% Hit Rate
- Overall: 90% Hit Rate@5 (27/30 queries)

Tell interviewers: "Here's my test code. I tested against 2 real PDFs with 30 diverse queries. My actual Hit Rate@5 is 90%."

