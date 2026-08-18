# End-to-End Testing Guide

Complete testing scenarios to verify all chatbot features work correctly.

---

## Prerequisites

Before running tests, ensure:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start PostgreSQL (for long-term memory)
docker compose up -d

# 3. Verify .env file has all required keys
# - GROQ_API_KEY=xxx
# - OPENWEATHER_API_KEY=xxx
# - LANGSMITH_API_KEY=xxx (optional)
# - LANGSMITH_TRACING=true/false
```

---

## Part 1: Quick Smoke Test (5 minutes)

### 1.1 Start the Chatbot

```bash
streamlit run chatbotFrontend.py
```

Browser should open at `http://localhost:8501`

**Check:**
- ✅ UI loads without errors
- ✅ Sidebar shows "SafeAI Agent" or project name
- ✅ Chat input field is active

### 1.2 Test Greeting

**Query:** `hello`

**Expected:**
```
Hello! How can I help?
```

**Check:**
- ✅ Bot responds immediately
- ✅ No API errors in terminal
- ✅ Response appears in chat

### 1.3 Test Time Query

**Query:** `what time is it`

**Expected:**
```
Current time: [current time]
Current date: [current date]
```

**Check:**
- ✅ Time matches your system
- ✅ No API calls made (local system time)
- ✅ Format is clear

---

## Part 2: External API Tests (10 minutes)

### 2.1 Test Weather API

**Scenario A: Standard format**
```
Query: weather in Delhi
Query: Mumbai weather
Query: what is today bangalore weather
Query: tell me rohtak weather
```

**Expected for all:**
```
| Location | Delhi |
| Temperature | 32°C |
| Feels like | 36°C |
| Condition | Clear |
| Humidity | 65% |
| Wind speed | 4.2 m/s |

🔧 Powered by: OpenWeather API
```

**Checks:**
- ✅ All 4 query formats work
- ✅ Correct location returned
- ✅ Temperature is realistic (0-50°C)
- ✅ Humidity 0-100%
- ✅ Wind speed > 0
- ✅ Source attribution present

**Scenario B: Edge cases**
```
Query: temperature in New York
Query: forecast London
Query: weather for Sydney today
```

**Expected:**
- ✅ All should return data (not error)
- ✅ Location autocorrected to proper name

**Scenario C: Error case**
```
Query: weather in XyZaBcD (fake city)
```

**Expected:**
```
City not found: [error message]
```

**Check:**
- ✅ Graceful error (no crash)

### 2.2 Test Stock Price API

**Queries:**
```
Tell me stock price of TCS
What is Apple stock price?
Show me Microsoft price
Stock price of Google
```

**Expected:**
```
Stock Price — TCS
Current Price: ₹3,521.05
Market Status: Open
Data source: Yahoo Finance
```

**Checks:**
- ✅ All formats extract stock symbol correctly
- ✅ Price is realistic (not 0 or negative)
- ✅ Shows market status if available
- ✅ Source attribution correct

**Error case:**
```
Query: stock price of XyZaBcDLtd
```

**Expected:**
```
Could not fetch stock price for 'XyZaBcDLtd'.
```

**Check:**
- ✅ No crash, graceful error

### 2.3 Test News/Web Search

**Queries:**
```
latest AI news
what's happening in tech today
Tell me about recent machine learning breakthroughs
Tell me about Groq AI
```

**Expected:**
```
[Multiple news headlines with sources]
- [Headline 1] - [Source 1]
- [Headline 2] - [Source 2]
...

🔧 Powered by: DuckDuckGo
```

**Checks:**
- ✅ Returns 3-5 relevant headlines
- ✅ Sources are included
- ✅ Results are current (not outdated)
- ✅ No hallucinations (real news only)

---

## Part 3: Memory Tests (10 minutes)

### 3.1 Short-Term Memory (STM)

**Query 1:** `My name is Alex and I work at Google`
**Expected:** Bot acknowledges

**Query 2:** `I also love Python programming`
**Expected:** Bot acknowledges

**Query 3:** `What do you know about me so far?`
**Expected:**
```
Based on what you've told me:
- Name: Alex
- Company: Google
- Interests: Python programming
```

**Checks:**
- ✅ Bot remembers all facts in same conversation
- ✅ No information lost

### 3.2 Long-Term Memory (LTM)

**Step 1: In conversation A:**
```
Query: My name is Alex, I love AI
Expected: Bot acknowledges
```

**Step 2: Refresh browser (new conversation)**

**Step 3: In conversation B:**
```
Query: what do you know about me?
Expected:
Hello Alex! Welcome back! 👋
I remember you love AI.
```

**Checks:**
- ✅ Bot greets with personalized message
- ✅ Bot recalls facts from previous conversation
- ✅ LTM database working
- ✅ Check PostgreSQL is running

### 3.3 Memory Edge Case

**Query in new conversation:** `Forget about me` or `reset my memory`
**Expected:** Bot acknowledges (or explains memory works automatically)

---

## Part 4: RAG (Retrieval-Augmented Generation) Tests (15 minutes)

### 4.1 Upload a Document

**Setup:**
1. Create a test PDF: `test_doc.pdf` with content:
   ```
   Company: Acme Corp
   Founded: 2020
   CEO: John Smith
   Headquarters: San Francisco
   Revenue: $10M
   
   Key Products:
   1. Widget A - Sales tool
   2. Widget B - Analytics tool
   3. Widget C - Automation tool
   ```

2. Upload to chatbot via sidebar

**Expected:**
- ✅ File uploads without error
- ✅ Progress indicator shows indexing
- ✅ "Document ready" confirmation

### 4.2 Test Semantic Search (Concept-based)

**Query 1:** `What are the main products?`
**Expected:**
```
Based on the document:
1. Widget A - Sales tool
2. Widget B - Analytics tool
3. Widget C - Automation tool

[1] test_doc.pdf (page 1)
```

**Query 2:** `Tell me about the company`
**Expected:**
```
Acme Corp is a company founded in 2020, headquartered in San Francisco...
[1] test_doc.pdf (page 1)
```

**Checks:**
- ✅ Both queries return relevant excerpts
- ✅ Citations included ([1], [2], etc.)
- ✅ Information accurate from document

### 4.3 Test Keyword Search (Exact match)

**Query:** `CEO name`
**Expected:**
```
The CEO is John Smith.
[1] test_doc.pdf
```

**Checks:**
- ✅ Exact match found
- ✅ Citation included

### 4.4 Test Hybrid Search (Semantic + Keyword)

**Query:** `What is the revenue and where is the company?`
**Expected:**
```
Acme Corp has revenue of $10M and is headquartered in San Francisco.
[1] test_doc.pdf (page 1)
```

**Checks:**
- ✅ Both semantic AND keyword parts answered
- ✅ Proper synthesis of information

### 4.5 Test Retrieval Quality Metrics

**Check LangSmith Dashboard (if enabled):**
1. Go to https://smith.langchain.com
2. Navigate to "SafeAI Agent" project
3. Look at recent traces

**Verify:**
- ✅ Hit Rate@5 shows in logs
- ✅ MRR (Mean Reciprocal Rank) calculated
- ✅ Token usage tracked

**Or check logs in terminal:**
```
[RAG] Building FAISS index...
[RAG] Hit Rate@5: 0.91
[RAG] MRR: 0.82
```

### 4.6 Test HITL (Human-In-The-Loop)

**Query:** `Tell me about quarterly revenue` (info NOT in document)
**Expected:**
- UI shows warning message
- "Yes, try to answer" and "No, skip" buttons appear
- Chat input hidden

**If click "Yes":**
- Bot attempts answer (may hallucinate)
- Response tagged with low confidence

**If click "No":**
- Bot says "Not enough context in document"
- Asks if you want web search instead

**Checks:**
- ✅ HITL pause triggered correctly
- ✅ Buttons appear and respond
- ✅ State saved (refresh doesn't lose it)
- ✅ User can decide before bot answers

### 4.7 Test Query Rewriting

**Query:** `what about it` (ambiguous)
**Expected:**
```
[Rewrite] "what about it" → "What information is provided about the main topic?"
[Retrieval] Searching for rewritten query...
```

**Check terminal:**
- ✅ Console shows rewrite happening
- ✅ Better results returned after rewrite

---

## Part 5: LangSmith Tracing Test (5 minutes)

### 5.1 Verify LangSmith Is Enabled

**Check .env:**
```
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=SafeAI Agent
```

### 5.2 Make a Query

**Query:** `What time is it?`

### 5.3 Check LangSmith Dashboard

1. Go to https://smith.langchain.com
2. Click "Traces"
3. Find your most recent query

**Verify trace shows:**
- ✅ `remember_node` execution
- ✅ `chat_node` execution
- ✅ Tool call (if applicable)
- ✅ LLM call (Groq API)
- ✅ Latency (typically 500-2000ms)
- ✅ Token usage displayed

**Example trace structure:**
```
Query: "What time is it?"
├─ remember_node (100ms)
├─ chat_node (1200ms)
│  ├─ Intent detected: time_query
│  ├─ Tool call: get_time()
│  └─ LLM response
└─ Total: 1300ms, 45 tokens
```

---

## Part 6: Multi-Thread Isolation Test (5 minutes)

### 6.1 Create Thread A

1. In sidebar, click "New Chat"
2. **Query:** `My name is Alice`
3. Expected: Bot acknowledges

### 6.2 Create Thread B

1. In sidebar, click "New Chat" again
2. **Query:** `My name is Bob`
3. Expected: Bot acknowledges

### 6.3 Return to Thread A

1. Click back to first thread
2. **Query:** `Who am I?`
3. **Expected:** `You are Alice`

### 6.4 Return to Thread B

1. Click back to second thread
2. **Query:** `Who am I?`
3. **Expected:** `You are Bob`

**Checks:**
- ✅ Each thread remembers its own context
- ✅ No context bleeding between threads
- ✅ Both threads persist after switching

---

## Part 7: Download & Export Test (5 minutes)

### 7.1 Start a Conversation

**Queries:**
```
1. Hello
2. What is 2+2?
3. Tell me a joke
4. Goodbye
```

### 7.2 Export Chat

1. In sidebar, click "⬇️ Download chat as .md"
2. File should download

### 7.3 Verify Export

**Open the .md file and verify:**
- ✅ All 4 queries appear
- ✅ All 4 responses appear
- ✅ Format is readable Markdown
- ✅ Timestamps present (if applicable)
- ✅ No garbled text or encoding issues

**Expected format:**
```markdown
# Chat Export

## Message 1
**User:** Hello
**Bot:** Hello! How can I help?

## Message 2
**User:** What is 2+2?
**Bot:** 2+2 equals 4.
...
```

---

## Part 8: Error Handling & Graceful Degradation (10 minutes)

### 8.1 Test Missing API Keys

**Temporarily disable OpenWeather API:**
1. In `.env`, comment out `OPENWEATHER_API_KEY`
2. Restart chatbot

**Query:** `weather in Delhi`
**Expected:**
```
Weather API key not set. Add OPENWEATHER_API_KEY to your .env file.
```

**Check:**
- ✅ Clear error message (not crash)
- ✅ Bot suggests fix

**Re-enable the key for next tests**

### 8.2 Test API Timeout

**Query:** `stock price of invalidstock123` (forces API lookup)
**Expected:**
```
Could not fetch stock price for 'invalidstock123'.
```

**Check:**
- ✅ Timeout handled gracefully
- ✅ No app crash

### 8.3 Test PDF Parse Failure

1. Upload a corrupted/non-PDF file
2. **Expected:**
   - ✅ Error message in UI
   - ✅ App continues working
   - ✅ Can upload another file

### 8.4 Test Network Disconnection

1. Disable internet while chatbot running
2. **Query:** `Tell me about AI news`
3. **Expected:**
   - ✅ Connection error message (not crash)
   - ✅ Bot suggests retry

---

## Part 9: Performance & Load Test (10 minutes)

### 9.1 Measure Latency

**Query 1:** `What time is it?`
- Expected latency: < 500ms
- Why: No API calls, just system time

**Query 2:** `Weather in Delhi`
- Expected latency: 500-2000ms
- Why: API call + LLM response

**Query 3:** `Summarize the document` (after uploading PDF)
- Expected latency: 1000-3000ms
- Why: RAG retrieval + LLM synthesis

**Tool:** Check LangSmith or terminal logs for timestamps

**Checks:**
- ✅ All latencies under 5s
- ✅ No hanging requests

### 9.2 Stress Test: Rapid Queries

**Send 5 queries rapidly:**
```
1. hello
2. what time is it
3. weather in Mumbai
4. Tell me a joke
5. goodbye
```

**Expected:**
- ✅ All respond correctly
- ✅ No dropped messages
- ✅ No out-of-order responses

### 9.3 Memory Usage

**Monitor during test:**
- Terminal/Activity Monitor
- Expected: < 1GB RAM

---

## Part 10: Security Test (5 minutes)

### 10.1 Verify API Keys Not Exposed

**Check:**
- ✅ No API keys in terminal output
- ✅ No API keys in error messages
- ✅ LangSmith traces don't show raw keys

### 10.2 Verify .env in .gitignore

```bash
cat .gitignore | grep ".env"
```

**Expected:** Both `.env` files are ignored

```
Chatbot/.env
.env
```

**Check:**
- ✅ `.env` files not tracked in git
- ✅ `.gitignore` properly configured

### 10.3 Test Input Sanitization

**Query with special chars:**
```
<script>alert('xss')</script>
'; DROP TABLE users; --
../../../etc/passwd
```

**Expected:**
- ✅ Queries treated as literal text (no injection)
- ✅ No errors or unexpected behavior

---

## Test Checklist

Print this out and check off as you go:

```
[ ] Part 1: Smoke Test
  [ ] UI loads
  [ ] Greeting works
  [ ] Time query works

[ ] Part 2: External APIs
  [ ] Weather API (4 formats)
  [ ] Stock price API
  [ ] News/web search
  [ ] Error handling

[ ] Part 3: Memory
  [ ] STM works (in same chat)
  [ ] LTM works (across chats)
  [ ] New user greeting works

[ ] Part 4: RAG
  [ ] Document upload
  [ ] Semantic search
  [ ] Keyword search
  [ ] Hybrid search
  [ ] Metrics visible
  [ ] HITL triggers
  [ ] Query rewriting

[ ] Part 5: LangSmith
  [ ] Tracing enabled
  [ ] Traces visible in dashboard
  [ ] Latency tracked

[ ] Part 6: Multi-thread
  [ ] Thread A isolated
  [ ] Thread B isolated
  [ ] Switching preserves context

[ ] Part 7: Export
  [ ] Download works
  [ ] File format correct
  [ ] All messages included

[ ] Part 8: Error Handling
  [ ] Missing API key handled
  [ ] API timeout handled
  [ ] PDF parse error handled
  [ ] Network error handled

[ ] Part 9: Performance
  [ ] Time query < 500ms
  [ ] Weather query < 2000ms
  [ ] RAG query < 3000ms
  [ ] Rapid 5 queries work
  [ ] Memory < 1GB

[ ] Part 10: Security
  [ ] No API keys exposed
  [ ] .env in .gitignore
  [ ] Input sanitization works
```

---

## Debugging Tips

### Check Logs

**Terminal output:**
```bash
# Watch for errors
# Look for [RAG], [LLM], [TOOL] prefixes
# Check latencies
```

### LangSmith Debugging

**URL:** https://smith.langchain.com/projects/SafeAI%20Agent

**Each trace shows:**
- Full input/output
- Which nodes executed
- Token usage
- Latency breakdown
- Errors (if any)

### Local Testing

**If you don't have PostgreSQL:**
- LTM will be disabled (expected message)
- App still works with STM only
- Start with: `docker compose up -d`

---

## Automated Testing (Optional)

For CI/CD, run pytest:

```bash
pytest test_chatbot_e2e.py -v
```

See `test_chatbot_e2e.py` for unit/integration tests.

---

## Common Issues & Fixes

| Issue | Fix |
|---|---|
| "Connection refused" on load | Start PostgreSQL: `docker compose up -d` |
| Weather returns "City not found" | Check OPENWEATHER_API_KEY in .env |
| "No results found" for queries | Check intent detection / location extraction |
| LangSmith shows no traces | Set LANGSMITH_TRACING=true in .env |
| Document upload fails | Verify file is valid PDF, not corrupted |
| HITL button doesn't appear | Context must be < 200 chars for pause |
| Slow responses (> 5s) | Check internet connection, API status |

---

## Post-Test Checklist

After completing all tests:

- [ ] Document any bugs found
- [ ] Take screenshots of LangSmith dashboard
- [ ] Save export .md file as example
- [ ] Note latencies for each query type
- [ ] Update README if new issues found
- [ ] Commit .gitignore with .env protection
