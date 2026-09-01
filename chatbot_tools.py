"""
chatbot_tools.py
-----------------
All external tools the chatbot can use:
- Web search (DuckDuckGo)
- Weather (OpenWeather API)
- Stock price (Yahoo Finance)
- Current date/time (system clock)

Also includes simple helper functions to detect what kind of
question the user is asking.
"""

import json
import os
import re
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yfinance as yf
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load Chatbot/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── LLM setup ──────────────────────────────────────────────────────────────
# Groq-hosted LLM via OpenAI-compatible API
llm = ChatOpenAI(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
    max_tokens=1650,  # Optimized range: 1500-1800 for balance of speed and completeness
    streaming=True,
)

# ── Config ─────────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


# ══════════════════════════════════════════════════════════════════════════
# INTENT DETECTORS
# Simple keyword checks to know what the user is asking about
# ══════════════════════════════════════════════════════════════════════════

def is_greeting(query: str) -> bool:
    """Return True if the message is a short greeting like 'hi' or 'hello'."""
    q = query.lower().strip()
    greetings = {"hi", "hello", "hey", "hii", "yo", "namaste"}
    words = [w.strip("!.?,") for w in q.split()]
    return len(words) <= 4 and any(w in greetings for w in words)


def is_weather_query(query: str) -> bool:
    """Return True if the question is about weather or temperature."""
    q = query.lower()
    return any(word in q for word in ["weather", "temperature", "temp", "forecast"])


def is_time_query(query: str) -> bool:
    """Return True if the question is about the current date or time."""
    q = query.lower()
    # Avoid treating temperature questions as time questions
    if is_weather_query(q):
        return False
    return any(word in q for word in ["time", "date", "today date", "current time"])


def is_news_query(query: str) -> bool:
    """Return True if the question is asking for news or headlines."""
    q = query.lower()
    return any(word in q for word in ["news", "headline", "trending", "latest"])


def is_stock_query(query: str) -> bool:
    """Return True if the question is about a stock price."""
    q = query.lower()
    return "stock" in q or ("price" in q and "weather" not in q)


def is_github_query(query: str) -> bool:
    """Return True if the query contains a GitHub URL or asks about GitHub repos."""
    q = query.lower()
    # Check for github.com in URL
    if "github.com/" in q:
        return True
    # Check for GitHub keywords
    if any(keyword in q for keyword in ["github", "repository", "repo", "github repo"]):
        # Verify there's likely a URL present
        if "github.com" in query or re.search(r'https?://\S+', query):
            return True
    return False


def extract_weather_location(query: str) -> str:
    """Pull the city/location name out of a weather question."""
    q = query.lower().strip()
    
    # Remove common question prefixes
    q = re.sub(r"^(?:what|what's|whats|tell|show|give)\s+(?:is|are|me)?\s*", "", q)
    q = re.sub(r"^(?:today|today's|todays)\s+", "", q)
    q = re.sub(r"(?:weather|temperature|temp|forecast)\s+", "", q)
    
    # Pattern 1: "weather in/of/for CITY"
    match = re.search(r"(?:in|of|for)\s+([a-z\s]+?)(?:\s+(?:today|now|forecast|weather))?[\?.,]?$", q)
    if match:
        location = match.group(1).strip()
        if location and len(location) > 0:
            return location
    
    # Pattern 2: "CITY weather"
    match = re.search(r"^([a-z]+(?:\s+[a-z]+)?)\s+(?:weather|temperature|temp|forecast)", q)
    if match:
        location = match.group(1).strip()
        if location and len(location) > 0:
            return location
    
    # Pattern 3: Just city name (for queries like "delhi" or "chennai")
    # Extract longest continuous word sequence that's not a keyword
    keywords = {"weather", "temperature", "temp", "forecast", "today", "today's", "todays", "current", "what", "is"}
    words = q.split()
    for word in words:
        clean_word = word.strip("?.,!:;").lower()
        if clean_word and clean_word not in keywords and len(clean_word) > 2:
            return clean_word
    
    return ""


def extract_stock_symbol(query: str) -> str:
    """Map company name to stock ticker (ORCL, MSFT, TCS.NS, etc)."""
    q = query.lower()
    
    # Direct ticker mapping (US)
    us_map = {
        "oracle": "ORCL", "google": "GOOGL", "alphabet": "GOOGL",
        "microsoft": "MSFT", "apple": "AAPL", "amazon": "AMZN",
        "meta": "META", "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
        "intel": "INTC", "amd": "AMD", "ibm": "IBM", "cisco": "CSCO",
    }
    
    # Direct ticker mapping (India) — add .NS suffix for NSE
    india_map = {
        "tcs": "TCS.NS",
        "infosys": "INFY.NS",
        "wipro": "WIPRO.NS",
        "reliance": "RELIANCE.NS",
        "hdfc": "HDFC.NS",
        "hdfc bank": "HDFCBANK.NS",
        "icici": "ICICIBANK.NS",
        "icici bank": "ICICIBANK.NS",
        "lt": "LT.NS",
        "bajaj": "BAJAJFINSV.NS",
        "itc": "ITC.NS",
        "sbi": "SBIN.NS",
        "axis": "AXISBANK.NS",
        "kotak": "KOTAKBANK.NS",
        "mrf": "MRF.NS",
        "hero": "HEROMOTOCO.NS",
        "maruti": "MARUTISUZUKI.NS",
        "hul": "HINDUNILVR.NS",
        "colgate": "COLPAL.NS",
    }
    
    # Check US companies first
    for name, symbol in us_map.items():
        if name in q:
            return symbol
    
    # Check Indian companies
    for name, symbol in india_map.items():
        if name in q:
            return symbol
    
    # Try to find an explicit ticker (with or without suffix)
    # Match patterns like "AAPL" or "TCS.NS" or "BTC-USD"
    match = re.search(r"\b([A-Z]{1,5})(?:\.NS|\.BO|-USD)?\b", query)
    if match:
        ticker = match.group(1)
        # If no suffix, check if it's an Indian stock
        if ticker in {"TCS", "INFY", "WIPRO", "RELIANCE", "HDFC", "ICICI", "LT", "BAJAJ", "ITC", "SBIN", "AXIS", "KOTAK", "MRF", "HERO", "MARUTI"}:
            return f"{ticker}.NS"
        return ticker
    
    return ""


# ══════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# Each function calls an external service and returns a plain string result
# ══════════════════════════════════════════════════════════════════════════

def call_search(query: str) -> str:
    """Search DuckDuckGo and return structured results as a list of dicts."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return results  # Return raw list — formatting is done in format_search_response
    except Exception as e:
        return []


def format_search_response(results, query: str = "") -> str:
    """
    Take raw DuckDuckGo results and use the LLM to produce a clean,
    well-formatted summary with numbered points and source links.

    Why use the LLM here?
    Raw search results are messy — truncated sentences, duplicate info,
    weird URLs. The LLM cleans them up into a readable answer.
    """
    if not results:
        return "No results found for your query."

    # Build a compact text block from the raw results to feed to the LLM
    raw_lines = []
    sources = []
    for r in results:
        title = str(r.get("title", "")).strip()
        body  = str(r.get("body",  "")).strip()[:300]
        url   = str(r.get("href",  "")).strip()
        if title:
            raw_lines.append(f"- {title}: {body}")
        if url:
            sources.append(url)

    if not raw_lines:
        return "No results found for your query."

    raw_text = "\n".join(raw_lines)

    # Ask the LLM to summarize into a clean response
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        prompt_system = (
            "You are a helpful assistant summarizing web search results.\n"
            "Format your response like this:\n\n"
            "**[Topic]**\n\n"
            "1. **[Title]** — one clear sentence summary.\n"
            "2. **[Title]** — one clear sentence summary.\n"
            "3. **[Title]** — one clear sentence summary.\n\n"
            "Rules:\n"
            "- Use the exact titles from the results as bold headers.\n"
            "- Write one clean sentence per result — no raw URLs in the text.\n"
            "- Keep it factual and concise.\n"
            "- Do NOT include URLs in the numbered list.\n"
            "- Do NOT add any intro like 'Here are the results'."
        )
        prompt_user = (
            f"Search query: {query}\n\n"
            f"Search results:\n{raw_text}"
        )
        response = llm.invoke([
            SystemMessage(content=prompt_system),
            HumanMessage(content=prompt_user),
        ])
        summary = str(response.content).strip()
    except Exception:
        # Fallback: plain numbered list if LLM fails
        summary = "\n".join(
            f"{i}. {line.lstrip('- ')}"
            for i, line in enumerate(raw_lines, 1)
        )

    # Append clean source links at the bottom
    if sources:
        source_lines = "\n".join(f"- {url}" for url in sources[:5])
        return f"{summary}\n\n**Sources:**\n{source_lines}"

    return summary


def call_weather(location: str) -> str:
    """
    Fetch current weather for a city from OpenWeather API.
    Returns a JSON string with temp, humidity, wind, etc.
    Returns an error string if the API key is missing or the city is not found.
    """
    if not OPENWEATHER_API_KEY:
        return "Weather API key not set. Add OPENWEATHER_API_KEY to your .env file."
    if not location.strip():
        return "No location provided."

    params = urlencode({"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"})
    url = f"{OPENWEATHER_URL}?{params}"
    try:
        with urlopen(Request(url, headers={"User-Agent": "Chatbot/1.0"}), timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"Weather API error: {e}"

    if str(data.get("cod")) != "200":
        return f"City not found: {data.get('message', 'unknown error')}"

    weather = (data.get("weather") or [{}])[0]
    main = data.get("main") or {}
    wind = data.get("wind") or {}

    return json.dumps({
        "location": data.get("name", location),
        "description": weather.get("description", ""),
        "temp_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "wind_mps": wind.get("speed"),
    })


def call_stock(symbol: str) -> str:
    """Fetch the latest closing stock price for a ticker symbol.
    
    Supports:
    - US stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, ORCL
    - Indian stocks: TCS, INFY, WIPRO, RELIANCE, HDFC, ICICI (add .NS or .BO suffix)
    - Cryptocurrencies: BTC-USD, ETH-USD
    """
    if not symbol or not symbol.strip():
        return "Please provide a stock symbol (e.g., 'AAPL', 'TCS.NS', 'BTC-USD')."
    
    symbol = symbol.strip().upper()
    
    try:
        # For Indian stocks without suffix, add .NS (NSE — National Stock Exchange)
        if symbol and not any(suffix in symbol for suffix in [".NS", ".BO", "-USD"]):
            # Check if it's likely an Indian stock
            indian_stocks = {"TCS", "INFY", "WIPRO", "RELIANCE", "HDFC", "ICICI", "LT", "BAJAJ", "ITC"}
            if symbol in indian_stocks:
                symbol = f"{symbol}.NS"
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        
        # Get history — yfinance might return empty if ticker is invalid
        history = ticker.history(period="1d")
        
        if history.empty or "Close" not in history.columns:
            print(f"[Stock] No data returned for {symbol} — ticker may be invalid")
            return f"Could not fetch stock price for '{symbol}'. Please check the ticker symbol.\nTip: Use .NS for Indian NSE stocks (e.g., 'TCS.NS')"
        
        price = history["Close"].iloc[-1]
        
        if price is None or price == 0:
            print(f"[Stock] Invalid price data for {symbol}: {price}")
            return f"Could not fetch valid price for '{symbol}'."
        
        # Get additional info
        try:
            info = ticker.info
            currency = info.get("currency", "USD")
            if currency == "INR":
                currency_symbol = "₹"
            else:
                currency_symbol = "$" if currency == "USD" else currency
        except:
            currency_symbol = "$"
        
        price_formatted = round(price, 2)
        print(f"[Stock] Fetched {symbol}: {price_formatted} {currency_symbol}")
        return f"{symbol}: {currency_symbol}{price_formatted}"
        
    except Exception as e:
        error_msg = str(e)
        print(f"[Stock] Error fetching {symbol}: {error_msg}")
        
        # Provide helpful error messages
        if "No data found" in error_msg or "no data" in error_msg.lower():
            return f"Ticker '{symbol}' not found. Try:\n- US: AAPL, MSFT, GOOGL, AMZN, TSLA\n- India: TCS.NS, INFY.NS, WIPRO.NS\n- Crypto: BTC-USD, ETH-USD"
        
        return f"Error fetching '{symbol}': {error_msg}. Please verify the ticker symbol."


def call_datetime() -> str:
    """Return the current local date and time as a readable string."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def format_weather_response(raw_json: str, location: str) -> str:
    """Format weather JSON as markdown table, or fallback to web search."""
    try:
        data = json.loads(raw_json)
    except Exception:
        data = {}

    if not data or "temp_c" not in data:
        # Fallback: search the web for weather info
        results = call_search(f"weather in {location}")
        summary = format_search_response(results, f"weather in {location}")
        return f"### Weather for {location}\n\n{summary}"

    return (
        f"### Weather for {data.get('location', location)}\n\n"
        f"| Detail | Value |\n"
        f"|---|---|\n"
        f"| Condition | {data.get('description', 'N/A').capitalize()} |\n"
        f"| Temperature | {data.get('temp_c')} °C |\n"
        f"| Feels like | {data.get('feels_like_c')} °C |\n"
        f"| Humidity | {data.get('humidity')}% |\n"
        f"| Wind speed | {data.get('wind_mps')} m/s |\n\n"
        f"**Sources:** https://openweathermap.org/"
    )
