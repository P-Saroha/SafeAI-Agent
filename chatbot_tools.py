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
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())

# ── LLM setup ──────────────────────────────────────────────────────────────
# Main LLM used for generating responses
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
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


def extract_weather_location(query: str) -> str:
    """Pull the city/location name out of a weather question."""
    q = query.lower()
    # Match patterns like "weather in Delhi" or "weather of Mumbai"
    match = re.search(r"weather\s+(?:in|of|for)\s+([a-z\s]+?)(?:\s+(?:today|now|forecast))?[\?.,]?$", q)
    if match:
        return match.group(1).strip()
    # Match patterns like "Delhi weather"
    match = re.search(r"([a-z]+(?:\s+[a-z]+)?)\s+weather", q)
    if match:
        return match.group(1).strip()
    return ""


def extract_stock_symbol(query: str) -> str:
    """Map a company name or ticker from the query to a stock symbol."""
    q = query.lower()
    name_map = {
        "oracle": "ORCL", "google": "GOOGL", "alphabet": "GOOGL",
        "microsoft": "MSFT", "apple": "AAPL", "amazon": "AMZN",
        "meta": "META", "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
    }
    for name, symbol in name_map.items():
        if name in q:
            return symbol
    # Try to find an uppercase ticker like ORCL or TSLA
    match = re.search(r"\b([A-Z]{1,5})\b", query)
    if match:
        return match.group(1)
    return ""


# ══════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# Each function calls an external service and returns a plain string result
# ══════════════════════════════════════════════════════════════════════════

def call_search(query: str) -> str:
    """Search DuckDuckGo and return a short text summary of results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        lines = []
        for r in results:
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()[:200]
            url = r.get("href", "").strip()
            if title:
                lines.append(f"{title} — {body} ({url})")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search error: {e}"


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
    """Fetch the latest closing stock price for a ticker symbol."""
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.history(period="1d")["Close"].iloc[-1]
        return f"{symbol} price is {round(price, 2)} USD"
    except Exception:
        return f"Could not fetch stock price for '{symbol}'."


def call_datetime() -> str:
    """Return the current local date and time as a readable string."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def format_weather_response(raw_json: str, location: str) -> str:
    """
    Turn the raw weather JSON string into a nicely formatted message.
    Falls back to a web search if the JSON is empty or invalid.
    """
    try:
        data = json.loads(raw_json)
    except Exception:
        data = {}

    if not data or "temp_c" not in data:
        # Fallback: search the web for weather info
        search_result = call_search(f"weather in {location}")
        return f"Weather (via web search):\n{search_result}\n\nSources:\n- https://duckduckgo.com/"

    return (
        f"Weather for {data.get('location', location)}:\n"
        f"- Condition : {data.get('description', 'N/A')}\n"
        f"- Temperature: {data.get('temp_c')} °C\n"
        f"- Feels like : {data.get('feels_like_c')} °C\n"
        f"- Humidity   : {data.get('humidity')}%\n"
        f"- Wind speed : {data.get('wind_mps')} m/s\n\n"
        f"Sources:\n- https://openweathermap.org/"
    )


def format_search_response(raw: str) -> str:
    """Wrap search results in a clean 'Top results:' block."""
    if not raw or raw.strip() == "No results found.":
        return "No results found."
    return f"Top results:\n{raw}"
