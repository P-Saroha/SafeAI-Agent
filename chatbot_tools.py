from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from ddgs import DDGS
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import yfinance as yf

load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    streaming=True,
)

router_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    streaming=False,
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def _needs_external_tools(query: str) -> bool:
    """Heuristic gate to avoid tool calls for simple knowledge questions."""
    q = query.lower().strip()
    if not q:
        return False

    if _is_simple_question(q):
        return False

    tool_markers = [
        "latest", "today", "current", "news", "headline", "price", "stock",
        "weather", "score", "live", "market", "exchange rate", "time now",
    ]
    return any(marker in q for marker in tool_markers)


def _is_simple_question(query: str) -> bool:
    """Return True for definition/explanation-style questions that the LLM can answer directly."""
    q = (query or "").strip().lower()
    if not q:
        return False

    if re.match(r"^(what is|what's|who is|who's|define|definition of|meaning of|explain)\b", q):
        return True
    if "purpose of" in q or "difference between" in q:
        return True

    return False


def _route_tool_with_llm(query: str) -> str:
    """LLM-based fallback router. Returns: weather|time|news|stock|search|none."""
    q = (query or "").strip()
    if not q:
        return "none"

    if _is_simple_question(q):
        return "none"

    prompt = (
        "You are a tool router. Choose the best tool for the user query.\n"
        "Return ONLY JSON in the form: {\"tool\": \"weather|time|news|stock|search|none\"}.\n"
        "Guidelines:\n"
        "- weather: weather, temperature, forecast, AQI, or location-based weather\n"
        "- time: date/time now or today date\n"
        "- news: latest news, headlines, trending\n"
        "- stock: stock price, ticker, market price\n"
        "- search: current facts not covered above\n"
        "- none: general knowledge, definitions, or chit-chat"
    )

    try:
        result = router_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=q)])
        raw = str(getattr(result, "content", "") or "").strip()
        data = json.loads(raw)
        tool = str(data.get("tool", "none")).strip().lower()
    except Exception:
        raw = str(raw) if "raw" in locals() else ""
        match = re.search(r"tool\s*[:=]\s*\"?(\w+)\"?", raw, re.IGNORECASE)
        tool = match.group(1).strip().lower() if match else "none"

    allowed = {"weather", "time", "news", "stock", "search", "none"}
    return tool if tool in allowed else "none"


def _llm_needs_tool(query: str) -> bool:
    """LLM gate to decide if a query needs external tools (real-time or unknown facts)."""
    q = (query or "").strip()
    if not q or _is_simple_question(q):
        return False

    prompt = (
        "You decide if a query needs external tools.\n"
        "Return ONLY JSON: {\"needs_tool\": true|false}.\n"
        "Use tools for real-time info (today, weather, news, stocks, live data) or unknown current facts.\n"
        "Do NOT use tools for definitions, explanations, or general knowledge."
    )
    try:
        result = router_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=q)])
        raw = str(getattr(result, "content", "") or "").strip()
        data = json.loads(raw)
        return bool(data.get("needs_tool"))
    except Exception:
        return _needs_external_tools(q)


def _is_news_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ["news", "headline", "trending", "top trending", "latest"])


def _is_time_query(query: str) -> bool:
    q = query.lower()
    if not q:
        return False
    if _is_weather_query(q) or "temperature" in q or "temp" in q:
        return False
    return any(word in q for word in ["time", "date", "today date", "current time"]) or q.strip() in {"what is today", "today date"}


def _is_greeting(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    if len(q.split()) > 4:
        return False
    greetings = {"hi", "hello", "hey", "hii", "hllo", "yo", "namaste"}
    tokens = [t.strip("!.?,") for t in q.split()]
    return any(t in greetings for t in tokens)


def _is_stock_query(query: str) -> bool:
    q = query.lower()
    return "stock" in q or "price" in q


def _is_weather_query(query: str) -> bool:
    q = query.lower()
    return (
        "weather" in q
        or "weaher" in q
        or "weather" in q
        or "wheather" in q
        or "temperature" in q
        or "temp" in q
    )


def _has_location_hint(query: str) -> bool:
    q = query.lower()
    if " in " in q or " at " in q or " of " in q:
        return True
    location = _extract_weather_location(query)
    return bool(location and location.strip())


def _extract_weather_location(query: str) -> str:
    q = query.lower()
    match = re.search(r"weather\s+(?:in|of|for)\s+([a-z\s]+?)(?:\s+(?:today|tomorrow|yesterday|now|forecast))?[\?\.,]?$", q)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?:^|\s)([a-z]+(?:\s+[a-z]+)?)\s+weather", q)
    if match:
        location = match.group(1).strip()
        time_words = {"today", "tomorrow", "yesterday", "now", "tonight"}
        words = location.split()
        filtered = [w for w in words if w not in time_words]
        return " ".join(filtered) if filtered else location
    return ""


def _parse_weather_payload(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    if "temp_c" not in data:
        return {}
    return data


def _truncate_text(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _extract_first_url(text: str) -> str:
    match = re.search(r"https?://\S+", text or "")
    if not match:
        return ""
    return match.group(0).rstrip(")].,;!")


def _collapse_duplicate_phrase(text: str, phrase: str) -> str:
    if not text or not phrase:
        return text
    doubled = f"{phrase}{phrase}"
    if doubled in text:
        return text.replace(doubled, phrase)
    return text


def _extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)]+", text or "")
    seen = set()
    ordered = []
    for url in urls:
        cleaned = url.rstrip("].,;!")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _format_sources(urls: list[str], fallback: str = "Internal knowledge") -> str:
    if not urls:
        return f"Sources:\n- {fallback}"
    lines = ["Sources:"]
    for url in urls:
        lines.append(f"- {url}")
    return "\n".join(lines)


def _format_search_results(raw_text: str, max_items: int = 5) -> str:
    lines = [l.strip() for l in (raw_text or "").splitlines() if l.strip()]
    entries = []
    sources = []
    for line in lines[:max_items]:
        url_match = re.search(r"\((https?://[^)]+)\)$", line)
        url = url_match.group(1) if url_match else ""
        if url:
            sources.append(url)
            line = line[: line.rfind("(")].strip()
        parts = [p.strip() for p in line.split(" - ") if p.strip()]
        title = parts[0] if parts else "Result"
        snippet = _truncate_text(" ".join(parts[1:]), 160) if len(parts) > 1 else ""
        domain = ""
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc or ""
        summary = f"{title}"
        if snippet:
            summary += f" — {snippet}"
        if domain:
            summary += f" (Source: {domain})"
        entries.append(summary)
    if not entries:
        entries.append("No results found.")
    numbered = [f"{idx}) {text}" for idx, text in enumerate(entries, start=1)]
    return "\n".join(numbered) + "\n\n" + _format_sources(sources, fallback="Web search")


def _extract_stock_symbol(query: str) -> str:
    q = query.lower()
    mapping = {
        "oracle": "ORCL",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "microsoft": "MSFT",
        "apple": "AAPL",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        "tesla": "TSLA",
        "nvidia": "NVDA",
    }
    for name, symbol in mapping.items():
        if name in q:
            return symbol

    match = re.search(r"\b([A-Z]{1,5})\b", query)
    if match:
        return match.group(1).upper()
    return ""


@tool
def search_tool(query: str) -> str:
    """Search latest information from web"""
    search_query = query
    if _is_weather_query(query):
        location = _extract_weather_location(query)
        if location:
            search_query = f"{location} weather today"

    with DDGS() as ddgs:
        results = list(ddgs.text(search_query, max_results=10))

    snippets = []
    seen = set()
    for r in results:
        body_raw = str(r.get("body", "")).strip()
        body_raw = re.sub(r"\s+", " ", body_raw)
        body = _truncate_text(body_raw, 200)

        title_raw = str(r.get("title", "")).strip()
        title_raw = re.sub(r"\s+", " ", title_raw)
        title = _truncate_text(title_raw, 120)

        url = str(r.get("href", "")).strip()
        if _is_weather_query(query):
            text_blob = f"{title} {body}".lower()
            if "weather" not in text_blob:
                continue

        dedup_key = re.sub(r"\s+", " ", f"{title} {body}".lower())[:100]
        if not title or dedup_key in seen:
            continue

        seen.add(dedup_key)
        text = " - ".join(part for part in [title, body] if part)
        if url:
            text = f"{text} ({url})"
        snippets.append(text)

        if len(snippets) >= 5:
            break

    return "\n".join(snippets) if snippets else "No results found."


@tool
def calculator(expression: str) -> str:
    """Solve math expressions"""
    try:
        return str(eval(expression, {"__builtins__": None}, vars(math)))
    except Exception:
        return "Error in calculation"


@tool
def get_stock_price(symbol: str) -> str:
    """Get stock price"""
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"{symbol} price is {round(price, 2)} USD"
    except Exception:
        return "Stock not found"


@tool
def get_current_date_time() -> str:
    """Get current local date and time."""
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location using OpenWeather."""
    if not OPENWEATHER_API_KEY:
        return "Weather API key not configured."
    loc = (location or "").strip()
    if not loc:
        return "Weather location not provided."

    params = urlencode({"q": loc, "appid": OPENWEATHER_API_KEY, "units": "metric"})
    url = f"{OPENWEATHER_BASE_URL}?{params}"
    req = Request(url, headers={"User-Agent": "Chatbot/1.0"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as err:
        return f"Weather API error: {err}"
    except Exception as err:
        return f"Weather API error: {err}"

    if not isinstance(data, dict) or str(data.get("cod")) not in {"200", "200.0"}:
        message = data.get("message") if isinstance(data, dict) else "unknown error"
        return f"Weather not found: {message}"

    weather = (data.get("weather") or [{}])[0]
    main = data.get("main") or {}
    wind = data.get("wind") or {}

    payload = {
        "location": data.get("name") or loc,
        "description": weather.get("description") or "",
        "temp_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "wind_mps": wind.get("speed"),
    }
    return json.dumps(payload)


def _call_search_tool(query: str) -> str:
    return str(search_tool.invoke({"query": query}))


def _call_stock_tool(symbol: str) -> str:
    return str(get_stock_price.invoke({"symbol": symbol}))


def _call_time_tool() -> str:
    return str(get_current_date_time.invoke({}))


def _call_weather_tool(location: str) -> str:
    return str(get_weather.invoke({"location": location}))
