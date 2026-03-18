# ==============================
# IMPORTS
# ==============================
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv

import sqlite3
import os
import math
from ddgs import DDGS
import yfinance as yf


# ==============================
# ENV + LLM
# ==============================
load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # change if needed
    temperature=0,
    streaming=True
)


# ==============================
# STATE
# ==============================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==============================
# TOOLS
# ==============================

@tool
def search_tool(query: str) -> str:
    """Search latest information from web"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return "\n".join([r["body"] for r in results])


@tool
def calculator(expression: str) -> str:
    """Solve math expressions"""
    try:
        return str(eval(expression, {"__builtins__": None}, vars(math)))
    except:
        return "Error in calculation"


@tool
def get_stock_price(symbol: str) -> str:
    """Get stock price"""
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"{symbol} price is {round(price,2)} USD"
    except:
        return "Stock not found"


tools = [search_tool, calculator, get_stock_price]


# ==============================
# LLM WITH TOOLS
# ==============================
llm_with_tools = llm.bind_tools(tools)


# ==============================
# CHAT NODE
# ==============================
def chat_node(state: ChatState):
    messages = [
        SystemMessage(content="""
You are a smart AI assistant.

Rules:
- Use calculator for math
- Use search tool for latest info
- Use stock tool for stock prices
- Avoid unnecessary tool usage
""")
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ==============================
# DATABASE (MEMORY)
# ==============================
def init_checkpointer():
    """Initialize checkpointer with automatic error recovery."""
    db_path = "chatbot_db"
    
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("SELECT 1")  # Test connection
        checkpointer = SqliteSaver(conn=conn)
        return checkpointer, conn
    except Exception as e:
        print(f"Checkpoint error detected: {e}. Recovering...")
        
        # Backup corrupted database
        if os.path.exists(db_path):
            backup_path = f"{db_path}.backup_{os.getpid()}"
            try:
                os.rename(db_path, backup_path)
                print(f"Backed up corrupted database to {backup_path}")
            except Exception as backup_err:
                print(f"Backup failed: {backup_err}")
        
        # Create fresh database
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        print("Fresh checkpoint database created")
        return checkpointer, conn

checkpointer, conn = init_checkpointer()


# ==============================
# GRAPH (AGENT)
# ==============================
builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chat_node")

builder.add_conditional_edges(
    "chat_node",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

builder.add_edge("tools", "chat_node")

chatbot = builder.compile(checkpointer=checkpointer)


# ==============================
# CHAT TITLE GENERATION
# ==============================
def generate_chat_title(user_message: str) -> str:
    """Generate a concise chat title from the first user message."""
    if not user_message or not user_message.strip():
        return "New Chat"
    
    msg = user_message.strip()[:200]  # Limit to first 200 chars
    
    try:
        # Try LLM-based title generation
        prompt = f"""Given this user message, generate a very short chat title (2-5 words max, no punctuation). 
Only return the title, nothing else.

User message: {msg}

Title:"""
        title = llm.invoke(prompt).content.strip()
        title = title.strip('"\'.!?,;:').strip()
        if title and len(title) > 0:
            return title[:50]
    except Exception as e:
        print(f"Title generation error (using fallback): {str(e)[:50]}")
    
    # Fallback: Use first 4-5 words from message
    words = msg.split()
    if len(words) > 0:
        fallback_title = ' '.join(words[:5])
        if len(fallback_title) > 50:
            fallback_title = fallback_title[:47] + "..."
        return fallback_title
    
    return "New Chat"


# ==============================
# THREAD UTIL
# ==============================
def unique_thread_pointer():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)