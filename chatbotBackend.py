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
conn = sqlite3.connect("chatbot_db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


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
# THREAD UTIL
# ==============================
def unique_thread_pointer():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)