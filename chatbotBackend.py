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
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except Exception:
    FAISS = None
    PyPDFLoader = None
    TextLoader = None

import sqlite3
import os
import math
import json
import hashlib
import numpy as np
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
    mode: str


# ==============================
# RAG CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RAG_DOCS_DIR = BASE_DIR / "knowledge_base"
RAG_INDEX_DIR = BASE_DIR / "faiss_index"

rag_retriever = None
rag_status = "RAG not initialized"


class HashEmbeddings(Embeddings):
    """Deterministic local embeddings fallback for FAISS without external API calls."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tolist()

        for token in tokens:
            vec[self._hash_token(token)] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _embedding_from_backend(backend: str):
    """Create embedding model instance from backend selector."""
    if backend == "google":
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model_name), {"backend": "google", "model": model_name}

    dim = int(os.getenv("RAG_HASH_EMBEDDING_DIM", "384"))
    return HashEmbeddings(dim=dim), {"backend": "hash", "dim": dim}


def _save_rag_meta(meta: dict) -> None:
    try:
        RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(RAG_INDEX_DIR / "meta.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)
    except Exception as err:
        print(f"RAG meta save warning: {err}")


def _load_rag_meta() -> dict:
    meta_path = RAG_INDEX_DIR / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _collect_rag_files() -> list[Path]:
    """Collect supported knowledge files for indexing."""
    supported = {".txt", ".md", ".pdf"}
    files: list[Path] = []

    if RAG_DOCS_DIR.exists():
        files.extend([p for p in RAG_DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in supported])

    default_candidates = [
        BASE_DIR / "ChatResumeWorkFlow.txt",
        PROJECT_ROOT / "README.md",
    ]

    for candidate in default_candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in supported:
            files.append(candidate)

    # Deduplicate while preserving order
    unique_files = list(dict.fromkeys(files))
    return sorted(unique_files)


def _load_documents(paths: list[Path]) -> list[Document]:
    """Load text and PDF files into LangChain documents."""
    documents: list[Document] = []

    for path in paths:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                if PyPDFLoader is None:
                    continue
                loader = PyPDFLoader(str(path))
                documents.extend(loader.load())
            else:
                if TextLoader is None:
                    continue
                loader = TextLoader(str(path), encoding="utf-8")
                documents.extend(loader.load())
        except Exception as err:
            print(f"RAG load warning for {path.name}: {err}")

    return documents


def _build_rag_retriever(force_rebuild: bool = False):
    """Build or load FAISS retriever from local knowledge base files."""
    global rag_status

    if FAISS is None:
        rag_status = "RAG unavailable: install faiss-cpu and document loaders"
        return None

    files = _collect_rag_files()
    if not files:
        rag_status = "RAG knowledge base is empty"
        return None

    preferred_backend = os.getenv("RAG_EMBEDDING_BACKEND", "hash").lower()

    try:
        if RAG_INDEX_DIR.exists() and not force_rebuild:
            meta = _load_rag_meta()
            load_backend = meta.get("backend", preferred_backend)
            embeddings, _ = _embedding_from_backend(load_backend)
            vectorstore = FAISS.load_local(
                str(RAG_INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            rag_status = f"RAG ready (loaded index, {len(files)} files, backend={load_backend})"
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as err:
        print(f"RAG index load warning, rebuilding: {err}")

    docs = _load_documents(files)
    if not docs:
        rag_status = "RAG could not load readable documents"
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        rag_status = "RAG found no chunks after splitting"
        return None

    embeddings, meta = _embedding_from_backend(preferred_backend)

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as err:
        rag_status = f"RAG embedding/index build failed: {err}"
        return None

    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(RAG_INDEX_DIR))
    meta.update({"files_indexed": len(files), "chunks": len(chunks)})
    _save_rag_meta(meta)

    rag_status = (
        f"RAG ready (built index with {len(chunks)} chunks from {len(files)} files, "
        f"backend={meta.get('backend')})"
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def ensure_rag_ready(force_rebuild: bool = False):
    """Lazy initialize retriever so startup remains fast."""
    global rag_retriever

    if rag_retriever is None or force_rebuild:
        rag_retriever = _build_rag_retriever(force_rebuild=force_rebuild)

    return rag_retriever


def rebuild_rag_index() -> str:
    """Rebuild FAISS index from files in Chatbot/knowledge_base."""
    retriever = ensure_rag_ready(force_rebuild=True)
    if retriever is None:
        return f"RAG rebuild failed: {rag_status}"
    return f"RAG rebuild successful: {rag_status}"


def _latest_user_query(messages: list[BaseMessage]) -> str:
    """Return the most recent human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def _rag_context_for_query(query: str, k: int = 4) -> str:
    """Fetch compact RAG context for prompt grounding."""
    if not query.strip():
        return ""

    retriever = ensure_rag_ready()
    if retriever is None:
        return ""

    try:
        docs = retriever.invoke(query)
    except Exception:
        return ""

    if not docs:
        return ""

    snippets = []
    for i, doc in enumerate(docs[:k], start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        page_info = f" (page {page + 1})" if isinstance(page, int) else ""
        chunk = doc.page_content.strip().replace("\n", " ")
        snippets.append(f"[{i}] {source}{page_info}: {chunk}")

    return "\n\n".join(snippets)


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


@tool
def rag_search(question: str) -> str:
    """Search local knowledge base documents for RAG context."""
    retriever = ensure_rag_ready()

    if retriever is None:
        return (
            "RAG is not ready. "
            f"{rag_status}. "
            f"Add .txt/.md/.pdf files in {RAG_DOCS_DIR} and rebuild index."
        )

    try:
        docs = retriever.invoke(question)
    except Exception as err:
        return f"RAG retrieval error: {err}"

    if not docs:
        return "No relevant context found in local knowledge base."

    contexts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_info = f" (page {page + 1})" if isinstance(page, int) else ""
        chunk = doc.page_content.strip().replace("\n", " ")
        contexts.append(f"[{i}] {Path(source).name}{page_info}: {chunk}")

    return "\n\n".join(contexts)


tools = [search_tool, calculator, get_stock_price, rag_search]
agent_tools = [search_tool, calculator, get_stock_price]


# ==============================
# LLM WITH TOOLS
# ==============================
llm_with_tools = llm.bind_tools(tools)
llm_with_agent_tools = llm.bind_tools(agent_tools)


# ==============================
# CHAT NODE
# ==============================
def chat_node(state: ChatState):
    mode = str(state.get("mode", "auto")).lower()
    if mode not in {"auto", "hybrid", "agent_only", "rag_only"}:
        mode = "auto"

    if mode == "auto":
        mode = "rag_only" if len(_collect_rag_files()) > 0 else "agent_only"

    latest_query = _latest_user_query(state["messages"])
    rag_context = _rag_context_for_query(latest_query) if mode in {"hybrid", "rag_only"} else ""

    rag_instructions = (
        f"RAG Context (highest priority if relevant):\n{rag_context}\n\n"
        "If RAG context clearly answers the question, answer from it and cite source tags like [1], [2]. "
        "If context is not relevant, then use tools as needed."
    ) if rag_context else "No RAG context available for this query."

    system_prompt = f"""
You are a smart AI assistant.

Rules:
- Current response mode: {mode}
- If mode is rag_only: answer only from RAG context and clearly say when answer is not found in context
- If mode is agent_only: use tools for web/stock/math and do not rely on RAG context
- If mode is hybrid: prefer RAG context when relevant, otherwise use tools
- Keep responses concise and useful

{rag_instructions}
"""

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    if mode == "rag_only":
        response = llm.invoke(messages)
    elif mode == "agent_only":
        response = llm_with_agent_tools.invoke(messages)
    else:
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