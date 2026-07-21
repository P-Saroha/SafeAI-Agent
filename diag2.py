"""Run: python diag2.py"""
import sqlite3
from chatbot_rag import has_documents

conn = sqlite3.connect("chatbot_db")
cur = conn.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

all_threads = set()
for t in tables:
    try:
        cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{t}")').fetchall()]
        if "thread_id" in cols:
            rows = cur.execute(f'SELECT DISTINCT thread_id FROM "{t}"').fetchall()
            for r in rows:
                all_threads.add(r[0])
    except Exception as e:
        print(f"  Error reading {t}: {e}")

print(f"\nAll thread IDs in SQLite ({len(all_threads)} total):")
for tid in sorted(all_threads):
    print(f"  {tid}  → has_documents={has_documents(tid)}")
