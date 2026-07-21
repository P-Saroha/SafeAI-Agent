"""Run: python diag.py — shows all threads and their doc status."""
from chatbot_rag import has_documents, _get_supported_files, get_docs_dir
from pathlib import Path

kb = Path("knowledge_base")
print("=== knowledge_base folders ===")
for folder in sorted(kb.iterdir()) if kb.exists() else []:
    files = list(folder.rglob("*.*"))
    print(f"  {folder.name}: {[f.name for f in files]}")

print("\n=== has_documents check ===")
for folder in sorted(kb.iterdir()) if kb.exists() else []:
    tid = folder.name
    print(f"  {tid}: has_documents={has_documents(tid)}")
