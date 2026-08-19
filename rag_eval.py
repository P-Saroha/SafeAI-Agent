"""
rag_eval.py - Evaluate RAG retrieval performance on test queries.
"""

import json
from pathlib import Path
from chatbot_rag import get_rag_context, _safe_id


def evaluate():
    """Evaluate RAG on curated test queries."""
    
    # Find thread with FineTuningLLM.pdf
    knowledge_base = Path("knowledge_base")
    thread_id = None
    for folder in knowledge_base.iterdir():
        if folder.is_dir():
            pdfs = list(folder.glob("*.pdf"))
            for pdf in pdfs:
                if "FineTuningLLM" in pdf.name:
                    thread_id = folder.name
                    break
            if thread_id:
                break
    
    if not thread_id:
        print("ERROR: FineTuningLLM.pdf not found")
        return
    
    tid = _safe_id(thread_id)
    
    # Load test queries
    with open("test_queries.json") as f:
        tests = json.load(f)["queries"]
    
    print(f"\nEvaluating RAG on {len(tests)} test queries...")
    print("=" * 60)
    
    h1_total = h3_total = h5_total = 0
    mrr_total = 0
    
    for i, test in enumerate(tests, 1):
        query = test["query"]
        expected = test["expected"]
        
        context = get_rag_context(query, tid)
        if not context:
            continue
        
        # Parse chunks
        chunks = context.split("\n\n[")
        chunks = ["[" + c if j > 0 else c for j, c in enumerate(chunks)]
        
        # Rate each chunk
        scores = []
        for chunk in chunks:
            if chunk.startswith("[") and "]" in chunk:
                content = chunk.split("]", 1)[1]
                if "Sources:" not in content and "Powered" not in content:
                    content_lower = content.lower()
                    found = sum(1 for sec in expected if sec.lower() in content_lower)
                    score = 5 if found == len(expected) else (3 if found > 0 else 0)
                    scores.append(score)
        
        # Calculate metrics
        threshold = 3
        h1 = 1 if scores and scores[0] >= threshold else 0
        h3 = 1 if any(s >= threshold for s in scores[:3]) else 0
        h5 = 1 if any(s >= threshold for s in scores[:5]) else 0
        mrr = 0
        for pos, score in enumerate(scores, start=1):
            if score >= threshold:
                mrr = 1.0 / pos
                break
        
        h1_total += h1
        h3_total += h3
        h5_total += h5
        mrr_total += mrr
    
    n = len(tests)
    print(f"\nResults on {n} queries:")
    print("=" * 60)
    print(f"Hit Rate@1: {h1_total/n:.1%}")
    print(f"Hit Rate@3: {h3_total/n:.1%}")
    print(f"Hit Rate@5: {h5_total/n:.1%}")
    print(f"MRR:        {mrr_total/n:.3f}")
    print("=" * 60)
    
    # Save report
    report = {
        "queries_evaluated": n,
        "hit_rate_at_5": round(h5_total/n, 3),
        "mean_reciprocal_rank": round(mrr_total/n, 3),
        "verdict": "Evaluation complete"
    }
    
    with open("rag_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: rag_eval_report.json")


if __name__ == "__main__":
    evaluate()
