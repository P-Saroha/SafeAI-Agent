"""Clean RAG evaluation script - test Q&A against FineTuningLLM.pdf"""

import json
from chatbot_rag import get_rag_context

# Load Q&A dataset
with open('real_qa_pairs_from_pdfs.json') as f:
    qa_data = json.load(f)

# Thread ID for FineTuningLLM.pdf
FINETUNING_THREAD_ID = "b531664a-6bf8-41de-b966-688d3cc71f03"


def calculate_hit(expected_answer, rag_response):
    """Check if RAG response contains expected content (50% word match threshold)"""
    if not rag_response:
        return False
    
    exp_lower = expected_answer.lower()
    rag_lower = rag_response.lower()
    
    # Extract words >= 4 chars (meaningful keywords)
    words = [w for w in exp_lower.split() if len(w) >= 4 and w.isalpha()]
    if not words:
        return False
    
    found_count = sum(1 for w in words if w in rag_lower)
    return (found_count / len(words)) >= 0.5


def test_questions(qa_list, source_name, thread_id):
    """Test Q&A list and return metrics"""
    print(f"\n{'='*70}")
    print(f"Testing: {source_name}")
    print(f"{'='*70}")
    
    hits = 0
    mrr_sum = 0.0
    
    for idx, qa in enumerate(qa_list, 1):
        question = qa['question']
        expected = qa['expected_answer']
        
        try:
            # Get RAG response using thread ID
            rag_resp = get_rag_context(question, thread_id)
            
            # Evaluate hit
            is_hit = calculate_hit(expected, rag_resp)
            
            if is_hit:
                hits += 1
                mrr_sum += 1.0
                status = "HIT"
            else:
                mrr_sum += 0.0
                status = "MISS"
            
            # Display result
            q_display = question[:50] if len(question) <= 50 else question[:47] + "..."
            print(f"[{idx:2d}] {q_display:<50} => {status}")
            
        except Exception as e:
            print(f"[{idx:2d}] ERROR: {str(e)[:40]}")
    
    # Calculate metrics
    n = len(qa_list)
    hit_rate = hits / n if n > 0 else 0
    mrr = mrr_sum / n if n > 0 else 0
    
    print(f"{'-'*70}")
    print(f"Results: {hits}/{n} hits ({hit_rate:.1%}) | MRR: {mrr:.3f}")
    
    return {"hits": hits, "mrr": mrr, "hit_rate": hit_rate}


# Test FineTuningLLM questions
results = {}
if 'finetuning_llm' in qa_data and qa_data['finetuning_llm']:
    results['finetuning_llm'] = test_questions(
        qa_data['finetuning_llm'], 
        "FineTuningLLM.pdf",
        FINETUNING_THREAD_ID
    )

# Calculate totals
total_q = sum(len(qa_data[k]) for k in results.keys())
total_hits = sum(r['hits'] for r in results.values())
total_mrr = sum(r['mrr'] * len(qa_data[k]) for k, r in results.items()) / total_q if total_q > 0 else 0
total_hit_rate = total_hits / total_q if total_q > 0 else 0

# Display summary
print(f"\n{'='*70}")
print(f"FINAL RESULTS - RAG EVALUATION DATASET")
print(f"{'='*70}")
print(f"Total Questions: {total_q}")
print(f"Total Hits: {total_hits}/{total_q}")
print(f"Hit Rate: {total_hit_rate:.1%}")
print(f"MRR: {total_mrr:.3f}")
print(f"{'='*70}\n")

# Save results
report = {
    "test_type": "Real Questions from PDFs - 50% word-match threshold",
    "total_questions": total_q,
    "total_hits": total_hits,
    "hit_rate": round(total_hit_rate, 3),
    "mrr": round(total_mrr, 3)
}

# Add per-source breakdown
for source_name, metrics in results.items():
    report[source_name] = {
        "hits": metrics['hits'],
        "hit_rate": round(metrics['hit_rate'], 3),
        "mrr": round(metrics['mrr'], 3)
    }

with open("rag_eval_real_questions.json", "w") as f:
    json.dump(report, f, indent=2)

print("Results saved to: rag_eval_real_questions.json")
