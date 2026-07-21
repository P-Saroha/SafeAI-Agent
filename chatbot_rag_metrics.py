from chatbot_rag import get_rag_context


def calculate_hit_rate(retrieved_contexts, expected_sources, k=5):
    hits = 0
    for context, expected in zip(retrieved_contexts, expected_sources):
        if expected.lower() in context.lower():
            hits += 1
    return hits / len(retrieved_contexts) if retrieved_contexts else 0.0


def calculate_mrr(retrieved_contexts, expected_sources):
    reciprocal_ranks = []
    for context, expected in zip(retrieved_contexts, expected_sources):
        chunks = context.split("\n\n")
        found_rank = None
        for chunk_idx, chunk in enumerate(chunks, start=1):
            if expected.lower() in chunk.lower():
                found_rank = chunk_idx
                break
        if found_rank:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def evaluate_retriever(thread_id, test_queries, expected_sources):
    retrieved_contexts = []
    for query in test_queries:
        context = get_rag_context(query, thread_id)
        retrieved_contexts.append(context)
    
    metrics = {
        "hit_rate@5": calculate_hit_rate(retrieved_contexts, expected_sources, k=5),
        "hit_rate@10": calculate_hit_rate(retrieved_contexts, expected_sources, k=10),
        "mrr": calculate_mrr(retrieved_contexts, expected_sources),
        "num_queries": len(test_queries),
    }
    print(format_metrics_report(metrics))
    return metrics


def format_metrics_report(metrics):
    return f"""
Hit Rate@5:  {metrics['hit_rate@5']:.1%}
Hit Rate@10: {metrics['hit_rate@10']:.1%}
MRR:         {metrics['mrr']:.2f}
Queries:     {metrics['num_queries']}
"""
