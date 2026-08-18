"""
RAG metrics - measures how well the retriever is working.
Hit@5 = % of top-5 results that were relevant
MRR = Mean Reciprocal Rank (1 / average position of first relevant result)
"""

_LATEST_METRICS_BY_THREAD = {}


def get_cached_metrics(thread_id):
    """Get the last computed metrics for this thread."""
    return _LATEST_METRICS_BY_THREAD.get(str(thread_id), {})


def save_metrics(thread_id, metrics):
    """Save metrics for a thread."""
    _LATEST_METRICS_BY_THREAD[str(thread_id)] = metrics



def format_metrics_report(metrics):
    return f"""
Hit Rate@5:  {metrics['hit_rate@5']:.1%}
Hit Rate@10: {metrics['hit_rate@10']:.1%}
MRR:         {metrics['mrr']:.2f}
Queries:     {metrics['num_queries']}
"""
