from chatbot_rag import get_rag_context
from chatbot_tools import llm


def is_ambiguous_query(query):
    q = query.lower().strip()
    if len(q) < 10:
        return True
    vague = {"it", "that", "this", "tell", "say", "show"}
    return any(word in q for word in vague)


def rewrite_query(query):
    if not is_ambiguous_query(query):
        return query
    try:
        prompt = f"Rewrite this vague query to be specific. Return only the rewritten query.\n\nVague: {query}\n\nClear:"
        response = llm.invoke(prompt)
        return str(response.content).strip()
    except:
        return query


def get_rag_context_with_rewriting(query, thread_id, filename_filter=""):
    rewritten = rewrite_query(query)
    if rewritten != query:
        print(f"[Rewrite] {query} → {rewritten}")
    context = get_rag_context(rewritten, thread_id, filename_filter)
    return rewritten, context
