"""
Smart query handling:
1. Try original query for RAG first
2. If retrieval is poor (empty/low confidence), check if ambiguous
3. If ambiguous, ask user to clarify
4. If not ambiguous but still no chunks, try rewrite as fallback
"""

from chatbot_rag import get_rag_context
from chatbot_tools import llm


def is_ambiguous_query(query):
    """Check if query uses vague pronouns without context."""
    q = query.lower().strip()
    
    # Too short = ambiguous
    if len(q) < 5:
        return True
    
    # Vague pronouns alone or with minimal context
    vague_only = {"it", "that", "this", "what", "how"}
    words = [w.strip(".,!?;:") for w in q.split()]
    
    if len(words) <= 2 and words[0] in vague_only:
        return True
    
    return False


def rewrite_query(query):
    """Rewrite truly vague query to be specific."""
    if not is_ambiguous_query(query):
        return query
    try:
        prompt = f"Rewrite this vague query to be specific. Return only the rewritten query, nothing else.\n\nVague: {query}\n\nClear:"
        response = llm.invoke(prompt)
        rewritten = str(response.content).strip()
        if rewritten and rewritten != query:
            return rewritten
    except:
        pass
    return query


def get_rag_context_with_rewriting(query, thread_id, filename_filter=""):
    """
    Smart retrieval strategy:
    1. Try original query
    2. If fails and ambiguous -> ask user to clarify
    3. If fails and not ambiguous -> try rewrite as fallback
    """
    # Try original query first
    context = get_rag_context(query, thread_id, filename_filter)
    
    # If we got good context, return it
    if context and len(context) > 100:
        return query, context
    
    # No good context - check if ambiguous
    if is_ambiguous_query(query):
        # Ask user to clarify - signal via empty context
        return query, ""
    
    # Not ambiguous but no chunks - try rewrite as fallback
    rewritten = rewrite_query(query)
    if rewritten != query:
        print(f"[Fallback Rewrite] No chunks found for '{query}' → trying '{rewritten}'")
        context = get_rag_context(rewritten, thread_id, filename_filter)
        return rewritten, context
    
    # Rewrite didn't help either - return original with empty context
    return query, context
