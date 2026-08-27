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
    Smart retrieval strategy (with confidence scoring):
    1. Try original query with confidence scoring
    2. If fails and ambiguous -> ask user to clarify
    3. If fails and not ambiguous -> return empty (no fallback)
    
    Returns: (query, context, confidence_score)
    - confidence_score: 0-1, used by backend to detect hallucination risk
    """
    from chatbot_rag import get_rag_context_with_confidence
    
    # Try original query first with confidence scoring
    context, confidence = get_rag_context_with_confidence(query, thread_id, filename_filter)
    
    # If we got good context, return it
    if context and len(context) > 100:
        return query, context, confidence
    
    # No good context - check if ambiguous
    if is_ambiguous_query(query):
        # Ask user to clarify - signal via empty context and 0 confidence
        return query, "", 0.0
    
    #  DISABLED: Auto-rewrite fallback removed to prevent hallucination
    # Previously: Tried rewriting query if no chunks found
    # Problem: Rewrites can introduce assumptions, defeating RAG guardrails
    # Solution: Return empty context - let backend refuse gracefully
    
    # Return original query with empty context and low confidence
    # This signals to chatbotBackend that no chunks were found
    # Backend will refuse with: "I don't have this information in the uploaded document"
    return query, context, confidence
