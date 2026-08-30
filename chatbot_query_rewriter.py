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
    Smart retrieval strategy with confidence scoring and security guardrails.
    
    This function implements a SAFE fallback strategy:
    1. Try original query with confidence scoring
    2. If ambiguous → ask user to clarify
    3. If not ambiguous but no chunks → refuse (NOT rewrite)
    
    Why no automatic rewrite?
    - Rewriting can introduce assumptions beyond what's in documents
    - Defeats RAG security by bypassing the original question intent
    - User's intent might be different from rewritten version
    - Better to ask user or refuse than to answer a different question
    
    Returns: (query, context, confidence_score)
    - query: The question that will be answered
    - context: Formatted chunks, or "" if retrieval failed
    - confidence_score: 0-1 indicating retrieval quality
    """
    from chatbot_rag import get_rag_context_with_confidence
    
    # ── STEP 1: RETRIEVE WITH CONFIDENCE SCORING ─────────────────────
    # Get both context AND confidence score from RAG
    # Confidence is used by backend to decide: answer now or ask user?
    context, confidence = get_rag_context_with_confidence(query, thread_id, filename_filter)
    
    # ── STEP 2: CHECK IF RETRIEVAL WAS GOOD ──────────────────────────
    # If we got >= 100 chars of context, confident enough to answer
    if context and len(context) > 100:
        print(f"[RETRIEVER] Good context found ({len(context)} chars)")
        return query, context, confidence
    
    # ── STEP 3: CHECK IF QUERY IS AMBIGUOUS ──────────────────────────
    # If query uses vague pronouns, ask user to clarify
    # Examples: "What is that?", "How?", "It?" (too short)
    if is_ambiguous_query(query):
        print(f"[RETRIEVER] Query is ambiguous - asking user to clarify")
        # Return empty context + 0 confidence to signal backend
        # Backend will ask: "Your question is unclear. Could you provide more details?"
        return query, "", 0.0
    
    # ── STEP 4: NO REWRITE FALLBACK (SECURITY) ──────────────────────
    # DISABLED: Auto-query rewriting removed to prevent hallucination
    #
    # Previous behavior (REMOVED):
    #   if no chunks found:
    #     rewritten_query = llm.rewrite(query)
    #     context = retrieve(rewritten_query)
    #
    # Why removed?
    # - LLM rewrites can misinterpret user intent
    # - Defeats RAG guardrails by circumventing the original question
    # - If retrieval failed, rewriting doesn't guarantee success
    # - Better to ask user or refuse than to answer a different question
    #
    # Example of risk:
    # User: "What is quantum computing?"
    # No chunks found → LLM rewrites to: "Explain quantum mechanics"
    # LLM then answers general knowledge (hallucination!)
    #
    # Safer approach: Return empty, let backend refuse gracefully
    
    print(f"[RETRIEVER] No good chunks found - returning empty (no rewrite fallback)")
    # Backend sees empty context + low confidence
    # Backend will either trigger HITL or refuse with:
    # "I don't have this information in the uploaded document"
    return query, context, confidence
