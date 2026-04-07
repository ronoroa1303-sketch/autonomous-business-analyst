"""
rag/rag_agent.py

RAG Agent — retrieves relevant document chunks for a given query.

How it works (viva explanation):
  1. Embed the incoming query with the same model used at index-build time.
  2. Run an inner-product (cosine) search on the FAISS index.
  3. Filter out chunks below a similarity threshold.
  4. Join the top-k chunk texts into one string and return it.
  5. If nothing is relevant enough → return "INVALID_QUERY".

No reranking, no exotic pipelines — just embed → search → filter → join.
"""

from rag.embedder import embed_query
from rag.vector_store import VectorStore

# Load the FAISS index once when the module is first imported.
_store = VectorStore()

# Minimum cosine similarity score to consider a chunk relevant.
# With L2-normalized 384-dim embeddings on IndexFlatIP, scores
# typically range from ~0.2 (irrelevant) to ~0.8 (strong match).
SIMILARITY_THRESHOLD = 0.3


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Search the FAISS vector store for the most relevant chunks.

    Args:
        query  : The user's natural-language question.
        top_k  : Number of chunks to retrieve (default 3).

    Returns:
        A single string of the relevant chunks joined by newlines,
        or "INVALID_QUERY" if no chunk passes the similarity threshold.
    """
    # Step 1: embed the query
    query_embedding = embed_query(query)

    # Step 2: search
    results = _store.search(query_embedding, query_text=query, top_k=top_k)

    if not results:
        return "INVALID_QUERY"

    # Step 3: filter by similarity threshold
    relevant = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]

    if not relevant:
        print(f"RAG: all scores below threshold ({SIMILARITY_THRESHOLD}). "
              f"Best score was {results[0]['score']:.3f}")
        return "INVALID_QUERY"

    # Step 4: extract chunk texts and join
    chunks = [r["chunk_text"] for r in relevant]
    print(f"RAG: returning {len(chunks)} chunks (best score: {relevant[0]['score']:.3f})")
    return "\n\n".join(chunks)

