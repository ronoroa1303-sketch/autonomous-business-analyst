import os
from rag.loader import load_pdfs
from rag.chunker import process_documents
from rag.embedder import generate_embeddings, embed_query
from rag.vector_store import VectorStore

# Initialize Vector Store
store = VectorStore()

def build_index(data_dir: str = "data/docs"):
    """
    Complete pipeline flow: Load PDFs -> Chunk -> Embed -> Store.
    """
    print(f"Building index from {data_dir}...")
    
    # 1. Load all PDFs
    documents = load_pdfs(data_dir)
    if not documents:
        print("No documents loaded. Index unchanged.")
        return
        
    print(f"Loaded {len(documents)} document pages.")
    
    # 2. Chunk them
    processed_chunks = process_documents(documents)
    print(f"Created {len(processed_chunks)} chunks.")
    if not processed_chunks:
        return
        
    # Extract texts for embedding
    texts_to_embed = [item["chunk_text"] for item in processed_chunks]
    
    # 3. Generate embeddings
    embeddings = generate_embeddings(texts_to_embed)
    print(f"Generated {len(embeddings)} embeddings.")
    
    # 4. Store in vector DB
    store.add_embeddings(embeddings, processed_chunks)
    print("Successfully built and saved vector store.")


def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieval Function:
    - embed query
    - search top_k = 5
    - return relevant chunks prioritization applied
    """
    query_emb = embed_query(query)
    results = store.search(query_emb, query, top_k=top_k)
    return results

if __name__ == "__main__":
    # If run directly, test the indexing and retrieving
    build_index()
    test_q = "How does delivery time impact customer satisfaction?"
    res = retrieve_context(test_q)
    print("\nTest Retrieval for:", test_q)
    for r in res:
        print(f"[{r['metadata']['chunk_type']}] Score: {r['score']:.4f} - Src: {r['metadata']['source_file']}")
