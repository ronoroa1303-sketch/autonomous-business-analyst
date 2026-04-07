import faiss
import numpy as np
import pickle
import os

INDEX_FILE = "rag/store/faiss.index"
METADATA_FILE = "rag/store/metadata.pkl"

class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        # Using IndexFlatIP since vectors are L2 normalized, it equals Cosine Similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = {}  # Map of id -> {chunk_text, metadata_dict}
        self.load()

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: list[dict]):
        if len(embeddings) != len(metadata_list):
            raise ValueError("Mismatched embeddings and metadata length")
            
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        for i, meta in enumerate(metadata_list):
            self.metadata[start_id + i] = meta
            
        self.save()

    def save(self):
        os.makedirs("rag/store", exist_ok=True)
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Loaded existing index with {self.index.ntotal} vectors.")
        else:
            print("No existing index found. Starting fresh.")

    def search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        # Fetch more initial candidates so we can prioritize safely
        fetch_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, fetch_k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                break
            doc = self.metadata[idx]
            results.append({
                "score": float(scores[0][rank]),
                "chunk_text": doc["chunk_text"],
                "metadata": doc["metadata"]
            })

        # --- Optimization: Prioritize Based on Query ---
        # Detect if query relates to schema/data structure
        schema_keywords = ["schema", "column", "table", "datatype", "definition", "structure"]
        is_schema_query = any(kw in query_text.lower() for kw in schema_keywords)
        
        def priority_score(item):
            # Base score is the cosine similarity
            score = item["score"]
            chunk_type = item["metadata"]["chunk_type"]
            
            # Boost score based on chunk_type matching what is relevant
            if is_schema_query and chunk_type == "schema":
                score += 0.2  # Arbitrary boost
            elif not is_schema_query and chunk_type == "insight":
                score += 0.2
            return score

        # Sort with custom priority logic and return top_k
        results.sort(key=priority_score, reverse=True)
        return results[:top_k]
