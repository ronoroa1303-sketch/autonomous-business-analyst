from xml.parsers.expat import model

from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once at the module level to avoid reloading
MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def normalize_l2(vector):
    """
    L2 Normalize vector for cosine similarity
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generates L2 normalized embeddings for a list of texts.
    """
    model = get_model()
    texts = ["passage: " + t for t in texts]
    embeddings = model.encode(texts, convert_to_numpy=True)    
    # Normalize for cosine similarity with FAISS Inner Product
    normalized_embeddings = np.array([normalize_l2(emb) for emb in embeddings]).astype('float32')
    return normalized_embeddings

def embed_query(query: str) -> np.ndarray:
    """
    Generate embedding for a single query text.
    """
    model = get_model()
    query = "query: " + query
    embedding = model.encode([query], convert_to_numpy=True)[0]
    return np.array([normalize_l2(embedding)]).astype('float32')
