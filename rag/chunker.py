import re

def approximate_tokens(text: str) -> list[str]:
    """
    Very simple approximation: split by whitespace. 
    In production, use tiktoken or similar, but for this constraint we keep it simple.
    """
    return text.split()

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 80) -> list[str]:
    """
    Chunks a text into sizes of approx `chunk_size` words, 
    with `overlap` words of overlap between consecutive chunks.
    """
    words = approximate_tokens(text)
    chunks = []
    
    if not words:
        return chunks

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
        i += (chunk_size - overlap)
        # Avoid infinite loop if overlap >= chunk_size
        if chunk_size - overlap <= 0:
            break
            
    return chunks

def tag_chunk_type(source_file: str) -> str:
    """
    Determine if a chunk is 'schema' or 'insight' based on its filename.
    """
    lower_file = source_file.lower()
    if "dictionary" in lower_file or "schema" in lower_file:
        return "schema"
    return "insight"

def process_documents(documents: list[dict], chunk_size: int = 300, overlap: int = 80) -> list[dict]:
    """
    Process loaded documents, chunk them, and attach metadata.
    """
    processed_chunks = []
    for doc in documents:
        text = doc["text"]
        
        # Don't create chunks for empty pages
        if not text:
            continue
            
        chunks = chunk_text(text, chunk_size, overlap)
        chunk_type = tag_chunk_type(doc["source_file"])
        
        for c in chunks:
            processed_chunks.append({
                "chunk_text": c,
                "metadata": {
                    "source_file": doc["source_file"],
                    "chunk_type": chunk_type,
                    "page_number": doc["page_number"]
                }
            })
            
    return processed_chunks
