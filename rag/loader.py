import fitz  # PyMuPDF
import os
import re

def clean_text(text: str) -> str:
    """
    Remove excessive whitespaces and repetitive headers/footers.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Generic footer/header removal (e.g. Page X of Y, etc.) could be added here
    # For now, stripping edge spaces is sufficient for clean reading
    return text.strip()

def load_pdfs(directory: str) -> list[dict]:
    """
    Loads all PDFs in a directory.
    Returns a list of dicts containing the filename, page number, and text.
    """
    documents = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return documents

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".pdf"):
            continue
            
        filepath = os.path.join(directory, filename)
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    documents.append({
                        "source_file": filename,
                        "page_number": page_num + 1,
                        "text": clean_text(text)
                    })
            doc.close()
            print(f"Loaded {filename} successfully.")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            
    return documents
