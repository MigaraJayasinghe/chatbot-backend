# pdf_reader.py
from pdfminer.high_level import extract_text
import re

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    Uses pdfminer.six for better Sinhala, Tamil, and English support.
    """
    try:
        text = extract_text(pdf_path)
        if not text.strip():
            print(f"⚠️ Warning: No text extracted from {pdf_path}")
        return text
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""

def split_text_into_chunks(text, max_length=500):
    """
    Split extracted text into chunks for embedding.
    Handles multiple languages and punctuation marks.
    """
    # Split by common sentence endings: ., ?, !, Sinhala/Tamil punctuation, or newlines
    sentences = re.split(r'[.!?。\n]', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
