import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pdf_reader import extract_text_from_pdf, split_text_into_chunks

# Load multilingual model (handles Sinhala, English, Tamil, and more)
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Folder containing PDFs
pdf_folder = "pdfs"
all_chunks = []

# Read all PDFs and split into chunks
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        print(f"📄 Processing: {filename}")
        try:
            text = extract_text_from_pdf(filepath)
            if text.strip():  # Make sure text is not empty
                chunks = split_text_into_chunks(text)
                all_chunks.extend(chunks)
            else:
                print(f"⚠️ No text extracted from: {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# Check if chunks were extracted
if not all_chunks:
    raise ValueError("No text chunks extracted. Please check your PDFs.")

# Generate sentence embeddings
print("🔎 Generating embeddings...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save FAISS index and corresponding text chunks
faiss.write_index(index, "commerce_index.faiss")
np.save("commerce_texts.npy", np.array(all_chunks))

print("✅ All language PDFs (Sinhala, English, Tamil) embedded and stored successfully!")
