import openai
import faiss
import numpy as np

openai.api_key = "your-openai-api-key"

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Example: Store embeddings
questions = ["What is opportunity cost?", "Explain the law of demand."]
embeddings = np.array([get_embedding(q) for q in questions])

# Use FAISS for efficient searching
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)