from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# === Load Embedded Data ===
index_file = "commerce_index.faiss"
texts_file = "commerce_texts.npy"

if not os.path.exists(index_file):
    raise FileNotFoundError(f"❌ FAISS index file '{index_file}' not found!")

if not os.path.exists(texts_file):
    raise FileNotFoundError(f"❌ Text data file '{texts_file}' not found!")

index = faiss.read_index(index_file)
documents = np.load(texts_file, allow_pickle=True)

# === Load Multilingual Embedding Model ===
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# === Google Gemini Setup ===
API_KEY = "AIzaSyDjMXyYGKB0yO2eIGLAx7hv1a-oRVtrLYQ"  # Your API key
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request and Response Models ===
class ChatRequest(BaseModel):
    message: str
    language: str = "sinhala"  # Default language

# === Retrieve Most Relevant Text Chunk ===
def retrieve_relevant_text(query: str) -> str:
    query_embedding = model.encode([query])

    if query_embedding.shape[1] != index.d:
        raise ValueError(f"Embedding dimension mismatch: expected {index.d}, got {query_embedding.shape[1]}")

    _, indices = index.search(query_embedding.astype("float32"), k=3)  # Top 3 chunks
    relevant_chunks = [documents[idx] for idx in indices[0] if idx != -1]
    return "\n\n".join(relevant_chunks)

# === Chat Endpoint ===
@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    try:
        relevant_text = retrieve_relevant_text(request.message)

        # Language prompt control
        lang_prompt = {
            "sinhala": "Answer the question in Sinhala",
            "english": "Answer the question in English",
            "tamil": "Answer the question in Tamil",
        }.get(request.language.lower(), "Answer the question in Sinhala")

        final_query = f"{relevant_text}\n\n{lang_prompt}: {request.message}"

        response = gemini_model.generate_content(final_query)

        return {"response": response.text.strip()}

    except Exception as e:
        return {"error": str(e)}

# === Root Test Endpoint ===
@app.get("/")
async def root():
    return {"message": "✅ Multilingual PDF Chatbot API is live!"}
