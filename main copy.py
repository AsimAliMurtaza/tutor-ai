from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os
import faiss
import numpy as np
import chromadb
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import ollama
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator


app = FastAPI()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load Ollama model
MODEL_NAME = "llama3.2:1b"


async def generate_text_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Stream response from Ollama AI model."""
    for response in ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], stream=True):
        yield response["message"]["content"]  # Send one chunk at a time


@app.get("/stream_generate/")
async def stream_generate(query: str):
    """API endpoint for streaming AI responses."""
    model_query = f"Briefly explain the {query}."
    return StreamingResponse(generate_text_stream(model_query), media_type="text/plain")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = " ".join([page.extract_text()
                    for page in reader.pages if page.extract_text()])
    return text.strip()

# Function to chunk text


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to store chunks in ChromaDB


def store_chunks(chunks):
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], metadatas=[
                       {"source": "PDF"}], ids=[str(i)])

# API to upload and process a PDF


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_pdf(content)

    if not text:
        return {"error": "No extractable text found"}

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # Store in ChromaDB
    store_chunks(chunks)

    return {"message": "PDF processed successfully", "chunks_stored": len(chunks)}

# API to retrieve relevant chunks


@app.get("/retrieve/")
async def retrieve_chunks(query: str, top_k: int = 5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return {"relevant_chunks": results["documents"][0]}

# API to generate a response using retrieved context


@app.get("/generate/")
async def generate_response(query: str):
    results = collection.query(query_texts=[query], n_results=5)
    context = " ".join(results["documents"][0])

    model_query = f"Briefly explain the {query} with the following context: {context}"

    response = ollama.chat(model=MODEL_NAME, messages=[
                           {"role": "user", "content": model_query}])

    return {"response": response["message"]["content"]}




#Working Code
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import ollama
import asyncio
from pymongo import MongoClient
from bson import ObjectId

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  # Next.js Dev Server
    "http://127.0.0.1:3000",
    "http://cognivia.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


MODEL_NAME = "qwen2.5:3b"


async def generate_text_stream(prompt: str, request: Request) -> AsyncGenerator[str, None]:
    """Stream response from Ollama AI model while handling client disconnects."""
    try:
        for response in ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], stream=True):
            if await request.is_disconnected():
                print("⚠️ Client disconnected. Stopping streaming.")
                break  # Stop sending data if client disconnects

            yield response["message"]["content"]

            # Prevent tight loops consuming CPU
            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        print("⚠️ Streaming task was cancelled.")
    except Exception as e:
        print(f"❌ Error in streaming response: {e}")


@app.get("/ask")
async def ask_ollama(query: str, request: Request):
    """API endpoint for streaming AI responses with better error handling."""
    model_query = f"Briefly explain the {query}."
    return StreamingResponse(generate_text_stream(model_query, request), media_type="text/plain")

