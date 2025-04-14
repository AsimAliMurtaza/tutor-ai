from fastapi import FastAPI, Request, Header, HTTPException, Query, Body, Path
import ollama
import asyncio
import datetime
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
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

client = MongoClient(
    "mongodb+srv://moasimmurtaza:u6bHfg5pGzaJvRDz@cluster0.t3yr7os.mongodb.net/cognivia?retryWrites=true&w=majority")
db = client["cognivia"]
notes_collection = db["generated_notes"]

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


@app.get("/ollama")
async def ask_ollama(query: str, request: Request):
    """API endpoint for streaming AI responses with better error handling."""
    model_query = f"Briefly explain the {query}."

    return StreamingResponse(generate_text_stream(model_query, request), media_type="text/plain")


@app.get("/ask")
async def ask(query: str, request: Request, authorization: str = Header(None)):

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing user ID")

    user_id = authorization.replace("Bearer ", "").strip()

    prompt = f"Make Detailed note on the following topic:\n\n{query}"
    queue = asyncio.Queue()
    full_response = ""

    async def generate_and_store():
        nonlocal full_response
        try:
            for response in ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], stream=True):
                if await request.is_disconnected():
                    break
                chunk = response.get("message", {}).get("content", "")
                full_response += chunk
                await queue.put(chunk)
                await asyncio.sleep(0.005)
        except Exception as e:
            print("❌ Error:", e)
        finally:
            await queue.put(None)
            if full_response.strip():
                notes_collection.insert_one({
                    "userID": user_id,  # ✅ Add this line
                    "prompt": query,
                    "generated_quiz": full_response,
                    "createdAt": datetime.datetime.utcnow()
                })

    async def stream_response() -> AsyncGenerator[str, None]:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    asyncio.create_task(generate_and_store())
    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/notes")
async def get_notes(
    skip: int = 0,
    limit: int = 10,
    user_id: str = Query(...)
):
    query = {"userID": user_id}

    total = notes_collection.count_documents(query)
    notes = notes_collection.find(query).sort(
        "createdAt", -1).skip(skip).limit(limit)

    return {
        "total": total,
        "notes": [
            {
                "_id": str(note["_id"]),
                "prompt": note.get("prompt", ""),
                "generated_quiz": note.get("generated_quiz", ""),
                "createdAt": note.get("createdAt"),
                "userID": note.get("userID", None),
            }
            for note in notes
        ]
    }


@app.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    try:
        result = notes_collection.delete_one({"_id": ObjectId(note_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Note not found")
        return {"message": "Note deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid ID or deletion failed: {str(e)}")


@app.put("/notes/{note_id}")
async def update_note(
    note_id: str = Path(...),
    payload: dict = Body(...)
):
    try:
        updatedContent = payload.get("updatedContent")
        if not updatedContent:
            raise HTTPException(status_code=400, detail="No content provided")

        result = notes_collection.update_one(
            {"_id": ObjectId(note_id)},
            {"$set": {"generated_quiz": updatedContent}},
        )

        if result.modified_count == 1:
            return {"message": "Note updated successfully"}

        raise HTTPException(status_code=404, detail="Note not found")

    except Exception as e:
        print("❌ Error during update:", e)
        raise HTTPException(status_code=500, detail="Failed to update note")
