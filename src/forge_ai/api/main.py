"""Forge AI FastAPI Backend.

Provides a REST interface to interact with the ForgeAgent.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from forge_ai.agent.forge import ForgeAgent

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="Forge AI API",
    description="REST API for the Forge ML Assistant",
    version="1.0.0",
)

# Allow CORS for development (e.g., Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for agent sessions
# In production, this should be backed by Redis or a database
sessions: dict[str, ForgeAgent] = {}


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    session_id: str
    query: str
    provider: str = "groq"
    model: str | None = None
    temperature: float = 0.1
    file_path: str | None = None


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    output: str
    steps: list[dict[str, Any]]


def get_agent(session_id: str, provider: str, model: str | None, temperature: float) -> ForgeAgent:
    """Retrieve or create an agent for the session."""
    if session_id not in sessions:
        try:
            sessions[session_id] = ForgeAgent(
                provider=provider,
                model=model,
                temperature=temperature,
                verbose=False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")
    return sessions[session_id]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint to interact with the ML assistant."""
    agent = get_agent(
        session_id=request.session_id,
        provider=request.provider,
        model=request.model,
        temperature=request.temperature,
    )

    kwargs = {}
    if request.file_path:
        kwargs["file_path"] = request.file_path

    try:
        result = agent.run(request.query, **kwargs)
        return ChatResponse(output=result["output"], steps=result.get("steps", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Endpoint to upload a dataset temporarily for analysis."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Create a temporary file that persists until manually deleted
        fd, temp_path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        
        return {"filename": file.filename, "file_path": temp_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()


@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """Clear memory for a given session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "success", "message": f"Session {session_id} cleared"}
    return {"status": "not_found", "message": f"Session {session_id} not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
