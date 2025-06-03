import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Import the LangGraph runner
from tools.chatbot_tools import run_chat_through_graph

# -------------------------------------------------
# 1) Single FastAPI app instance
# -------------------------------------------------
app = FastAPI(title="Chat API (CORS-Enabled)")

# -------------------------------------------------
# 2) Enable no-restriction CORS middleware
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow any origin (all domains/files) :contentReference[oaicite:6]{index=6}
    allow_credentials=True,      # Allow cookies/auth headers :contentReference[oaicite:7]{index=7}
    allow_methods=["*"],         # Allow all HTTP methods :contentReference[oaicite:8]{index=8}
    allow_headers=["*"],         # Allow all headers :contentReference[oaicite:9]{index=9}
)

# -------------------------------------------------
# 3) Request/Response Models for /chat endpoint
# -------------------------------------------------
class ChatRequest(BaseModel):
    """
    - message: the user’s new message
    - thread_id: optional; if provided, preserves conversation
    """
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    """
    - response: the AI’s reply
    - thread_id: the UUID to pass for conversation continuity
    """
    response: str
    thread_id: str

# -------------------------------------------------
# 4) /chat endpoint (defined BEFORE static mount)
# -------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Receives a user message and an optional thread_id, runs it through LangGraph,
    and returns the AI response plus the updated thread_id.
    """
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="`message` cannot be empty.")

    try:
        result = run_chat_through_graph(user_msg, req.thread_id)
        return ChatResponse(response=result["response"], thread_id=result["thread_id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  

# -------------------------------------------------
# 5) Serve static files (UI) at the root path "/"
#    MOUNTED AFTER the /chat endpoint to avoid 405s
# -------------------------------------------------
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static"
)  # Now visiting "/" serves static/index.html without shadowing /chat :contentReference[oaicite:12]{index=12}
