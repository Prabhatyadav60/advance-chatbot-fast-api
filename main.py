import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Import the graph‐runner that now includes our system instructions
from tools.chatbot_tools import run_chat_through_graph

app = FastAPI(title="Chat API")

# -------------------------------------------------
# 1) Serve ./static folder at /static

# -------------------------------------------------
# 2) Data models
# -------------------------------------------------
class ChatRequest(BaseModel):
    """
    - message: the user’s new message
    - thread_id: optional; if provided, the same conversation state is used
    """
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    """
    - response: the AI’s reply (with any tool results embedded)
    - thread_id: the UUID you should pass next time to keep the conversation going
    """
    response: str
    thread_id: str

# -------------------------------------------------
# 3) /chat endpoint
# -------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="`message` cannot be empty.")

    try:
        result = run_chat_through_graph(user_msg, req.thread_id)
        return ChatResponse(response=result["response"], thread_id=result["thread_id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

