import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional


from tools.chatbot_tools import run_chat_through_graph


app = FastAPI(title="Chat API (CORS-Enabled)")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,      
    allow_methods=["*"],        
    allow_headers=["*"],        
)


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
