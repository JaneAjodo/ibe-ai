from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum

class ToolUsed(str, Enum):
    SEMANTIC_SEARCH = "semantic_search"
    ANALYTICS = "analytics"
    SUMMARY = "summary"
    COMBINED = "combined"

class IngestResponse(BaseModel):
    status: str
    message: str
    records_ingested: int
    collection_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Source(BaseModel):
    policy_id: str
    client_name: str
    policy_type: str
    region: str
    relevance_score: float

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Your question about the insurance data")
    session_id: Optional[str] = Field(default="default", description="Session ID for conversation memory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the average claim amount for life insurance policies in Lagos?",
                "session_id": "user-session-001"
            }
        }

class ChatResponse(BaseModel):
    answer: str
    tool_used: ToolUsed
    confidence: str
    sources: Optional[List[Source]] = None
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    vector_store: str
    records_indexed: int
    gemini_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int
