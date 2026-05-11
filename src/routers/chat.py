from fastapi import APIRouter, Depends, HTTPException
from src.core.security import verify_api_key
from src.core.logging import setup_logger
from src.models.schemas import ChatRequest, ChatResponse
from src.utils.vector_store import get_vector_store
from src.utils.memory import get_memory
from src.tools.agent import IbeAgent

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = setup_logger("chat_router")

def get_agent() -> IbeAgent:
    vs = get_vector_store()
    memory = get_memory()
    return IbeAgent(vs, memory)

@router.post("/", response_model=ChatResponse, summary="Ask Ibe a question about your insurance data")
async def chat(
    request: ChatRequest,
    _: str = Depends(verify_api_key)
):
    """
    Agentic RAG chat endpoint.
    
    Ibe's agent decides which tool to use based on your query:
    - **Semantic Search** — client lookups, policy details, qualitative questions
    - **Analytics** — averages, totals, counts, statistical comparisons  
    - **Summary** — overviews of groups, patterns across multiple records
    
    Conversation history is maintained per session_id.
    
    Example questions:
    - "Tell me about Adaeze Okonkwo's policy"
    - "What is the average claim amount for Life insurance?"
    - "Summarise all high-risk clients in Lagos"
    - "Which agent has the most approved claims?"
    """
    vs = get_vector_store()
    if vs.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No data ingested yet. Please POST to /ingest first."
        )

    try:
        agent = get_agent()
        result = agent.run(
            query=request.query,
            session_id=request.session_id or "default"
        )

        return ChatResponse(
            answer=result["answer"],
            tool_used=result["tool_used"],
            confidence=result.get("confidence", "medium"),
            sources=result.get("sources"),
            session_id=result["session_id"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.delete("/history/{session_id}", summary="Clear conversation history for a session")
async def clear_history(
    session_id: str,
    _: str = Depends(verify_api_key)
):
    """Clear the conversation memory for a specific session."""
    memory = get_memory()
    memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
