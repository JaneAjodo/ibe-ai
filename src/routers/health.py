from fastapi import APIRouter
from src.core.config import settings
from src.core.logging import setup_logger
from src.models.schemas import HealthResponse
from src.utils.vector_store import get_vector_store
from google import genai

router = APIRouter(prefix="/health", tags=["System"])
logger = setup_logger("health_router")

@router.get("/", response_model=HealthResponse, summary="System health check")
async def health_check():
    """
    Returns the current health status of Ibe AI including:
    - Vector store status and record count
    - Gemini API connectivity
    - Application version info
    """
    vs = get_vector_store()
    record_count = vs.count()

    gemini_ok = False
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents="ping",
            config=genai.types.GenerateContentConfig(max_output_tokens=5)
        )
        gemini_ok = True
    except Exception as e:
        logger.warning(f"Gemini health check failed: {e}")

    status = "healthy" if gemini_ok and record_count > 0 else "degraded" if gemini_ok else "unhealthy"

    return HealthResponse(
        status=status,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        vector_store="ChromaDB (local)" if record_count >= 0 else "unavailable",
        records_indexed=record_count,
        gemini_connected=gemini_ok
    )
