from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from src.core.config import settings
from src.core.logging import setup_logger
from src.routers import ingest, chat, health

logger = setup_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — set it in .env file")
    yield
    logger.info("Ibe AI shutting down")

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Ingestion", "description": "Upload and process Excel insurance data"},
        {"name": "Chat", "description": "Agentic RAG question-answering with conversation memory"},
        {"name": "System", "description": "Health checks and system status"},
    ]
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration)
    logger.info(f"{request.method} {request.url.path} — {response.status_code} — {duration}ms")
    return response

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Routers
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(health.router)

@app.get("/", tags=["System"], summary="Root")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "tagline": "Wisdom, instantly.",
        "docs": "/docs",
        "health": "/health"
    }
