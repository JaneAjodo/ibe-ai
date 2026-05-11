import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "Ibe AI"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Intelligent answers from your insurance data — powered by Gemini"
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    
    API_KEY: str = os.getenv("IBE_API_KEY", "ibe-secret-key-2025")
    
    CHROMA_COLLECTION: str = "ibe_insurance_policies"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    
    MAX_CONVERSATION_HISTORY: int = 10
    TEMPERATURE_EXTRACTION: float = 0.1
    TEMPERATURE_CHAT: float = 0.3

settings = Settings()
