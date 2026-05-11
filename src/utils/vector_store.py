import chromadb
from chromadb.config import Settings as ChromaSettings
from google import genai
from typing import List, Dict, Any, Optional
from src.core.config import settings
from src.core.logging import setup_logger

logger = setup_logger("vector_store")

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.collection = self._get_or_create_collection()
        logger.info(f"VectorStore initialised — collection: {settings.CHROMA_COLLECTION}")

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, text: str) -> List[float]:
        result = self.gemini.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=genai.types.EmbedContentConfig(task_type="retrieval_document")
        )
        return result.embeddings[0].values

    def _embed_query(self, text: str) -> List[float]:
        result = self.gemini.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=genai.types.EmbedContentConfig(task_type="retrieval_query")
        )
        return result.embeddings[0].values

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        logger.info(f"Embedding {len(documents)} documents...")
        embeddings = [self._embed(doc) for doc in documents]
        
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully indexed {len(documents)} documents")

    def search(self, query: str, n_results: int = None) -> Dict[str, Any]:
        k = n_results or settings.TOP_K_RESULTS
        query_embedding = self._embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"]
        )
        return results

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(settings.CHROMA_COLLECTION)
        self.collection = self._get_or_create_collection()
        logger.info("Vector store cleared")

# Singleton
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
