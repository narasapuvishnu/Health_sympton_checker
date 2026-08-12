import os
import warnings
import threading
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from models.schemas import DocumentChunk
from config import settings
from utils.logger import get_logger

# Suppress the harmless torch.classes path warning on Windows
warnings.filterwarnings("ignore", message=".*torch.classes.*")

logger = get_logger(__name__)

class QdrantVectorStore:
    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION
        
        # Determine embedding dimension based on the model (e.g. all-MiniLM-L6-v2 is 384)
        # In a real app, this might be dynamic or configured. 
        self.vector_size = 384 
        
        if settings.QDRANT_URL:
            logger.info(f"Connecting to Qdrant Cloud at {settings.QDRANT_URL}")
            self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        else:
            logger.info(f"Connecting to local Qdrant at {settings.QDRANT_LOCAL_PATH}")
            os.makedirs(settings.QDRANT_LOCAL_PATH, exist_ok=True)
            self.client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
            
    def create_collection(self):
        """
        Creates the Qdrant collection if it doesn't exist.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")
            
    def upsert_chunks(self, chunks: List[DocumentChunk]):
        """
        Upserts chunks with their embeddings into the vector store.
        """
        if not chunks:
            return
            
        points = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.id} has no embedding. Skipping.")
                continue
                
            points.append(PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "text": chunk.text,
                    **chunk.metadata
                }
            ))
            
        logger.info(f"Upserting {len(points)} vectors to Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info("Upsert complete.")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the vector store using the query vector.
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            formatted_results = []
            for hit in results:
                formatted_results.append({
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                })
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during Qdrant search: {e}")
            return []

# Thread-safe singleton — prevents the Qdrant file-lock race condition
# that occurs when Streamlit's multiple startup threads all call get_qdrant_client()
# simultaneously before the cache is populated.
_qdrant_client = None
_qdrant_lock = threading.Lock()

def get_qdrant_client() -> QdrantVectorStore:
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            # Double-checked locking: re-check inside the lock
            if _qdrant_client is None:
                _qdrant_client = QdrantVectorStore()
    return _qdrant_client
