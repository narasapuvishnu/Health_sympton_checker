from typing import List
from models.schemas import DocumentChunk
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        
        if self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.error("sentence-transformers is not installed. Run 'pip install sentence-transformers'")
                raise
        else:
            logger.warning(f"Embedding provider {self.provider} not fully supported. Falling back to dummy embeddings if model not loaded.")
            
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        """
        if self.model:
            # sentence-transformers returns a numpy array, convert to list of floats
            return self.model.encode(text).tolist()
        else:
            # Fallback or unsupported provider
            logger.warning("Using dummy embedding!")
            return [0.0] * 384
            
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for a list of DocumentChunks.
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        if self.model:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.model.encode(texts)
            
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb.tolist()
        else:
            for chunk in chunks:
                chunk.embedding = self.get_embedding(chunk.text)
                
        return chunks

# Singleton instance for the service — persists across Streamlit reruns
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

# Pre-warm: expose the model loader so Streamlit can cache it at app level
def get_sentence_transformer_model(model_name: str):
    """Load and return a SentenceTransformer model, cached at module level."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
