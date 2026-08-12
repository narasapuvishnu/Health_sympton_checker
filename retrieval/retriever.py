import threading
from typing import List, Dict, Any
from embeddings.embedding_service import get_embedding_service
from vectorstore.qdrant_client import get_qdrant_client
from models.schemas import RetrievalResult
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_qdrant_client()
        
    def retrieve_context(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Takes a natural language query, embeds it, and retrieves the most relevant
        medical documents from the vector store.
        """
        top_k = top_k or settings.TOP_K
        logger.info(f"Retrieving top {top_k} documents for query: {query}")
        
        # 1. Embed the query
        query_vector = self.embedding_service.get_embedding(query)
        
        # 2. Search Vector DB
        search_results = self.vector_store.search(query_vector=query_vector, top_k=top_k)
        
        # 3. Format results
        retrieval_results = []
        for res in search_results:
            retrieval_results.append(
                RetrievalResult(
                    text=res["text"],
                    metadata=res["metadata"],
                    score=res["score"]
                )
            )
            
        logger.info(f"Retrieved {len(retrieval_results)} relevant documents.")
        return retrieval_results

# Thread-safe singleton
_retriever = None
_retriever_lock = threading.Lock()

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever
