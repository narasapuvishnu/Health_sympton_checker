import os
from typing import List
from models.schemas import DocumentChunk
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from embeddings.embedding_service import get_embedding_service
from vectorstore.qdrant_client import get_qdrant_client
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

def run_ingestion_pipeline(data_dir: str):
    """
    Orchestrates the entire ingestion pipeline:
    1. Load documents
    2. Clean and chunk
    3. Generate embeddings
    4. Store in Vector DB
    """
    logger.info(f"Starting ingestion pipeline for directory: {data_dir}")
    
    # 1 & 2: Load and Chunk
    documents = load_documents(data_dir)
    if not documents:
        logger.warning("No documents found to ingest.")
        return
        
    logger.info(f"Loaded {len(documents)} documents. Chunking...")
    chunks: List[DocumentChunk] = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    
    # 3: Embed
    embedding_service = get_embedding_service()
    logger.info(f"Generating embeddings using {settings.EMBEDDING_PROVIDER} / {settings.EMBEDDING_MODEL}")
    chunks = embedding_service.embed_chunks(chunks)
    
    # 4: Store
    qdrant_client = get_qdrant_client()
    logger.info("Storing chunks in Qdrant...")
    qdrant_client.create_collection()
    qdrant_client.upsert_chunks(chunks)
    
    logger.info("Ingestion pipeline completed successfully.")
