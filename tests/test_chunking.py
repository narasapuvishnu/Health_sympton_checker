import pytest
from models.schemas import SourceDocument
from ingestion.chunker import chunk_documents
from config import settings

def test_chunking():
    doc = SourceDocument(
        file_path="test.txt",
        title="Test Document",
        content="This is a test. " * 100  # Generate some text
    )
    
    # Save current settings and mock them for the test
    original_size = settings.CHUNK_SIZE
    settings.CHUNK_SIZE = 50
    
    chunks = chunk_documents([doc])
    
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)
    assert chunks[0].metadata["title"] == "Test Document"
    
    # Restore settings
    settings.CHUNK_SIZE = original_size
