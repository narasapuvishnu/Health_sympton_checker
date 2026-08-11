import uuid
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.schemas import SourceDocument, DocumentChunk
from config import settings

def chunk_documents(documents: List[SourceDocument]) -> List[DocumentChunk]:
    """
    Splits documents into smaller chunks using langchain's RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    
    for doc in documents:
        # Split text
        texts = splitter.split_text(doc.content)
        
        for i, text in enumerate(texts):
            chunk_id = str(uuid.uuid4())
            metadata = {
                "source": doc.file_path,
                "title": doc.title,
                "chunk_index": i
            }
            chunks.append(DocumentChunk(id=chunk_id, text=text, metadata=metadata))
            
    return chunks
