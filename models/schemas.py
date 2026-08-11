from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float

class RAGResponse(BaseModel):
    symptoms_summary: str
    possible_conditions: str
    general_information: str
    warning_signs: str
    when_to_seek_care: str
    sources: List[Dict[str, str]]
    disclaimer: str = "This information is for educational purposes only and is not a medical diagnosis or a substitute for professional medical advice."
    
class SourceDocument(BaseModel):
    file_path: str
    title: str
    content: str
