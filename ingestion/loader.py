import os
import glob
from pypdf import PdfReader
from typing import List, Dict, Any
from models.schemas import SourceDocument
from utils.logger import get_logger

logger = get_logger(__name__)

def load_documents(data_dir: str) -> List[SourceDocument]:
    """
    Loads all txt, md, and pdf documents from the specified directory.
    """
    documents = []
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist.")
        return documents
        
    for file_path in glob.glob(f"{data_dir}/**/*.*", recursive=True):
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        title = os.path.basename(file_path)
        
        try:
            if ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif ext == '.pdf':
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            else:
                logger.debug(f"Unsupported file format skipped: {file_path}")
                continue
                
            if content.strip():
                from ingestion.cleaner import clean_text
                cleaned_content = clean_text(content)
                if cleaned_content:
                    documents.append(SourceDocument(file_path=file_path, title=title, content=cleaned_content))
                    logger.info(f"Loaded document: {title}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            
    return documents
