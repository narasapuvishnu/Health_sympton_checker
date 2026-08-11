import re

def clean_text(text: str) -> str:
    """
    Cleans and normalizes text extracted from documents.
    """
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove null bytes or non-printable chars if necessary (basic cleanup)
    text = text.replace('\x00', '')
    
    return text.strip()
