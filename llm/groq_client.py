from groq import Groq
from config import settings
from utils.logger import get_logger
import json

logger = get_logger(__name__)

class GroqClient:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            logger.warning("GROQ_API_KEY is not set! LLM calls will fail.")
            self.client = None
            
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generates a response from the Groq LLM.
        """
        if not self.client:
            raise ValueError("Groq client is not initialized. Please provide an API key.")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        logger.info(f"Sending request to Groq model: {self.model}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2, # Low temperature for factual RAG
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise

# Singleton pattern
_groq_client = None

def get_groq_client() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client
