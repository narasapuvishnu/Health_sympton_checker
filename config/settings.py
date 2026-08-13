import os
from dotenv import load_dotenv

# override=True ensures .env always wins over any shell-inherited env vars
load_dotenv(override=True)

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Vector Database Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_knowledge")
# Fallback to local storage if URL is not provided
QDRANT_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_storage")

# Embeddings Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", 5))

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
