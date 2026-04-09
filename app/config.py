import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant local settings
QDRANT_URL = None  # None = use local
QDRANT_API_KEY = None
QDRANT_COLLECTION_NAME = "colpali_docs"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLPALI_MODEL_NAME = "vidore/colsmol-256M"
DPI = 100
TOP_K = 3