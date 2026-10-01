import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Models
TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pinecone
PINECONE_INDEX_NAME = "novacore-multimodal-rag"

# Directories
IMAGE_DIR = Path("extracted_images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
