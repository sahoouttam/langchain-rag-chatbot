import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "tourist_info")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

WIKIPEDIA_QUERY = os.getenv("WIKIPEDIA_QUERY", "Paestum")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")
CILENTO_DIR = os.getenv("CILENTO_DIR", "./documents/CilentoTouristInfo")
PAESTUM_DIR = os.getenv("PAESTUM_DIR", "./documents/Paestum")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))


if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
        "(get one free at https://console.groq.com/keys)."
    )