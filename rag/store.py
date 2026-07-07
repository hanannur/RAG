"""Shared settings and constructors for the local Chroma store."""

from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "models/gemini-embedding-001"
load_dotenv()


def get_api_key() -> str:
    """Read the required Gemini key without storing it in project files."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your .env file before using Gemini.")
    return api_key


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embedding model (credentials come from the environment)."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=get_api_key())


def open_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )
