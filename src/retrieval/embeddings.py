"""
Embedding provider wrapper.
Supports both Ollama (local) and sentence-transformers embeddings.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_PROVIDER, EMBEDDING_MODEL


def get_embeddings():
    """
    Returns the configured embedding model.
    
    Returns:
        Embeddings: LangChain-compatible embedding model
    """
    if EMBEDDING_PROVIDER == "ollama":
        print("Using Ollama embeddings (nomic-embed-text)...")
        return OllamaEmbeddings(model="nomic-embed-text")
    
    elif EMBEDDING_PROVIDER == "sentence-transformers":
        print(f"Using sentence-transformers embeddings ({EMBEDDING_MODEL})...")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
    
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")