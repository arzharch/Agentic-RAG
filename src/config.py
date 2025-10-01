"""
Configuration settings for the LangGraph Agentic RAG system.
Centralizes all configurable parameters for easy adjustment.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===== MODEL CONFIGURATION =====
# Primary LLM for all reasoning tasks
LLM_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_TEMPERATURE = 0.1  # Low temperature for consistent, factual responses

# Embedding model for vector search
# Options: "ollama" (uses nomic-embed-text) or "sentence-transformers"
EMBEDDING_PROVIDER = "sentence-transformers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions

# ===== FILE PATHS =====
# Directory containing the .txt files to analyze
FILES_DIRECTORY = os.path.join(os.getcwd(), "files_to_work_with")

# Vector store persistence
VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vectorstore")

# ===== RETRIEVAL PARAMETERS =====
# Number of document chunks to retrieve per query
TOP_K_CHUNKS = 5

# Chunk size for splitting documents (in characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Number of top files to examine in detail
MAX_FILES_TO_EXAMINE = 3

# ===== AGENT PARAMETERS =====
# Maximum iterations for the ReAct agent in Evidence Gatherer node
MAX_REACT_ITERATIONS = 5

# Timeout for agent execution (seconds)
AGENT_TIMEOUT = 30

# ===== VECTOR STORE SETTINGS =====
# Whether to rebuild vector store on each startup.
# Set to False after first run to speed up initialization.
# Set to True if you change files or chunking settings.
REBUILD_VECTORSTORE = True

# ===== LOGGING =====
VERBOSE = True  # Enable detailed logging for debugging