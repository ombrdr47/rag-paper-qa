"""
Configuration Constants for Research Paper RAG System

This module contains all configurable parameters and default values
used throughout the application.
"""

from pathlib import Path

# API Configuration
DEFAULT_EMBEDDING_PROVIDER = "huggingface"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.0

# Vector Store Configuration
DEFAULT_VECTOR_STORE_TYPE = "faiss"
DEFAULT_VECTOR_STORE_DIR = "./vector_store"

# Text Processing Configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K_RESULTS = 5

# Batch Processing Configuration
BATCH_SIZE = 20
BATCH_DELAY_SECONDS = 0.1

# Chat Configuration
CHAT_HISTORY_LENGTH = 2  # Number of Q&A pairs to keep in context
CHAT_SESSIONS_DIR = "./chat_sessions"

# Retrieval Strategies
RETRIEVAL_STRATEGIES = ["similarity", "mmr", "compression", "multi_query"]
DEFAULT_RETRIEVAL_STRATEGY = "similarity"

# Logging Configuration
LOG_FILE = "research_paper_agent.log"
CONSOLE_LOG_LEVEL = "ERROR"
FILE_LOG_LEVEL = "DEBUG"

# File Paths
DATA_DIR = Path("./data")
VECTOR_STORE_PATH = Path(DEFAULT_VECTOR_STORE_DIR)
CHAT_SESSIONS_PATH = Path(CHAT_SESSIONS_DIR)

# Validation
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 5000
MIN_TOP_K = 1
MAX_TOP_K = 20
