"""
Research Paper Q&A Agent

A production-ready RAG system for intelligent question-answering on research papers.
Built with LangChain, Google Gemini, and HuggingFace embeddings.
"""

__version__ = "1.0.0"
__author__ = "Research Paper RAG Team"
__description__ = "RAG system for research paper question answering"

from .main import ResearchPaperAgent
from .pdf_parser import ResearchPaperParser
from .rag_system import ResearchPaperRAG
from .chat_manager import ChatManager, ChatSession
from .exceptions import (
    ResearchPaperRAGError,
    PDFParsingError,
    EmbeddingError,
    VectorStoreError,
    ChatSessionError,
    ConfigurationError,
    APIError,
    RetrievalError
)

__all__ = [
    "ResearchPaperAgent",
    "ResearchPaperParser",
    "ResearchPaperRAG",
    "ChatManager",
    "ChatSession",
    "ResearchPaperRAGError",
    "PDFParsingError",
    "EmbeddingError",
    "VectorStoreError",
    "ChatSessionError",
    "ConfigurationError",
    "APIError",
    "RetrievalError",
]
