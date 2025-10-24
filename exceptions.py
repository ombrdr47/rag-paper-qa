"""
Custom Exceptions for Research Paper RAG System

Defines specific exception types for better error handling
and debugging throughout the application.
"""


class ResearchPaperRAGError(Exception):
    """Base exception for all RAG system errors"""
    pass


class PDFParsingError(ResearchPaperRAGError):
    """Raised when PDF parsing fails"""
    pass


class EmbeddingError(ResearchPaperRAGError):
    """Raised when embedding generation fails"""
    pass


class VectorStoreError(ResearchPaperRAGError):
    """Raised when vector store operations fail"""
    pass


class ChatSessionError(ResearchPaperRAGError):
    """Raised when chat session operations fail"""
    pass


class ConfigurationError(ResearchPaperRAGError):
    """Raised when configuration is invalid"""
    pass


class APIError(ResearchPaperRAGError):
    """Raised when API calls fail"""
    pass


class RetrievalError(ResearchPaperRAGError):
    """Raised when document retrieval fails"""
    pass
