"""
Hybrid Retriever - Combines BM25 (lexical) with Dense Embeddings (semantic)

This module implements hybrid search combining:
1. BM25Retriever for keyword-based retrieval (lexical matching)
2. FAISS for semantic similarity (dense embeddings)
3. Weighted ensemble scoring via Reciprocal Rank Fusion (RRF)
"""

from typing import List, Dict
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Custom hybrid retriever combining BM25 and dense vector search
    """
    
    bm25_retriever: BM25Retriever
    dense_retriever: BaseRetriever
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    k: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search with RRF"""
        # Get results from both retrievers
        bm25_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)
        
        # Reciprocal Rank Fusion
        doc_scores: Dict[str, float] = {}
        
        # Score BM25 results
        for rank, doc in enumerate(bm25_docs, 1):
            doc_id = doc.page_content[:100]  # Use content snippet as ID
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.bm25_weight / (rank + 60)
        
        # Score dense results
        for rank, doc in enumerate(dense_docs, 1):
            doc_id = doc.page_content[:100]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.dense_weight / (rank + 60)
        
        # Create document map
        doc_map = {}
        for doc in bm25_docs + dense_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Sort by score and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:self.k]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version - not implemented"""
        return self._get_relevant_documents(query)


def create_hybrid_retriever(
    vector_store,
    documents: List[Document],
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    k: int = 5
):
    """
    Create a hybrid retriever combining BM25 and dense search
    
    Args:
        vector_store: FAISS vector store instance
        documents: List of documents for BM25 indexing
        bm25_weight: Weight for BM25 retriever (default 0.5)
        dense_weight: Weight for dense retriever (default 0.5)
        k: Number of documents to retrieve
        
    Returns:
        HybridRetriever combining BM25 and dense search with RRF
    """
    logger.info("Creating hybrid retriever...")
    logger.info(f"  BM25 weight: {bm25_weight}")
    logger.info(f"  Dense weight: {dense_weight}")
    logger.info(f"  Top-k: {k}")
    
    # Create BM25 retriever
    logger.info(f"Building BM25 index with {len(documents)} documents...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    logger.info("BM25 retriever created")
    
    # Create dense retriever from vector store
    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )
    logger.info("Dense retriever created")
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        k=k
    )
    
    logger.info("Hybrid retriever created successfully")
    return hybrid_retriever
