"""
RAG System for Research Paper Question Answering

This module implements the core RAG (Retrieval Augmented Generation) system
using LangChain components including:
- Advanced retrievers with multiple strategies
- Custom prompts optimized for research papers
- Chain components for question answering
- Context-aware response generation
"""

from typing import List, Dict, Any, Optional
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from hybrid_retriever import create_hybrid_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchPaperRAG:
    """
    Advanced RAG system for research paper question answering
    """
    
    # Custom prompt template for research papers
    QA_PROMPT_TEMPLATE = """You are an AI assistant specialized in analyzing research papers. 
You have access to specific sections, figures, tables, and references from a research paper.

Use the following pieces of context from the research paper to answer the question at the end. 
If you don't know the answer based on the provided context, just say that you don't know, 
don't try to make up an answer.

When answering:
1. Be precise and cite specific sections when possible
2. If referencing figures or tables, mention them explicitly
3. If the answer involves methodology, explain it clearly
4. For questions about results, provide quantitative details when available
5. Always base your answer strictly on the provided context

Context from the research paper:
{context}

Question: {question}

Detailed Answer:"""

    CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question that captures all relevant context 
from the research paper.

Chat History:
{chat_history}

Follow Up Question: {question}
Standalone Question:"""
    
    def __init__(
        self,
        vector_store,
        documents: List[Document] = None,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        retrieval_strategy: str = "similarity",
        top_k: int = 5
    ):
        """
        Initialize the RAG system
        
        Args:
            vector_store: Vector store instance (FAISS or Chroma)
            documents: List of Document objects (required for hybrid retrieval)
            model_name: Google Gemini model name
            temperature: Temperature for generation (0 = deterministic)
            retrieval_strategy: Strategy for retrieval ('similarity', 'mmr', 'compression', 'multi_query', 'hybrid')
            top_k: Number of documents to retrieve
        """
        logger.info("="*80)
        logger.info("Initializing Research Paper RAG System")
        logger.info("="*80)
        logger.info(f"LLM Model: {model_name}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Retrieval Strategy: {retrieval_strategy}")
        logger.info(f"Top K Documents: {top_k}")
        
        self.vector_store = vector_store
        self.documents = documents
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        logger.info("Google Gemini LLM initialized")
        
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k
        
        # Initialize retriever
        logger.info(f"Setting up {retrieval_strategy} retriever...")
        self.retriever = self._create_retriever()
        
        # Initialize QA chain
        logger.info("Creating QA chain...")
        self.qa_chain = self._create_qa_chain()
        logger.info("QA chain ready")
        
        # Chat history for conversational QA
        self.chat_history = []
        logger.info("="*80)
    
    def _create_retriever(self):
        """Create retriever based on specified strategy"""
        
        logger.info(f"Creating retriever with strategy: {self.retrieval_strategy}")
        
        if self.retrieval_strategy == "similarity":
            # Simple similarity search
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
            logger.info(f"Similarity retriever created (top_k={self.top_k})")
        
        elif self.retrieval_strategy == "mmr":
            # Maximum Marginal Relevance (diverse results)
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.top_k,
                    "fetch_k": self.top_k * 3,  # Fetch more, then re-rank
                    "lambda_mult": 0.5  # Diversity factor
                }
            )
            logger.info(f"MMR retriever created (top_k={self.top_k}, fetch_k={self.top_k*3})")
        
        elif self.retrieval_strategy == "compression":
            # Contextual compression - extract only relevant parts
            logger.info("Setting up contextual compression retriever...")
            base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k * 2}
            )
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            logger.info(f"Compression retriever created (base_k={self.top_k*2})")
        
        elif self.retrieval_strategy == "multi_query":
            # Multi-query retrieval - generate multiple queries from one
            logger.info("Setting up multi-query retriever...")
            base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=self.llm
            )
            logger.info(f"Multi-query retriever created (top_k={self.top_k})")
        
        elif self.retrieval_strategy == "hybrid":
            # Hybrid retrieval - BM25 + Dense embeddings
            if self.documents is None:
                logger.error("Hybrid retrieval requires documents list")
                raise ValueError("documents parameter is required for hybrid retrieval strategy")
            
            logger.info("Setting up hybrid retriever (BM25 + Dense)...")
            retriever = create_hybrid_retriever(
                vector_store=self.vector_store,
                documents=self.documents,
                bm25_weight=0.5,
                dense_weight=0.5,
                k=self.top_k
            )
            logger.info(f"Hybrid retriever created (top_k={self.top_k})")
        
        else:
            logger.error(f"Unknown retrieval strategy: {self.retrieval_strategy}")
            raise ValueError(f"Unknown retrieval strategy: {self.retrieval_strategy}")
        
        return retriever
    
    def _create_qa_chain(self):
        """Create the question-answering chain"""
        
        # Create custom prompt
        prompt = PromptTemplate(
            template=self.QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Create chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join(
                f"[Source: {doc.metadata.get('section', 'Unknown')}]\n{doc.page_content}"
                for doc in docs
            )
        
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask(self, question: str, return_source_documents: bool = False) -> Dict[str, Any]:
        """
        Ask a question about the research paper
        
        Args:
            question: Question to ask
            return_source_documents: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional source documents
        """
        logger.info("="*80)
        logger.info("Processing Question")
        logger.info("="*80)
        logger.info(f"Question: {question}")
        logger.info(f"Return sources: {return_source_documents}")
        
        try:
            # Get answer
            logger.info("Generating answer using RAG chain...")
            answer = self.qa_chain.invoke(question)
            logger.info("Answer generated successfully")
            
            result = {
                "question": question,
                "answer": answer
            }
            
            # Optionally retrieve source documents
            if return_source_documents:
                logger.info("Retrieving source documents...")
                docs = self.retriever.invoke(question)
                result["source_documents"] = docs
                result["sources"] = self._format_sources(docs)
                logger.info(f"Retrieved {len(docs)} source documents")
            
            logger.info("="*80)
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise
    
    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and return detailed source information
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info("="*80)
        logger.info("Processing Question with Source Attribution")
        logger.info("="*80)
        logger.info(f"Question: {question}")
        
        try:
            # Retrieve relevant documents
            logger.info("Retrieving relevant documents...")
            docs = self.retriever.invoke(question)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Generate answer using the documents
            logger.info("Generating answer with context...")
            context = "\n\n".join(
                f"[Source: {doc.metadata.get('section', 'Unknown')} - Page {doc.metadata.get('page_number', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            )
            
            prompt = PromptTemplate(
                template=self.QA_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            answer = self.llm.invoke(prompt.format(context=context, question=question))
            logger.info("Answer generated with source attribution")
            logger.info("="*80)
            
            return {
                "question": question,
                "answer": answer.content,
                "source_documents": docs,
                "sources": self._format_sources(docs),
                "num_sources": len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error in ask_with_sources: {e}")
            raise
    
    def conversational_ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question with conversation history
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer and updated chat history
        """
        # Build context from recent chat history
        context_messages = []
        if self.chat_history:
            # Include last 2 Q&A pairs for context
            recent_history = self.chat_history[-2:]
            for q, a in recent_history:
                # Just include questions, keep it brief
                context_messages.append(f"Previous Q: {q}")
        
        # Append context to current question if there's history
        if context_messages:
            enhanced_question = f"{' | '.join(context_messages)}\n\nCurrent Question: {question}"
        else:
            enhanced_question = question
        
        # Get answer using enhanced question
        result = self.ask(enhanced_question, return_source_documents=True)
        result["original_question"] = question
        
        # Update chat history with original question
        self.chat_history.append((question, result["answer"]))
        
        return result
    
    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for display"""
        sources = []
        
        for doc in docs:
            sources.append({
                "section": doc.metadata.get("section", "Unknown"),
                "page_number": doc.metadata.get("page_number", "N/A"),
                "doc_type": doc.metadata.get("doc_type", "Unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return sources
    
    def get_paper_summary(self) -> str:
        """Generate a summary of the research paper"""
        logger.info("Generating paper summary...")
        summary_prompt = """Based on the title, abstract, and main sections of this research paper, 
provide a comprehensive summary covering:
1. The main research question or problem
2. The methodology used
3. Key findings and results
4. Main contributions

Keep the summary concise but informative (300-400 words)."""
        
        result = self.ask(summary_prompt)
        logger.info("Paper summary generated")
        return result["answer"]
    
    def find_methodology(self) -> Dict[str, Any]:
        """Extract methodology from the paper"""
        logger.info("Extracting methodology...")
        question = "What methodology or approach does this paper use? Describe it in detail."
        result = self.ask_with_sources(question)
        logger.info("Methodology extracted")
        return result
    
    def find_key_results(self) -> Dict[str, Any]:
        """Extract key results from the paper"""
        logger.info("Extracting key results...")
        question = "What are the main results and findings of this research? Include quantitative results if available."
        result = self.ask_with_sources(question)
        logger.info("Key results extracted")
        return result
    
    def find_contributions(self) -> Dict[str, Any]:
        """Extract main contributions"""
        logger.info("Extracting contributions...")
        question = "What are the main contributions of this research paper?"
        result = self.ask_with_sources(question)
        logger.info("Contributions extracted")
        return result
    
    def compare_with_related_work(self) -> Dict[str, Any]:
        """Get information about related work"""
        logger.info("Analyzing related work...")
        question = "What related work is discussed in this paper? How does this work differ from or build upon previous research?"
        result = self.ask_with_sources(question)
        logger.info("Related work analyzed")
        return result
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
        logger.info("Conversation history cleared")
