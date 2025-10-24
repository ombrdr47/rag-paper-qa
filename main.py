"""
Main Application - Research Paper Q&A Agent

This is the main entry point for the research paper agent.
It provides a command-line interface for:
1. Ingesting research papers
2. Asking questions about the paper
3. Managing conversational chat sessions
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse
from dotenv import load_dotenv

from data_ingestion import ResearchPaperIngestionPipeline
from rag_system import ResearchPaperRAG
from chat_manager import ChatManager
from logger import setup_logging, get_logger
from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_TOP_K_RESULTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_VECTOR_STORE_TYPE,
    DEFAULT_VECTOR_STORE_DIR
)
from exceptions import ResearchPaperRAGError, ConfigurationError

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


class ResearchPaperAgent:
    """
    Main agent for research paper question answering
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
        top_k: int = DEFAULT_TOP_K_RESULTS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        vector_store_type: str = DEFAULT_VECTOR_STORE_TYPE,
        persist_directory: str = DEFAULT_VECTOR_STORE_DIR
    ):
        """
        Initialize the research paper agent.
        
        Args:
            embedding_model: Embedding model name (overrides env var)
            llm_model: LLM model name (overrides env var)
            temperature: Temperature for generation (0.0 = deterministic)
            retrieval_strategy: Retrieval strategy ('similarity', 'mmr', 'compression', 'multi_query')
            top_k: Number of documents to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_store_type: Type of vector store ('faiss')
            persist_directory: Directory to persist vector store
            
        Raises:
            ConfigurationError: If API key is missing or configuration is invalid
        """
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING RESEARCH PAPER AGENT")
        logger.info("="*80)
        
        # Validate API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ConfigurationError("GOOGLE_API_KEY not found in environment variables")
        
        # Get from env or use defaults
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.llm_model = llm_model or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        self.temperature = temperature
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k
        
        logger.info(f"Configuration:")
        logger.info(f"  - Embedding Provider: {self.embedding_provider}")
        logger.info(f"  - Embedding Model: {self.embedding_model}")
        logger.info(f"  - LLM Model: {self.llm_model}")
        logger.info(f"  - Temperature: {self.temperature}")
        logger.info(f"  - Retrieval Strategy: {self.retrieval_strategy}")
        logger.info(f"  - Top K: {self.top_k}")
        logger.info(f"  - Chunk Size: {chunk_size}")
        logger.info(f"  - Chunk Overlap: {chunk_overlap}")
        logger.info(f"  - Vector Store: {vector_store_type}")
        logger.info(f"  - Persist Directory: {persist_directory}")
        
        # Initialize ingestion pipeline
        logger.info("\nInitializing ingestion pipeline...")
        self.ingestion_pipeline = ResearchPaperIngestionPipeline(
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            vector_store_type=vector_store_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist_directory=persist_directory
        )
        
        self.rag_system = None
        self.current_paper_info = None
        
        # Initialize chat manager
        self.chat_manager = ChatManager()
        logger.info(f"Chat Manager initialized")
        
        logger.info("="*80 + "\n")
    
    def ingest_paper(self, pdf_path: str) -> dict:
        """
        Ingest a research paper
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("="*80)
        logger.info("STARTING PAPER INGESTION")
        logger.info("="*80)
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"PDF Path: {pdf_path}")
        logger.info(f"File Size: {os.path.getsize(pdf_path) / 1024:.2f} KB")
        
        # Ingest the paper
        stats = self.ingestion_pipeline.ingest_paper(pdf_path)
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        self._initialize_rag_system()
        
        # Store paper info
        self.current_paper_info = self.ingestion_pipeline.get_paper_info()
        
        logger.info("Agent ready for questions")
        logger.info("="*80 + "\n")
        
        return stats
    
    def load_existing_paper(self):
        """Load an existing ingested paper from disk"""
        logger.info("Loading existing vector store...")
        
        self.ingestion_pipeline.load_vector_store()
        self._initialize_rag_system()
        
        logger.info("Vector store loaded successfully")
    
    def _initialize_rag_system(self):
        """Initialize the RAG system with the vector store"""
        vector_store = self.ingestion_pipeline.get_vector_store()
        
        if vector_store is None:
            raise ValueError("Vector store not initialized. Please ingest a paper first.")
        
        self.rag_system = ResearchPaperRAG(
            vector_store=vector_store,
            model_name=self.llm_model,
            temperature=self.temperature,
            retrieval_strategy=self.retrieval_strategy,
            top_k=self.top_k
        )
    
    def ask(self, question: str, show_sources: bool = False) -> dict:
        """
        Ask a question about the paper
        
        Args:
            question: Question to ask
            show_sources: Whether to show source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        if self.rag_system is None:
            raise ValueError("RAG system not initialized. Please ingest or load a paper first.")
        
        if show_sources:
            return self.rag_system.ask_with_sources(question)
        else:
            return self.rag_system.ask(question, return_source_documents=False)
    
    def interactive_mode(self):
        """Run in interactive mode with chat session management"""
        print("\n" + "="*80)
        print("Research Paper Q&A Agent - Interactive Mode")
        print("="*80)
        
        # Check for existing sessions
        sessions = self.chat_manager.list_sessions()
        
        if sessions:
            print("\nFound existing chat sessions:")
            for i, session in enumerate(sessions[:5], 1):  # Show last 5
                print(f"  {i}. [{session['session_id']}] {session['paper_title']}")
                print(f"     Last updated: {session['updated_at']}, Messages: {session['num_messages']}")
            
            choice = input("\nLoad existing session? (Enter number, 'new' for new chat, or Enter to skip): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(sessions[:5]):
                session_id = sessions[int(choice)-1]['session_id']
                loaded_session = self.chat_manager.load_session(session_id)
                if loaded_session:
                    # Restore chat history to RAG system
                    self.rag_system.chat_history = loaded_session.chat_history.copy()
                    print(f"\nLoaded session: {session_id}")
                    print(f"Restored {len(loaded_session.chat_history)} previous messages")
            elif choice.lower() == 'new':
                paper_title = self.current_paper_info['title'] if self.current_paper_info else "Unknown"
                self.chat_manager.create_new_session(paper_title)
                print(f"\nCreated new chat session: {self.chat_manager.current_session.session_id}")
        else:
            # No existing sessions, create new
            paper_title = self.current_paper_info['title'] if self.current_paper_info else "Unknown"
            self.chat_manager.create_new_session(paper_title)
            print(f"\nCreated new chat session: {self.chat_manager.current_session.session_id}")
        
        if self.current_paper_info:
            print(f"\nCurrent Paper: {self.current_paper_info['title']}")
            print(f"Authors: {', '.join(self.current_paper_info['authors'][:3])}")
            print(f"Sections: {len(self.current_paper_info['sections'])}")
        
        current_session = self.chat_manager.get_current_session()
        if current_session:
            print(f"Active Chat: {current_session.session_id} ({len(current_session.chat_history)} messages)")
        
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'summary' for a paper summary")
        print("  - Type 'methodology' to see the methodology")
        print("  - Type 'results' to see key results")
        print("  - Type 'contributions' to see main contributions")
        print("  - Type 'info' to see paper information")
        print("  - Type 'sessions' to list all chat sessions")
        print("  - Type 'new' to start a new chat session")
        print("  - Type 'load <session_id>' to load a different session")
        print("  - Type 'sources' to toggle source display")
        print("  - Type 'clear' to clear current conversation history")
        print("  - Type 'quit' or 'exit' to quit")
        print("\n" + "="*80 + "\n")
        
        show_sources = False
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'info':
                    self._display_paper_info()
                    continue
                
                if question.lower() == 'sources':
                    show_sources = not show_sources
                    print(f"Source display: {'ON' if show_sources else 'OFF'}")
                    continue
                
                if question.lower() == 'clear':
                    self.rag_system.clear_history()
                    if self.chat_manager.current_session:
                        self.chat_manager.clear_current_history()
                    print("Conversation history cleared.")
                    continue
                
                if question.lower() == 'sessions':
                    self._display_sessions()
                    continue
                
                if question.lower() == 'new':
                    paper_title = self.current_paper_info['title'] if self.current_paper_info else "Unknown"
                    self.chat_manager.create_new_session(paper_title)
                    self.rag_system.clear_history()
                    print(f"Started new chat session: {self.chat_manager.current_session.session_id}")
                    continue
                
                if question.lower().startswith('load '):
                    session_id = question[5:].strip()
                    loaded_session = self.chat_manager.load_session(session_id)
                    if loaded_session:
                        self.rag_system.chat_history = loaded_session.chat_history.copy()
                        print(f"Loaded session: {session_id} with {len(loaded_session.chat_history)} messages")
                    else:
                        print(f"Session not found: {session_id}")
                    continue
                
                # Handle special commands
                if question.lower() == 'summary':
                    answer = self.rag_system.get_paper_summary()
                    print(f"\n{answer}\n")
                    continue
                
                if question.lower() == 'methodology':
                    result = self.rag_system.find_methodology()
                    print(f"\n{result['answer']}\n")
                    if show_sources:
                        self._display_sources(result['sources'])
                    continue
                
                if question.lower() == 'results':
                    result = self.rag_system.find_key_results()
                    print(f"\n{result['answer']}\n")
                    if show_sources:
                        self._display_sources(result['sources'])
                    continue
                
                if question.lower() == 'contributions':
                    result = self.rag_system.find_contributions()
                    print(f"\n{result['answer']}\n")
                    if show_sources:
                        self._display_sources(result['sources'])
                    continue
                
                # Regular question - use conversational mode for context awareness
                result = self.rag_system.conversational_ask(question)
                print(f"\n{result['answer']}\n")
                
                # Save to chat session
                if self.chat_manager.current_session:
                    self.chat_manager.add_to_current(question, result['answer'])
                
                if show_sources and 'sources' in result:
                    self._display_sources(result['sources'])
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}\n")
    
    def _display_paper_info(self):
        """Display information about the current paper"""
        if not self.current_paper_info:
            print("No paper information available")
            return
        
        print("\n" + "="*80)
        print("PAPER INFORMATION")
        print("="*80)
        print(f"Title: {self.current_paper_info['title']}")
        print(f"Authors: {', '.join(self.current_paper_info['authors'])}")
        print(f"\nAbstract:\n{self.current_paper_info['abstract'][:500]}...")
        print(f"\nSections: {', '.join(self.current_paper_info['sections'])}")
        print(f"References: {self.current_paper_info['num_references']}")
        print(f"Figures: {self.current_paper_info['num_figures']}")
        print(f"Tables: {self.current_paper_info['num_tables']}")
        print("="*80 + "\n")
    
    def _display_sources(self, sources: list):
        """Display source documents"""
        print("\nSources:")
        print("-" * 80)
        for i, source in enumerate(sources, 1):
            print(f"{i}. Section: {source['section']} (Page {source['page_number']})")
            print(f"   Type: {source['doc_type']}")
            print(f"   Preview: {source['content_preview']}")
            print()
    
    def _display_sessions(self):
        """Display all available chat sessions"""
        sessions = self.chat_manager.list_sessions()
        
        if not sessions:
            print("\nNo chat sessions found.")
            return
        
        print("\n" + "="*80)
        print("AVAILABLE CHAT SESSIONS")
        print("="*80)
        
        current = self.chat_manager.current_session
        current_id = current.session_id if current else None
        
        for session in sessions:
            is_current = " (CURRENT)" if session['session_id'] == current_id else ""
            print(f"\nSession ID: {session['session_id']}{is_current}")
            print(f"Paper: {session['paper_title']}")
            print(f"Created: {session['created_at']}")
            print(f"Updated: {session['updated_at']}")
            print(f"Messages: {session['num_messages']}")
            if session['num_messages'] > 0:
                print(f"Last Q: {session['last_question'][:80]}...")
        
        print("\n" + "="*80)
        print(f"Total sessions: {len(sessions)}")
        print("="*80 + "\n")
        print("-" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Research Paper Q&A Agent")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to the research paper PDF"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing vector store instead of ingesting new paper"
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Ask a single question and exit"
    )
    parser.add_argument(
        "--strategy",
        choices=["similarity", "mmr", "compression", "multi_query"],
        default="similarity",
        help="Retrieval strategy (default: similarity)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show source documents in answers"
    )
    parser.add_argument(
        "--persist-dir",
        default="./vector_store",
        help="Directory to persist vector store (default: ./vector_store)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("RESEARCH PAPER Q&A AGENT")
    logger.info("="*80)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found in environment variables")
        sys.exit(1)
    
    logger.info("Google API key found")
    logger.info(f"Initializing agent with strategy: {args.strategy}")
    agent = ResearchPaperAgent(
        retrieval_strategy=args.strategy,
        top_k=args.top_k,
        persist_directory=args.persist_dir
    )
    
    # Load or ingest paper
    if args.load:
        agent.load_existing_paper()
    elif args.pdf_path:
        stats = agent.ingest_paper(args.pdf_path)
    else:
        logger.error("Please provide a PDF path or use --load to load existing vector store")
        parser.print_help()
        sys.exit(1)
    
    # Single question mode or interactive mode
    if args.question:
        result = agent.ask(args.question, show_sources=args.show_sources)
        print(f"\nAnswer: {result['answer']}\n")
        
        if args.show_sources and 'sources' in result:
            agent._display_sources(result['sources'])
    else:
        agent.interactive_mode()


if __name__ == "__main__":
    main()
