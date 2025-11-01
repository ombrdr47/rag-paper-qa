"""
Data Ingestion Pipeline for Research Papers

This module implements a smart data ingestion pipeline that:
1. Parses research papers with structure awareness
2. Chunks content intelligently based on document structure
3. Creates embeddings and stores them in a vector database
4. Maintains document hierarchy and metadata
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from pdf_parser import ResearchPaper, Section, parse_research_paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StructureAwareTextSplitter:
    """
    Custom text splitter that respects document structure
    Splits based on sections, subsections, and paragraphs
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_structure: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_structure = respect_structure
        
        # Fallback splitter for large sections
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_sections(self, paper: ResearchPaper) -> List[Document]:
        """
        Split research paper into chunks based on structure
        
        Args:
            paper: Parsed research paper
            
        Returns:
            List of Document objects with metadata
        """
        documents = []
        
        # Add title and abstract as separate documents
        if paper.title and paper.abstract:
            documents.append(Document(
                page_content=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
                metadata={
                    'source': 'title_abstract',
                    'section': 'Abstract',
                    'doc_type': 'research_paper',
                    'title': paper.title,
                    'authors': ', '.join(paper.authors)
                }
            ))
        
        # Process each section
        for section in paper.sections:
            section_docs = self._process_section(section, paper)
            documents.extend(section_docs)
        
        # Add references as a separate document
        if paper.references:
            ref_chunks = self._chunk_references(paper.references)
            for i, chunk in enumerate(ref_chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        'source': 'references',
                        'section': 'References',
                        'doc_type': 'references',
                        'chunk_id': i,
                        'title': paper.title
                    }
                ))
        
        # Add figure and table information
        for fig in paper.figures:
            documents.append(Document(
                page_content=f"Figure on page {fig.page_number}: {fig.caption}",
                metadata={
                    'source': 'figure',
                    'section': 'Figures',
                    'doc_type': 'figure',
                    'page_number': fig.page_number,
                    'title': paper.title
                }
            ))
        
        for table in paper.tables:
            documents.append(Document(
                page_content=f"Table on page {table.page_number}: {table.caption}",
                metadata={
                    'source': 'table',
                    'section': 'Tables',
                    'doc_type': 'table',
                    'page_number': table.page_number,
                    'title': paper.title
                }
            ))
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents
    
    def _process_section(self, section: Section, paper: ResearchPaper) -> List[Document]:
        """Process a single section into document chunks"""
        documents = []
        
        # If section is small enough, keep it as one chunk
        if len(section.content) <= self.chunk_size:
            documents.append(Document(
                page_content=f"Section: {section.title}\n\n{section.content}",
                metadata={
                    'source': 'section',
                    'section': section.title,
                    'doc_type': 'section',
                    'page_number': section.page_number,
                    'level': section.level,
                    'title': paper.title
                }
            ))
        else:
            # Split large sections
            chunks = self._split_text_with_overlap(section.content)
            
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=f"Section: {section.title}\n\n{chunk}",
                    metadata={
                        'source': 'section',
                        'section': section.title,
                        'doc_type': 'section',
                        'page_number': section.page_number,
                        'level': section.level,
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'title': paper.title
                    }
                ))
        
        return documents
    
    def _split_text_with_overlap(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    def _chunk_references(self, references: List[str]) -> List[str]:
        """Group references into manageable chunks"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for ref in references:
            ref_length = len(ref)
            
            if current_length + ref_length > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(ref)
            current_length += ref_length
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks


class ResearchPaperIngestionPipeline:
    """
    Complete ingestion pipeline for research papers
    """
    
    def __init__(
        self,
        embedding_model: str = "models/embedding-001",
        embedding_provider: str = "huggingface",
        vector_store_type: str = "faiss",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the ingestion pipeline
        
        Args:
            embedding_model: Embedding model name
            embedding_provider: 'google' or 'huggingface'
            vector_store_type: Type of vector store ('faiss' or 'chroma')
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist vector store
        """
        logger.info("="*80)
        logger.info("Initializing Research Paper Ingestion Pipeline")
        logger.info("="*80)
        logger.info(f"Embedding Provider: {embedding_provider}")
        logger.info(f"Embedding Model: {embedding_model}")
        logger.info(f"Vector Store Type: {vector_store_type}")
        logger.info(f"Chunk Size: {chunk_size}")
        logger.info(f"Chunk Overlap: {chunk_overlap}")
        logger.info(f"Persist Directory: {persist_directory or './vector_store'}")
        
        # Initialize embeddings based on provider
        if embedding_provider.lower() == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            logger.info("HuggingFace Embeddings initialized ")
        elif embedding_provider.lower() == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            logger.info("Google Generative AI Embeddings initialized")
        else:
            raise ValueError(f"Unknown embedding provider: {embedding_provider}")
        
        self.vector_store_type = vector_store_type
        self.persist_directory = persist_directory or "./vector_store"
        
        self.text_splitter = StructureAwareTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info("Structure-aware text splitter initialized")
        
        self.vector_store = None
        self.paper = None
        self.documents = None  # Store documents for hybrid retrieval
        logger.info("="*80)
    
    def ingest_paper(self, pdf_path: str) -> Dict[str, Any]:
        """
        Ingest a research paper into the vector store
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("="*80)
        logger.info(f"Starting ingestion pipeline for: {pdf_path}")
        logger.info("="*80)
        
        # Step 1: Parse the PDF
        logger.info("Step 1/3: Parsing PDF document")
        try:
            self.paper = parse_research_paper(pdf_path)
            logger.info(f"Successfully parsed PDF")
            logger.info(f"  Title: {self.paper.title[:100]}")
            logger.info(f"  Authors: {len(self.paper.authors)}")
            logger.info(f"  Sections: {len(self.paper.sections)}")
            logger.info(f"  References: {len(self.paper.references)}")
            logger.info(f"  Figures: {len(self.paper.figures)}")
            logger.info(f"  Tables: {len(self.paper.tables)}")
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise
        
        # Step 2: Create document chunks
        logger.info("Step 2/3: Creating document chunks")
        try:
            documents = self.text_splitter.split_sections(self.paper)
            self.documents = documents  # Store for hybrid retrieval
            logger.info(f"Created {len(documents)} document chunks")
            
            # Log chunk type distribution
            chunk_types = {}
            for doc in documents:
                doc_type = doc.metadata.get('doc_type', 'unknown')
                chunk_types[doc_type] = chunk_types.get(doc_type, 0) + 1
            
            logger.info("  Chunk distribution:")
            for doc_type, count in chunk_types.items():
                logger.info(f"    {doc_type}: {count}")
        except Exception as e:
            logger.error(f"Failed to create chunks: {e}")
            raise
        
        # Step 3: Create embeddings and store in vector database
        logger.info("Step 3/3: Creating embeddings and building vector store")
        logger.info(f"  Total chunks to embed: {len(documents)}")
        
        try:
            if self.vector_store_type == "faiss":
                logger.info(f"  Using FAISS vector store")
                
                # Process in batches with rate limiting to avoid quota issues
                batch_size = 20  # Process 20 documents at a time (faster with HuggingFace)
                delay_between_batches = 0.1  # 0.1 seconds between batches (HuggingFace is local, no API limits)
                
                if len(documents) > batch_size:
                    logger.info(f"  Processing in batches of {batch_size} with {delay_between_batches}s delays")
                    logger.info(f"  Total batches: {(len(documents) + batch_size - 1) // batch_size}")
                    
                    # Create vector store with first batch
                    first_batch = documents[:batch_size]
                    logger.info(f"  Creating vector store with first batch ({len(first_batch)} docs)...")
                    self.vector_store = FAISS.from_documents(first_batch, self.embeddings)
                    
                    # Add remaining batches
                    for i in range(batch_size, len(documents), batch_size):
                        batch_num = (i // batch_size) + 1
                        batch = documents[i:i + batch_size]
                        logger.info(f"  Processing batch {batch_num} ({len(batch)} docs)")
                        
                        # Add delay between batches to respect rate limits
                        time.sleep(delay_between_batches)
                        
                        # Add batch to existing vector store
                        self.vector_store.add_documents(batch)
                        logger.info(f"  Batch {batch_num} completed")
                else:
                    # If documents fit in one batch, process normally
                    logger.info(f"  Processing all {len(documents)} documents in single batch...")
                    self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                logger.info(f"  Saving to: {self.persist_directory}")
                self.vector_store.save_local(self.persist_directory)
                logger.info(f"FAISS vector store created and saved")
                
            elif self.vector_store_type == "chroma":
                logger.error("Chroma is not installed. Please install chromadb or use 'faiss' instead.")
                raise ValueError("Chroma vector store requires 'pip install chromadb'")
            else:
                raise ValueError(f"Unknown vector store type: {self.vector_store_type}")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            if "429" in str(e) or "quota" in str(e).lower():
                logger.error("  API quota exceeded. Please:")
                logger.error("     1. Wait for quota to reset (usually daily)")
                logger.error("     2. Check usage at: https://ai.dev/usage?tab=rate-limit")
                logger.error("     3. Consider upgrading your API plan")
                logger.error(f"     4. Current batch size: {batch_size} - you can reduce this further")
            raise
        
        logger.info("="*80)
        logger.info("INGESTION COMPLETE")
        logger.info("="*80)
        
        return {
            'title': self.paper.title,
            'authors': self.paper.authors,
            'num_sections': len(self.paper.sections),
            'num_references': len(self.paper.references),
            'num_figures': len(self.paper.figures),
            'num_tables': len(self.paper.tables),
            'num_chunks': len(documents),
            'vector_store_type': self.vector_store_type,
            'persist_directory': self.persist_directory
        }
    
    def load_vector_store(self) -> None:
        """Load an existing vector store from disk"""
        logger.info("="*80)
        logger.info("Loading existing vector store")
        logger.info("="*80)
        logger.info(f"Vector Store Type: {self.vector_store_type}")
        logger.info(f"Directory: {self.persist_directory}")
        
        try:
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded FAISS vector store")
                
                # Extract documents from vector store for hybrid retrieval
                if hasattr(self.vector_store, 'docstore') and hasattr(self.vector_store.docstore, '_dict'):
                    self.documents = list(self.vector_store.docstore._dict.values())
                    logger.info(f"Extracted {len(self.documents)} documents from vector store")
                else:
                    logger.warning("Could not extract documents from vector store")
                    self.documents = None
                    
            elif self.vector_store_type == "chroma":
                logger.error("Chroma is not installed. Please install chromadb or use 'faiss' instead.")
                raise ValueError("Chroma vector store requires 'pip install chromadb'")
            else:
                raise ValueError(f"Unknown vector store type: {self.vector_store_type}")
            
            logger.info("="*80)
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def get_vector_store(self):
        """Get the vector store instance"""
        return self.vector_store
    
    def get_documents(self) -> Optional[List[Document]]:
        """Get the list of documents for hybrid retrieval"""
        return self.documents
    
    def get_paper_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the ingested paper"""
        if not self.paper:
            return None
        
        return {
            'title': self.paper.title,
            'authors': self.paper.authors,
            'abstract': self.paper.abstract,
            'sections': [s.title for s in self.paper.sections],
            'num_references': len(self.paper.references),
            'num_figures': len(self.paper.figures),
            'num_tables': len(self.paper.tables)
        }
