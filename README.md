# Research Paper Q&A Agent

A RAG (Retrieval-Augmented Generation) system built with LangChain for intelligent question-answering on research papers. Features conversational memory, multi-session chat management, and structure-aware PDF parsing.

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Research Paper Q&A Agent                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐     ┌──────────────┐
│ PDF Parser   │    │  RAG System  │     │Chat Manager  │
│              │    │              │     │              │
│ - Structure  │    │ - Retrieval  │     │ - Sessions   │
│   Detection  │───▶│ - Generation │     │ - History    │
│ - Metadata   │    │ - Context    │     │ - Persistence│
│   Extraction │    │              │     │              │
└──────────────┘    └──────────────┘     └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐     ┌──────────────┐
│Data Ingestion│    │Vector Store  │     │   Storage    │
│              │    │              │     │              │
│ - Chunking   │───▶│ - FAISS      │     │ - JSON       │
│ - Embeddings │    │ - Similarity │     │ - Sessions   │
│ - Batching   │    │   Search     │     │              │
└──────────────┘    └──────────────┘     └──────────────┘
```

### Core Components

#### 1. **PDF Parser** (`pdf_parser.py`)
- **Purpose**: Structure-aware extraction of research paper content
- **Features**:
  - Identifies title, authors, abstract, sections
  - Extracts figures, tables, and references
  - Maintains document hierarchy and metadata
- **Technology**: PyMuPDF (fitz)
- **Error Handling**: Custom `PDFParsingError` exceptions

#### 2. **Data Ingestion Pipeline** (`data_ingestion.py`)
- **Purpose**: Transform PDF content into searchable vector embeddings
- **Process**:
  1. Receives parsed document structure
  2. Splits text into semantic chunks (1000 chars, 200 overlap)
  3. Generates embeddings using HuggingFace models
  4. Stores in FAISS vector database
- **Optimization**: Batch processing (20 docs/batch) with minimal delays
- **Configuration**: Centralized in `config.py`

#### 3. **RAG System** (`rag_system.py`)
- **Purpose**: Retrieval-Augmented Generation for accurate Q&A
- **Features**:
  - **Multiple Retrieval Strategies**:
    - Similarity search (default) - Dense vector similarity
    - Maximum Marginal Relevance (MMR) - Diversity-aware retrieval
    - Contextual compression - Context-aware filtering
    - Multi-query retrieval - Query expansion
    - **Hybrid search (NEW)** - BM25 + Dense embeddings with RRF
  - **Conversational Context**: Maintains chat history for follow-up questions
  - **Specialized Queries**: Summary, methodology, results extraction
- **Technology**: LangChain + Google Gemini LLM
- **Performance**: No extra LLM calls for context (fast context injection)

#### 4. **Chat Manager** (`chat_manager.py`)
- **Purpose**: Multi-session conversation management
- **Features**:
  - Create/load/save chat sessions
  - Persistent storage (JSON)
  - Session metadata tracking
  - Resume conversations from any point
- **Storage**: File-based with automatic timestamps
- **Error Handling**: Custom `ChatSessionError` exceptions

#### 5. **Hybrid Retriever** (`hybrid_retriever.py`) ⭐ NEW
- **Purpose**: retrieval combining lexical and semantic search
- **Algorithm**: Reciprocal Rank Fusion (RRF)
- **Components**:
  - **BM25 Retriever**: Keyword/term matching (sparse retrieval)
  - **Dense Retriever**: Semantic similarity (FAISS vector search)
  - **RRF Scoring**: `score = bm25_weight/(rank+60) + dense_weight/(rank+60)`
- **Benefits**:
  - Captures exact keyword matches (BM25)
  - Understands semantic meaning (dense embeddings)
  - Handles both technical terms and conceptual queries
- **Configuration**: Adjustable weights (default: 0.5/0.5)

#### 6. **Main Application** (`main.py`)
- **Purpose**: CLI interface and orchestration
- **Modes**:
  - **Single Query**: `python main.py <pdf> -q "question"`
  - **Interactive**: `python main.py --load` (chat interface)
- **Features**:
  - Session management UI
  - Source attribution toggle
  - Special commands (summary, methodology, etc.)
  - Multiple retrieval strategies via `--strategy` flag
  - Comprehensive error handling

#### 7. **Configuration System** (`config.py`, `logger.py`, `exceptions.py`)
- **config.py**: Centralized constants and default values
- **logger.py**: Structured logging with file/console handlers
- **exceptions.py**: Custom exception hierarchy for precise error handling

## Technology Stack

### Embeddings
- **Provider**: HuggingFace Transformers (local, no API quota)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Advantage**: Runs on local hardware (CPU/MPS), fast inference

### Retrieval
- **Dense Search**: FAISS vector similarity (semantic)
- **Sparse Search**: BM25 algorithm (keyword matching via `rank-bm25`)
- **Hybrid**: Reciprocal Rank Fusion combining both

### LLM
- **Provider**: Google Gemini
- **Model**: `gemini-2.5-pro`
- **Temperature**: 0.0 (deterministic)
- **Use**: Text generation for answers

### Vector Store
- **Technology**: FAISS (Facebook AI Similarity Search)
- **Index Type**: Flat L2
- **Persistence**: Local disk storage

### Framework
- **LangChain**: Core RAG orchestration
  - `langchain-core`: Base abstractions
  - `langchain-community`: Vector stores, retrievers
  - `langchain-classic`: QA chains
  - `langchain-google-genai`: Gemini integration
  - `langchain-huggingface`: Local embeddings

## Setup

### Prerequisites
- Python 3.12+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone/Download the repository**
   ```bash
   git clone https://github.com/ombrdr47/rag-paper-qa
   cd rag-paper-qa
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   
   **Option A: Standard installation**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Option B: Package installation (development mode)**
   ```bash
   pip install -e .
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

5. **Verify installation**
   ```bash
   python main.py --help
   ```

## Usage

### Process a Research Paper

```bash
python main.py data/your_paper.pdf
```

This will:
- Parse the PDF structure
- Create embeddings (local HuggingFace)
- Build FAISS vector store
- Save to `./vector_store/` and `./chat_sessions/`

### Single Question

```bash
python main.py data/paper.pdf -q "What is the main contribution?"
```

### Retrieval Strategies

```bash
# Use hybrid search (BM25 + dense embeddings)
python main.py --load --strategy hybrid -q "What is attention?"

# Use MMR for diverse results
python main.py --load --strategy mmr -q "Explain the methodology"

# Use contextual compression
python main.py --load --strategy compression -q "What are the results?"

# Use multi-query for comprehensive answers
python main.py --load --strategy multi_query -q "How does it work?"
```

**Available Strategies:**
- `similarity` - Dense vector search (default)
- `mmr` - Maximum Marginal Relevance for diversity
- `compression` - Contextual compression filtering
- `multi_query` - Expands single query to multiple perspectives
- `hybrid` - Combines BM25 (keyword) + dense (semantic) with Reciprocal Rank Fusion

### Interactive Chat Mode

```bash
python main.py --load
```

**Available Commands:**
- Type questions naturally
- `summary` - Generate paper summary
- `methodology` - Extract methodology details
- `results` - Get key results and findings
- `contributions` - List main contributions
- `sessions` - View all chat sessions
- `new` - Start new chat session
- `load <session_id>` - Resume previous chat
- `sources` - Toggle source attribution
- `clear` - Clear current conversation
- `quit` - Exit

### Chat Session Management

**Example Workflow:**
```bash
# Start interactive mode
python main.py --load

# System shows existing sessions:
# 1. [20251023_162001] Attention is All You Need
#    Last updated: 2025-10-23 16:25:30, Messages: 5

# Load existing: Enter "1"
# New session: Enter "new"
# Skip: Press Enter

# Chat naturally with context:
You: What is the Transformer?
AI: [Detailed answer...]

You: What are its advantages?  # "its" understood from context
AI: [Contextual answer...]

# Sessions auto-saved, resume anytime
```

## Project Structure

```
.
├── main.py                 # CLI application & orchestration
├── cli.py                  # Command-line entry point
├── pdf_parser.py           # Structure-aware PDF parsing
├── data_ingestion.py       # Embedding generation & vector store
├── rag_system.py           # RAG with multiple retrieval strategies
├── hybrid_retriever.py     # Hybrid search (BM25 + Dense) ⭐ NEW
├── chat_manager.py         # Session management & persistence
├── config.py               # Configuration constants
├── logger.py               # Centralized logging setup
├── exceptions.py           # Custom exception types
├── __init__.py             # Package initialization
├── requirements.txt        # Python dependencies (includes rank-bm25)
├── .env.example           # Environment configuration template
├── data/                  # Research papers (PDF files)
├── vector_store/          # FAISS index & metadata
└── chat_sessions/         # Saved conversation history (JSON)
```




**Built with LangChain | Powered by Google Gemini & HuggingFace**
