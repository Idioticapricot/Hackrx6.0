# ⚖️ LegalEase AI - Legal Document Simplifier

An AI-powered platform that demystifies complex legal documents, making them accessible to everyone. Built with Google Gemini and advanced RAG technology to protect users from unfavorable terms and legal risks.

## 🚀 Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Set your Gemini API key**: Add `GEMINI_API_KEY` to `.env`
3. **Run the server**: `python main.py`
4. **Open browser**: `http://localhost:8000`
5. **Upload legal document**: Drag & drop or paste URL
6. **Get insights**: View simplified version with risk highlights

**Detailed setup:** [📖 Setup Guide](SETUP.md)

## 🛠️ Tech Stack

### Core Framework
- **FastAPI** - High-performance async web framework
- **Python 3.11** - Modern Python with enhanced performance
- **Uvicorn** - Lightning-fast ASGI server

### AI/ML Components
- **Google Gemini 1.5 Pro** - Legal document analysis and simplification
- **BGE-M3** (`BAAI/bge-m3`) - Multilingual legal text embeddings
- **BGE Reranker v2-M3** - Semantic reranking for legal clauses
- **Advanced Risk Detection** - AI-powered identification of unfavorable terms
- **Multi-language Support** - Legal explanations in 8+ languages

### Vector Database & Search
- **FAISS** - Facebook AI Similarity Search for vector operations
- **Hybrid Search** - Combines semantic similarity + BM25 keyword matching
- **Advanced Caching** - Persistent embeddings and response caching

### Document Processing
- **PyMuPDF** - High-performance PDF processing
- **python-docx** - Microsoft Word document handling
- **python-pptx** - PowerPoint presentation processing
- **openpyxl** - Excel spreadsheet processing
- **BeautifulSoup4** - HTML/XML parsing
- **LangChain** - Document loading and text splitting

### Infrastructure
- **Docker** - Containerized deployment
- **Conda** - Environment management
- **Rich** - Enhanced terminal UI
- **Pydantic** - Data validation and settings management

## 🏛️ System Architecture & Modularity

### 📦 Modular Design
The system follows a **clean, modular architecture** with clear separation of concerns:

```
src/
├── 🧠 ai/              # AI models & prompts
├── 🌐 api/             # FastAPI endpoints & server
├── ⚙️  core/           # Configuration hub (❤️ Heart of system)
├── 📄 document_processing/  # Loaders & retrieval
├── 📊 models/          # Pydantic schemas
└── 🛠️  utils/          # Helpers & utilities
```

**Benefits:**
- 🔧 **Easy Maintenance** - Independent module updates
- 🚀 **Scalable Development** - Team can work on different modules
- 🧪 **Testable Components** - Isolated unit testing
- 🔄 **Reusable Code** - Modules can be used across projects

### ⚙️ Configuration System - Heart of the System ❤️

The **`src/core/config.py`** is the **central nervous system** that controls every aspect:

```python
# 🎯 Everything is configurable through centralized config classes
@dataclass
class ModelConfig:
    embedding_model: str = "BAAI/bge-m3"     # Switch embedding models
    llm_model: str = "anthropic/claude-3.5-sonnet"  # Change LLM
    device: str = "cuda"                      # GPU/CPU selection
    batch_size: int = 32                      # Processing batch size

@dataclass 
class RetrievalConfig:
    top_k_retrieval: int = 20                 # Search candidates
    semantic_weight: float = 0.7              # Hybrid search weights
    keyword_weight: float = 0.3               # BM25 influence
    use_reranking: bool = True                 # Enable/disable reranking
```

**🎛️ What You Can Configure:**
- 🤖 **AI Models** - Gemini API settings, embedding models
- 🔍 **Search Parameters** - Tune hybrid search weights & thresholds
- 📊 **Processing Settings** - Chunk sizes, overlap, CPU-optimized batch sizes
- 🚀 **Performance** - Rate limits, timeouts, cache settings (MacBook optimized)
- 🛡️ **Security** - API keys, validation rules, content filters
- 📁 **Storage** - Cache directories, log paths, temp folders

**🔥 Dynamic Configuration:**
```python
# Change settings at runtime
config.update_config('retrieval', semantic_weight=0.8)
config.update_config('models', batch_size=16)  # CPU optimized for MacBook
```

---

## ✨ Key Features

### ⚖️ Legal Document Simplification
- **Plain Language Translation**: Convert legal jargon to everyday English
- **Visual Risk Highlighting**: Red/yellow/green indicators for dangerous clauses
- **Multi-language Support**: Explanations in Spanish, French, German, Hindi, Chinese, etc.
- **Smart Clause Detection**: Automatically identify key terms, fees, penalties

### 🚨 Advanced Risk Detection
- **🔴 High Risk**: Hidden fees, penalties, automatic renewals, unlimited liability
- **🟡 Medium Risk**: Unclear obligations, restrictive conditions, ambiguous terms
- **🟢 Low Risk**: Standard terms, fair conditions, balanced agreements
- **Risk Categories**: Financial, liability, termination, renewal risks

### 📄 Legal Document Support
- **Contracts**: Rental agreements, employment contracts, service agreements
- **Financial**: Loan documents, credit agreements, insurance policies
- **Digital**: Terms of service, privacy policies, software licenses
- **Business**: Partnership agreements, NDAs, vendor contracts

### 🧠 Intelligent Legal Analysis
- **Document Simplification**: AI-powered conversion to plain language
- **Risk Assessment**: Comprehensive analysis of potentially harmful clauses
- **Interactive Q&A**: Ask specific questions about your legal documents
- **Contextual Understanding**: Deep comprehension of legal implications

### 🚀 Performance Optimizations
- **CPU Optimization**: MacBook-optimized processing for embeddings and reranking
- **Async Processing**: Non-blocking I/O operations
- **Rate Limiting**: Configurable API call throttling
- **Memory Management**: Efficient caching and cleanup
- **Batch Embeddings**: Process multiple queries simultaneously with CPU-optimized batch sizes

### 🛡️ Security & Reliability
- **Content Filtering**: Advanced malicious prompt detection
- **Input Validation**: Pydantic-based request validation
- **Error Handling**: Comprehensive exception management
- **Request Logging**: Detailed operation tracking
- **Fallback Mechanisms**: Graceful degradation on failures

## 🏗️ Pipeline Overview

### 1. Document Ingestion
```
URL Input → Download → Format Detection → Loader Selection → Content Extraction
```

### 2. Content Processing
```
Raw Content → Text Cleaning → Chunking (512 tokens, 150 overlap) → Metadata Enrichment
```

### 3. Embedding Generation
```
Text Chunks → BGE-M3 Encoder → 1024-dim Vectors → FAISS Index → Cache Storage
```

### 4. Query Processing
```
Questions → Batch Embedding → Hybrid Search → Candidate Retrieval → Reranking
```

### 5. Legal Analysis
```
Legal Document → Risk Detection → Plain Language Simplification → Visual Highlighting → User Interface
```

## 🔧 System Architecture

### Core Components

**Document Processing Layer**
- Multi-format loaders with specialized parsers
- Intelligent text extraction and cleaning
- Metadata preservation and enrichment

**Embedding & Retrieval Layer**
- BGE-M3 multilingual embeddings (1024 dimensions)
- FAISS vector database with L2 similarity
- Hybrid search combining semantic + keyword matching
- BGE reranker for result optimization

**Legal AI Layer**
- Google Gemini 1.5 Pro for legal document analysis
- Legal-specific prompts for risk detection and simplification
- Multi-language legal translation
- CPU-optimized processing for MacBook

**API & Service Layer**
- FastAPI with async request handling
- Rate limiting and request throttling
- Comprehensive error handling
- Real-time progress tracking

### Configuration Management
Centralized configuration system with:
- Model parameters and device allocation
- Processing thresholds and chunk sizes
- API endpoints and authentication
- Cache settings and cleanup policies
- Hybrid search weights and parameters

### Caching Strategy
- **Embedding Cache**: Persistent vector storage per document
- **Response Cache**: API response caching with TTL
- **Model Cache**: In-memory model instances
- **File Cache**: Temporary document storage with cleanup

## 🚦 Processing Flow

### Standard Document Flow
1. **Input Validation**: URL format and accessibility check
2. **Document Download**: Secure file retrieval with timeout
3. **Format Detection**: Extension-based loader selection
4. **Content Extraction**: Format-specific parsing
5. **Text Processing**: Cleaning, chunking, and metadata addition
6. **Embedding Generation**: BGE-M3 vectorization with CPU optimization for MacBook
7. **Index Creation**: FAISS index construction and caching
8. **Query Processing**: Batch question embedding and hybrid search
9. **Context Retrieval**: Top-k candidates with reranking
10. **Answer Generation**: LLM inference with document-specific prompts

### Special Processing Modes
- **Small Documents**: Direct full-text processing (< 5000 tokens)
- **Image URLs**: Direct vision model processing
- **PPTX Files**: Combined text + AI image analysis
- **Secret Tokens**: Special authentication handling
- **Flight Data**: Custom API integration

### Performance Optimizations
- **Concurrent Processing**: Parallel question handling
- **Memory Management**: CPU memory optimization for MacBook
- **Batch Operations**: Efficient embedding generation
- **Smart Caching**: Multi-level cache hierarchy
- **Resource Pooling**: Connection and thread pool management

## 📊 Key Metrics & Thresholds

- **Chunk Size**: 512 tokens with 150 token overlap
- **Embedding Dimension**: 1024 (BGE-M3)
- **Batch Size**: 16 (CPU optimized for MacBook)
- **Retrieval**: Top-20 candidates → Top-10 after reranking
- **Context Limit**: 10,000 tokens maximum
- **Small Doc Threshold**: 5,000 tokens (bypass RAG)
- **Hybrid Weights**: 70% semantic, 30% keyword
- **Rate Limits**: 20 LLM calls, 1 image call concurrent
- **Cache TTL**: 24 hours with 10GB limit
- **Device**: CPU optimized for MacBook performance
- **Languages**: English, Spanish, French, German, Italian, Portuguese, Hindi, Chinese
- **Risk Categories**: Financial, Liability, Termination, Renewal, General
- **Document Types**: Contracts, Agreements, Terms of Service, Legal PDFs

## 🎯 Real-World Impact

**Protects Users From:**
- Hidden fees and penalties in contracts
- Predatory loan terms and conditions
- Unfair rental agreement clauses
- Automatic renewals and cancellation traps
- Unlimited liability exposure

**Empowers Users With:**
- Plain language explanations of complex terms
- Visual risk indicators for dangerous clauses
- Multi-language accessibility
- Interactive Q&A for specific concerns
- Informed decision-making tools

**Use Cases:**
- 🏠 **Rental Agreements**: Understand lease terms, fees, and tenant rights
- 💰 **Loan Contracts**: Identify interest rates, penalties, and payment terms
- 📱 **Terms of Service**: Decode privacy policies and user agreements
- 💼 **Employment Contracts**: Clarify obligations, benefits, and termination clauses
- 🛡️ **Insurance Policies**: Understand coverage, exclusions, and claim processes

This platform democratizes legal understanding, helping ordinary people protect themselves from unfavorable terms and make informed decisions about important legal documents.