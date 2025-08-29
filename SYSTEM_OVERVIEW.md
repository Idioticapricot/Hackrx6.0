# LegalEase AI - System Overview

## 🏗️ Architecture

**Frontend**: Static HTML/CSS/JS with 3 tabs (Simplified View, Risk Analysis, Q&A)
**Backend**: FastAPI with async processing
**AI**: Google Gemini 1.5 Flash + MiniLM embeddings
**Strategy**: Hybrid approach - Full context for analysis, RAG for Q&A

## 🤖 AI Models Used

### Primary LLM
- **Google Gemini 1.5 Flash** - Document analysis, simplification, risk detection, Q&A
- **Context Window**: ~1M tokens (allows full document processing)
- **API**: Google Generative AI REST API

### Embeddings & Retrieval
- **all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`) - 384-dim lightweight embeddings
- **Jina Reranker v1 Tiny** (`jinaai/jina-reranker-v1-tiny-en`) - Compact reranking model
- **FAISS** - Vector similarity search
- **Device**: CPU optimized for MacBook (batch_size: 16)

## 📊 Processing Flow

### Document Analysis (Simplified View + Risk Analysis)
```
Document URL → Download → Extract Text → Send FULL CONTEXT to Gemini → Analysis
```
- Uses entire document content (up to ~8000 chars sent to API)
- Single API call for comprehensive analysis
- Better for understanding document structure and relationships

### Q&A Processing
```
Document URL → Chunk → Embed → FAISS Index → Query → Retrieve → Rerank → Answer
```
- Traditional RAG pipeline for targeted question answering
- Retrieves top-10 relevant chunks per question
- More efficient for specific questions

## ⚙️ Key Configuration

### API Settings
- **Model**: `gemini-1.5-flash`
- **Max Output Tokens**: 2048 (bottleneck for long simplifications)
- **Temperature**: 0.3
- **Safety Filters**: Disabled for legal content

### Processing
- **Chunk Size**: 800 tokens, 200 overlap
- **Small Doc Threshold**: 5000 tokens (bypass RAG)
- **Batch Size**: 16 (CPU optimized)

### Rate Limits
- **LLM Calls**: 20 concurrent
- **Image Calls**: 1 concurrent

## 🔄 Request Flow

1. **Document Upload** → URL validation
2. **Document Processing** → Download, extract, chunk
3. **Strategy Selection**:
   - **Legal Analysis**: Full context → Gemini
   - **Q&A**: RAG pipeline → Context retrieval → Gemini
4. **Response Generation** → Format and return

## 🚨 Known Issues

### Token Limit Bottleneck
- **Problem**: 2048 max tokens insufficient for document simplification
- **Impact**: Truncated responses appear as "content blocked"
- **Solution**: Increase to 8192+ tokens for simplification

### Safety Filters
- **Disabled**: All Gemini safety categories set to `BLOCK_NONE`
- **Reason**: Legal content often triggers false positives

## 🎯 Optimization Points

1. **Token Limits**: Different limits per endpoint type
2. **Chunking**: Optimize chunk size for legal documents
3. **Caching**: Persistent embeddings and response caching
4. **Error Handling**: Better distinction between truncation vs blocking

## 📁 File Structure

```
src/
├── ai/                 # Gemini client, prompts
├── api/                # FastAPI endpoints
├── core/               # Configuration (heart of system)
├── document_processing/ # Loaders, chunking
├── models/             # Pydantic schemas
├── utils/              # Legal analyzer, helpers
└── static/             # Frontend files
```

## 🔧 Quick Fixes Needed

1. **Increase maxOutputTokens** to 8192 for simplification
2. **Add response length validation** before sending to frontend
3. **Implement chunked simplification** for very large documents