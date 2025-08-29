# 🚀 Quick Setup Guide - Gemini RAG Pipeline

## Prerequisites

- **Python 3.11+** (recommended for best performance)
- **macOS** (optimized for Apple Silicon)
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))

## 🎯 One-Command Setup

```bash
./setup.sh
```

## 📋 Manual Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 4. Run the Server
```bash
python main.py
```

## 🔑 API Key Setup

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to your `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## 🧪 Test the Setup

```bash
curl -X POST "http://localhost:8000/hackathon" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

## 🔧 Migration from Claude

If you're migrating from the previous Claude-based version:

1. **API Key**: Replace `OPENROUTER_API_KEY` with `GEMINI_API_KEY`
2. **Models**: System now uses Gemini 1.5 Pro for both text and vision
3. **Performance**: Optimized for CPU/MacBook usage
4. **Batch Size**: Reduced to 16 for better CPU performance

## 📊 System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for models and cache
- **CPU**: Apple Silicon recommended (M1/M2/M3)
- **Network**: Stable internet for API calls

## 🛠️ Troubleshooting

### Common Issues

**Import Error**: Ensure virtual environment is activated
```bash
source venv/bin/activate
```

**API Key Error**: Verify your Gemini API key is valid
```bash
echo $GEMINI_API_KEY  # Should show your key
```

**Memory Issues**: Reduce batch size in config
```python
config.update_config('models', batch_size=8)
```

## 🚀 Performance Tips

1. **Apple Silicon**: Use native Python build for best performance
2. **Memory**: Close other applications during heavy processing
3. **Batch Size**: Adjust based on your MacBook's RAM
4. **Cache**: Enable persistent caching for faster repeated queries

## 📖 Next Steps

- Check out the [API Documentation](http://localhost:8000/docs)
- Explore the [Configuration Guide](README.md#configuration-system)
- Try different document formats (PDF, DOCX, PPTX, etc.)
- Experiment with hybrid search weights