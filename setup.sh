#!/bin/bash

# RAG Pipeline Setup Script - Gemini API & MacBook Optimized
echo "🚀 Setting up RAG Document Q&A System"
echo "🤖 Powered by Google Gemini API"
echo "💻 Optimized for MacBook/CPU usage"
echo "=" * 50

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env and add your GEMINI_API_KEY"
    echo "🔗 Get your API key from: https://makersuite.google.com/app/apikey"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating cache directories..."
mkdir -p document_cache logs temp models

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your GEMINI_API_KEY"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the server: python main.py"
echo ""
echo "🔧 Migration Summary:"
echo "   • LLM: Claude → Gemini 1.5 Pro"
echo "   • Device: GPU → CPU (MacBook optimized)"
echo "   • Batch Size: 32 → 16 (CPU optimized)"
echo "   • All models optimized for Apple Silicon"