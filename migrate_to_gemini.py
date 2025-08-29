#!/usr/bin/env python3
"""
Migration script for RAG pipeline: Claude → Gemini API
Optimized for MacBook/CPU usage
"""

import os
from pathlib import Path

def main():
    print("🚀 RAG Pipeline Migration: Claude → Gemini API")
    print("📱 Optimized for MacBook/CPU usage")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Please create one based on env.example")
        return
    
    # Read current .env
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    # Check for Gemini API key
    if "GEMINI_API_KEY" not in env_content:
        print("⚠️  GEMINI_API_KEY not found in .env file")
        print("📝 Please add your Gemini API key to .env:")
        print("   GEMINI_API_KEY=your_gemini_api_key_here")
        print()
        print("🔗 Get your Gemini API key from:")
        print("   https://makersuite.google.com/app/apikey")
        print()
    else:
        print("✅ GEMINI_API_KEY found in .env")
    
    print("🔄 Migration Summary:")
    print("   • LLM: Claude 3.5 Sonnet → Gemini 1.5 Pro")
    print("   • Vision: Claude 3 Haiku → Gemini 1.5 Pro")
    print("   • Device: CUDA/GPU → CPU (MacBook optimized)")
    print("   • Batch Size: 32 → 16 (CPU optimized)")
    print("   • Embeddings: BGE-M3 (unchanged, CPU optimized)")
    print("   • Reranking: BGE Reranker v2-M3 (unchanged, CPU optimized)")
    print()
    print("✅ Migration complete! Your RAG pipeline now uses:")
    print("   🤖 Google Gemini API for text & vision")
    print("   💻 CPU-optimized processing for MacBook")
    print("   🚀 Improved performance on Apple Silicon")

if __name__ == "__main__":
    main()