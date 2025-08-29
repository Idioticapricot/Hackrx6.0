# src/ai/embedding_models.py
"""ML model initialization and management"""

import torch
import warnings
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from ..core.config import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*legacy.*", category=FutureWarning)

# Device configuration - Optimized for MacBook/CPU usage
EMBEDDING_DEVICE = 'cpu'
RERANKER_DEVICE = 'cpu'
print("💻 MacBook Optimization: Using CPU for all operations (stable & memory-efficient)")
if torch.cuda.is_available():
    print(f"ℹ️ GPU available ({torch.cuda.get_device_name(0)}) but using CPU for better MacBook compatibility")

# Global model instances
embed_model = None
reranker_model = None

def initialize_models():
    """Initialize ML models with optimized GPU/CPU allocation"""
    global embed_model, reranker_model
    
    print(f"🔄 Loading BGE embedding model '{EMBEDDING_MODEL_NAME}' onto '{EMBEDDING_DEVICE.upper()}'...")
    # BGE-M3 on CPU for MacBook optimization
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE, trust_remote_code=True)
    _ = embed_model.encode(["Warming up the BGE embedding model..."], show_progress_bar=False)
    print(f"✅ BGE embedding model loaded on {EMBEDDING_DEVICE.upper()} (MacBook optimized)")

    print(f"🔄 Loading BGE reranker model '{RERANKER_MODEL_NAME}' onto '{RERANKER_DEVICE.upper()}'...")
    # BGE reranker on CPU for stable MacBook performance
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=RERANKER_DEVICE, trust_remote_code=True)
    print(f"✅ BGE reranker model loaded on {RERANKER_DEVICE.upper()} (MacBook optimized)")
    
    # FAISS uses CPU for MacBook-optimized indexing/searching
    print("💻 FAISS configured for CPU-based similarity search (MacBook optimized)")

def get_embed_model():
    """Get the embedding model instance"""
    return embed_model

def get_reranker_model():
    """Get the reranker model instance"""
    return reranker_model

def get_device():
    """Get the current device (cpu for MacBook optimization)"""
    return EMBEDDING_DEVICE