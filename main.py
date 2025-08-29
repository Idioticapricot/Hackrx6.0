# main.py
"""
Main entry point for the RAG-based document Q&A system.
This is a well-organized version that maintains the exact same functionality
as the original final.py but structured into proper Python packages.
"""

if __name__ == "__main__":
    # Suppress warnings for cleaner startup output
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    import sys
    import uvicorn
    from src.api import app
    
    # Parse port argument
    port = 8000
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                except ValueError:
                    print("❌ Invalid port number")
                    sys.exit(1)
    
    print("🚀 Starting RAG Document Q&A Server...")
    print("📚 All modules loaded and organized into proper packages")
    print("🔧 Core RAG pipeline remains unchanged")
    print("🏗️ Professional package structure for better maintainability")
    print(f"🌐 Server will start on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)