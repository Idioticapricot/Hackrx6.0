# src/api/server.py
"""FastAPI server configuration and setup"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ..ai import initialize_models
from .endpoints import hackathon_endpoint, legal_analysis_endpoint, simplify_document_endpoint, detect_risks_endpoint
from ..utils.terminal_ui import display_startup_info

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Q&A System",
    description="A modular document question-answering system using RAG",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    display_startup_info()
    print("📚 Initializing AI models...")
    initialize_models()
    print("\n✅ All systems ready! Server is now accepting requests.\n")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Frontend routes
@app.get("/")
async def legal_frontend():
    return FileResponse("static/legal.html")

@app.get("/original")
async def original_frontend():
    return FileResponse("static/index.html")

# Register endpoints
app.post("/hackathon")(hackathon_endpoint)
app.post("/hackrx/run")(hackathon_endpoint)
app.post("/legal/analyze")(legal_analysis_endpoint)
app.post("/legal/simplify")(simplify_document_endpoint)
app.post("/legal/risks")(detect_risks_endpoint)