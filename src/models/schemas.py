# src/models/schemas.py
"""Pydantic models for API requests and responses"""

from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class HackathonRequest(BaseModel):
    """Request model for the hackathon endpoint"""
    documents: HttpUrl
    questions: List[str]

class LegalAnalysisRequest(BaseModel):
    """Request model for legal document analysis"""
    document_url: HttpUrl
    language: Optional[str] = 'en'