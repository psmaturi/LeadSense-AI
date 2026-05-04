from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="The lead description to classify")
    lead_name: Optional[str] = Field(None, description="Name of the lead")
    company: Optional[str] = Field(None, description="Company name")
    source: Optional[str] = Field("email", description="Source of the lead (email, crm, etc.)")

class ClassifyResponse(BaseModel):
    record_id: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    engine: str
    applied_rules: List[str] = []
    lead_name: Optional[str]
    company: Optional[str]
    source: Optional[str]
    retrieved_context: Optional[str] = None
    intelligence_summary: Optional[str] = None
    rag_enabled: bool = False
    timestamp: str

class StatusResponse(BaseModel):
    model_ready: bool

class HistoryResponse(BaseModel):
    history: List[ClassifyResponse]

class AnalyticsResponse(BaseModel):
    session: Dict
