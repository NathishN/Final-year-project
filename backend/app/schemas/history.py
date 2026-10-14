from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HistoryItem(BaseModel):
    id: str
    timestamp: str
    filename: Optional[str] = None
    species: str
    species_confidence: float
    grade: int
    grade_name: str
    grade_confidence: float
    is_ood: bool
    energy_score: float
    routing_action: str
    latency_ms: float
    thumbnail_base64: Optional[str] = None

class HistoryDetail(HistoryItem):
    discrete_probabilities: Optional[List[float]] = None
    device: Optional[str] = None

class HistoryListResponse(BaseModel):
    total: int
    items: List[HistoryItem]

class DeleteResponse(BaseModel):
    status: str
    message: str
    deleted_id: Optional[str] = None
