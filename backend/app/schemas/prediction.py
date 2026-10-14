from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class SpeciesPrediction(BaseModel):
    species: str = Field(..., description="Predicted fruit species name")
    confidence: float = Field(..., description="Softmax confidence score for predicted species")
    probabilities: Dict[str, float] = Field(..., description="Softmax distribution across supported fruit species")

class OrdinalGradePrediction(BaseModel):
    grade: int = Field(..., description="Predicted discrete ordinal grade (0 to 4)")
    grade_name: str = Field(..., description="Severity name (e.g. Pristine, Mild, Moderate, Severe, Critical)")
    grade_description: str = Field(..., description="Practical description of the grade")
    confidence: float = Field(..., description="Estimated probability of the predicted grade")
    cumulative_probabilities: List[float] = Field(..., description="CORAL sigmoid probabilities P(y > k)")
    discrete_probabilities: List[float] = Field(..., description="Derived discrete PMF P(y = k)")

class OODAssessment(BaseModel):
    detector: str = Field(..., description="OOD detector used (e.g., Free Energy)")
    energy_score: float = Field(..., description="Computed free energy score (higher = more anomalous)")
    threshold: float = Field(..., description="Operating threshold for OOD flagging")
    is_ood: bool = Field(..., description="True if input is flagged as out-of-distribution or anomalous")
    interpretation: str = Field(..., description="User-facing interpretation of OOD score")

class SelectiveRouting(BaseModel):
    abstain_from_decision: bool = Field(..., description="Whether system abstains from automated decision")
    routing_action: str = Field(..., description="Action: AUTOMATED_ACCEPT, ROUTE_TO_MANUAL_INSPECTION, or REJECT_UNRELIABLE_OOD_ANOMALY")
    explanation: str = Field(..., description="Explanation for routing decision")

class PredictionResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique UUID for this analysis record")
    timestamp: str = Field(..., description="ISO 8601 timestamp of analysis")
    species: SpeciesPrediction = Field(..., description="Fruit species classification results")
    ordinal_grade: OrdinalGradePrediction = Field(..., description="Ordinal severity grading results")
    ood: OODAssessment = Field(..., description="Open-set / Out-of-distribution assessment")
    selective_routing: SelectiveRouting = Field(..., description="Selective prediction and abstention decision")
    inference_latency_ms: float = Field(..., description="Inference latency in milliseconds")
    device_used: str = Field(..., description="Execution device (e.g. cuda:0 or cpu)")
    thumbnail_base64: Optional[str] = Field(None, description="Optional base64 preview thumbnail")
    limitations: List[str] = Field(..., description="Research caveats and operational limitations")
