from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class CheckpointInfo(BaseModel):
    name: str
    path: str
    loaded: bool
    parameter_count: int

class GradeDefinition(BaseModel):
    grade: int
    name: str
    description: str

class ResearchMetrics(BaseModel):
    benchmark_name: str
    image_only_qwk_mean: float
    image_only_qwk_std: float
    n_seeds: int
    spearman_rho: float
    accuracy: float
    ood_fpr95_inverted_real: float
    ood_auroc_inverted_real: float

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    backbone_architecture: str
    coral_head_type: str
    num_grades: int
    grades: List[GradeDefinition]
    supported_species: List[str]
    input_resolution: List[int]
    input_normalization: Dict[str, List[float]]
    ood_detector: str
    ood_threshold: float
    abstention_threshold: float
    loaded_checkpoints: Dict[str, CheckpointInfo]
    device: str
    model_ready: bool
    research_evaluation: ResearchMetrics
    documentation_url: str
