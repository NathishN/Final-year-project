import os
from pydantic_settings import BaseSettings
from typing import List, Tuple

class Settings(BaseSettings):
    PROJECT_NAME: str = "FruitGrade AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Project paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ROOT_DIR: str = os.path.dirname(BASE_DIR)
    
    BACKBONE_CHECKPOINT: str = os.path.join(ROOT_DIR, "outputs", "checkpoints", "best_model.pth")
    HEAD_CHECKPOINT: str = os.path.join(ROOT_DIR, "outputs", "casm_v2", "phase50_checkpoints", "model_A_seed83_best.pth")
    DATABASE_PATH: str = os.path.join(BASE_DIR, "app", "database", "fruitgrade_history.db")
    
    # Security & limits
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    # Model defaults
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70
    ENERGY_OOD_THRESHOLD: float = -1.50
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

settings = Settings()

# Top-level exports and domain constants
API_PREFIX = settings.API_V1_STR
ALLOWED_ORIGINS = settings.CORS_ORIGINS
MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_BYTES
ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS
DEFAULT_OOD_THRESHOLD = settings.ENERGY_OOD_THRESHOLD
DEFAULT_ABSTENTION_THRESHOLD = settings.DEFAULT_CONFIDENCE_THRESHOLD
BACKBONE_CHECKPOINT = settings.BACKBONE_CHECKPOINT
CORAL_CHECKPOINT = settings.HEAD_CHECKPOINT

BACKBONE_NAME = "convnext_tiny"
CORAL_HEAD_TYPE = "monotonic_coral"

SUPPORTED_SPECIES: List[str] = ["Apple", "Banana", "Mango", "Orange", "Tomato"]

GRADE_NAMES: List[str] = ["Pristine", "Mild", "Moderate", "Severe", "Critical"]
GRADE_DESCRIPTIONS: List[str] = [
    "Grade 0 - Pristine export quality: defect-free surface, optimal firmness and color uniformity.",
    "Grade 1 - Mild superficial blemishes: minor cosmetic defects, fully intact flesh, high retail value.",
    "Grade 2 - Moderate defect severity: noticeable bruising or skin damage, suitable for immediate local consumption or processing.",
    "Grade 3 - Severe degradation: structural breakdown, deep lesions, unsuitable for fresh consumption.",
    "Grade 4 - Critical decay / cull: advanced rot, mold, or widespread decomposition requiring immediate discard."
]

IMAGE_SIZE: int = 224
IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
