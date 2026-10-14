from fastapi import APIRouter
from app.inference.engine import get_engine

router = APIRouter()

@router.get("/health")
async def health_check():
    engine = get_engine()
    ready = engine.is_ready()
    return {
        "status": "healthy" if ready else "degraded",
        "service": "FruitGrade AI Backend",
        "version": "1.0.0",
        "model_ready": ready,
        "device": engine.device
    }
