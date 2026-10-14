from fastapi import APIRouter
from app.api.v1.endpoints import health, model_info, predict, history, system

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(model_info.router, tags=["Model Information"])
api_router.include_router(predict.router, tags=["Inference"])
api_router.include_router(history.router, tags=["History"])
api_router.include_router(system.router, tags=["System Diagnostics"])
