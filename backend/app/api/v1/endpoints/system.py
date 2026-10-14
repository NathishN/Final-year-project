import sys
import torch
from fastapi import APIRouter
from app.schemas.system import SystemInfoResponse
from app.inference.engine import get_engine
from app.core.config import MAX_UPLOAD_SIZE_BYTES, ALLOWED_EXTENSIONS

router = APIRouter()

@router.get("/system-info", response_model=SystemInfoResponse)
async def get_system_info():
    engine = get_engine()
    ready = engine.is_ready()
    
    gpu_name = None
    gpu_mem_alloc = None
    gpu_mem_res = None
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_alloc = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        gpu_mem_res = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
        
    return SystemInfoResponse(
        application_name="FruitGrade AI",
        version="1.0.0",
        status="ready" if ready else "degraded",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        pytorch_version=torch.__version__,
        device_available=engine.device,
        gpu_name=gpu_name,
        gpu_memory_allocated_mb=gpu_mem_alloc,
        gpu_memory_reserved_mb=gpu_mem_res,
        max_upload_size_mb=MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
        allowed_extensions=sorted(list(ALLOWED_EXTENSIONS)),
        model_loaded=ready
    )
