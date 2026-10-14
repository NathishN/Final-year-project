from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class SystemInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    application_name: str
    version: str
    status: str
    python_version: str
    pytorch_version: str
    device_available: str
    gpu_name: Optional[str] = None
    gpu_memory_allocated_mb: Optional[float] = None
    gpu_memory_reserved_mb: Optional[float] = None
    max_upload_size_mb: int
    allowed_extensions: list[str]
    model_loaded: bool
