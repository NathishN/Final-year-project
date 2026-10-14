import sys
import os
from contextlib import asynccontextmanager

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import API_PREFIX, ALLOWED_ORIGINS
from app.api.v1.api import api_router
from app.database.session import init_db
from app.inference.engine import get_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database and load model into memory once
    init_db()
    engine = get_engine()
    engine.load_models()
    yield
    # Shutdown logic if needed

app = FastAPI(
    title="FruitGrade AI Backend",
    description="Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic CORAL & Free Energy OOD Scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global unhandled exception handler to avoid exposing raw stacktraces in public responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred during processing. Please verify the request parameters or server logs.",
            "error_type": exc.__class__.__name__
        }
    )

# Include v1 router
app.include_router(api_router, prefix=API_PREFIX)

@app.get("/")
async def root():
    return {
        "name": "FruitGrade AI API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "online"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
