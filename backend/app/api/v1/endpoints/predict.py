import os
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from PIL import Image, UnidentifiedImageError
from app.schemas.prediction import PredictionResponse
from app.inference.engine import get_engine
from app.database.session import save_analysis
from app.core.config import MAX_UPLOAD_SIZE_BYTES, ALLOWED_EXTENSIONS

router = APIRouter()

# Protect against PIL decompression bombs
Image.MAX_IMAGE_PIXELS = 25000000

@router.post("/predict", response_model=PredictionResponse)
async def predict_fruit_quality(file: UploadFile = File(...)):
    # 1. Filename and extension check
    filename = file.filename or "unknown.jpg"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
    # 2. File size validation
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum permitted size of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
        )
        
    # 3. Safe Image Decoding
    import io
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify headers without decoding entire stream
        # Re-open after verify() because PIL verify() invalidates the file pointer
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image file. Please upload a valid PNG, JPG, or JPEG."
        )
        
    # 4. Inference Engine Execution
    engine = get_engine()
    if not engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently not ready or checkpoints are missing."
        )
        
    try:
        result = engine.predict(image, original_filename=filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )
        
    # 5. Persist to Local History Database
    try:
        save_analysis(result, original_filename=filename)
    except Exception as db_err:
        # Non-fatal logging: prediction still succeeds if history write fails
        pass

    return result
