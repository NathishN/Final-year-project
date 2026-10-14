from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.history import (
    HistoryListResponse,
    HistoryItem,
    HistoryDetail,
    DeleteResponse,
)
from app.database.session import (
    get_history_items,
    get_history_item_by_id,
    delete_history_item,
    clear_all_history,
)

router = APIRouter()

@router.get("/history", response_model=HistoryListResponse)
async def list_history(
    limit: int = Query(50, ge=1, le=200, description="Number of records to retrieve"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    search: Optional[str] = Query(None, description="Search term for species or grade name"),
    species: Optional[str] = Query(None, description="Filter by fruit species"),
    grade: Optional[int] = Query(None, ge=0, le=4, description="Filter by grade (0-4)"),
    is_ood: Optional[bool] = Query(None, description="Filter by OOD flag"),
):
    total, items = get_history_items(
        limit=limit,
        offset=offset,
        search=search,
        species=species,
        grade=grade,
        is_ood=is_ood,
    )
    return HistoryListResponse(
        total=total,
        items=[HistoryItem(**item) for item in items]
    )

@router.get("/history/{analysis_id}", response_model=HistoryDetail)
async def get_history_detail(analysis_id: str):
    item = get_history_item_by_id(analysis_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record with ID '{analysis_id}' was not found."
        )
    return HistoryDetail(**item)

@router.delete("/history/{analysis_id}", response_model=DeleteResponse)
async def delete_history_record(analysis_id: str):
    deleted = delete_history_item(analysis_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record with ID '{analysis_id}' was not found."
        )
    return DeleteResponse(
        status="success",
        message=f"Analysis record '{analysis_id}' was successfully deleted.",
        deleted_id=analysis_id
    )

@router.delete("/history", response_model=DeleteResponse)
async def clear_history(confirm: bool = Query(False, description="Must be true to confirm clearing history")):
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required: set confirm=true to clear all analysis history."
        )
    count = clear_all_history()
    return DeleteResponse(
        status="success",
        message=f"Successfully cleared {count} analysis records from history."
    )
