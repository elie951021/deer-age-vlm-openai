from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services.estimator import SQLITE_DB_PATH, get_all_predictions

# Auth is enforced for the whole router in main.py (require_auth).
router = APIRouter()


@router.get("/predictions")
async def list_predictions(
    usermail: Optional[str] = Query(None, description="Filter by user email (partial match)"),
    has_feedback: Optional[bool] = Query(None, description="Filter by whether feedback exists"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    predictions, total = get_all_predictions(
        db_path=SQLITE_DB_PATH,
        usermail=usermail,
        has_feedback=has_feedback,
        limit=limit,
        offset=offset,
    )

    return JSONResponse(content={
        "total": total,
        "limit": limit,
        "offset": offset,
        "count": len(predictions),
        "predictions": predictions,
    })
