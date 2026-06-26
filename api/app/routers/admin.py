from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.estimator import SQLITE_DB_PATH, get_all_predictions

router = APIRouter()


def _require_admin(x_admin_key: Optional[str]) -> None:
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API is not configured (set ADMIN_API_KEY in .env)",
        )
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")


@router.get("/predictions")
async def list_predictions(
    x_admin_key: Optional[str] = Header(None),
    usermail: Optional[str] = Query(None, description="Filter by user email (partial match)"),
    has_feedback: Optional[bool] = Query(None, description="Filter by whether feedback exists"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    _require_admin(x_admin_key)

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
