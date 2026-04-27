from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.services.estimator import clear_prediction_history_by_usermail, estimate_deer_age, estimate_deer_age_two_step, get_prediction_history_by_usermail
from typing import Optional
from app.services.estimator import init_database, SQLITE_DB_PATH

router = APIRouter()


@router.post("/predict")
async def estimate(
    image: UploadFile = File(...),
    mode: str = Query(
        default="single",
        description="Pipeline mode: 'single' (one-shot observe+classify) or 'two_step' (observe then classify separately)",
    ),
    usermail: Optional[str] = None
):
    # print(usermail)
    # if mode == "two_step":
    #     return await estimate_deer_age_two_step(image_bytes)
    return await estimate_deer_age(image, usermail)

@router.on_event("startup")
def startup_event():
    init_database(SQLITE_DB_PATH)

@router.post('/history')
async def get_history(request: Request, usermail: Optional[str] = Form(None)):
    resolved_usermail = (usermail or "").strip()
    if not resolved_usermail:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            resolved_usermail = str(payload.get("usermail") or "").strip()

    usermail = resolved_usermail
    if not usermail:
        raise HTTPException(status_code=400, detail="'usermail' is required (form-data or JSON body)")

    history = get_prediction_history_by_usermail(SQLITE_DB_PATH, usermail)
    return JSONResponse(content={
        "usermail": usermail,
        "count": len(history),
        "history": history,
    })


@router.post('/history/clear')
async def clear_history(request: Request, usermail: Optional[str] = Form(None)):
    resolved_usermail = (usermail or "").strip()
    if not resolved_usermail:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            resolved_usermail = str(payload.get("usermail") or "").strip()

    usermail = resolved_usermail
    if not usermail:
        raise HTTPException(status_code=400, detail="'usermail' is required (form-data or JSON body)")

    deleted_count = clear_prediction_history_by_usermail(SQLITE_DB_PATH, usermail)
    return JSONResponse(content={
        "ok": True,
        "usermail": usermail,
        "deleted_count": deleted_count,
    })

