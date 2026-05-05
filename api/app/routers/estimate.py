from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.services.estimator import clear_prediction_history_by_usermail, estimate_deer_age, save_prediction_feedback, get_prediction_history_by_usermail
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

@router.post('/feedback')
async def submit_feedback(
    prediction_id: int = Form(...),
    exact_age: str = Form(...),
    feedback: Optional[str] = Form(None),
):
    exact_age_text = exact_age.strip()
    if not exact_age_text:
        raise HTTPException(status_code=400, detail="exact_age is required")
    if len(exact_age_text) > 50:
        raise HTTPException(status_code=400, detail="exact_age is too long (max 50 chars)")

    feedback_text = (feedback or "").strip()
    if len(feedback_text) > 2000:
        raise HTTPException(status_code=400, detail="feedback is too long (max 2000 chars)")

    updated = save_prediction_feedback(
        db_path=SQLITE_DB_PATH,
        prediction_id=prediction_id,
        exact_age=exact_age_text,
        feedback=feedback_text or None,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="prediction not found")

    return JSONResponse(content={
        "ok": True,
        "prediction_id": prediction_id,
    })


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

