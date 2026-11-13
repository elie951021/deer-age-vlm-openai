from fastapi import Depends, FastAPI
from pathlib import Path
from app.core.auth import require_auth
from app.services.estimator import UPLOAD_DIR
from app.routers import admin, auth, estimate, train
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Deer Age Estimation API",
    description="Estimates deer age from jaw/teeth images using LLM vision.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://127.0.0.1:8081"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Public user-facing routes (predict/feedback/history) stay open for the ui app.
# Admin and training routes require a login token or X-Admin-Key.
protected = [Depends(require_auth)]

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(estimate.router, prefix="/api")
app.include_router(admin.router, prefix="/api/admin", tags=["admin"], dependencies=protected)
app.include_router(train.router, prefix="/api", tags=["train"], dependencies=protected)

if Path(UPLOAD_DIR).exists():
    app.mount("/upload", StaticFiles(directory=UPLOAD_DIR), name="upload")
