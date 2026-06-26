from fastapi import FastAPI
from pathlib import Path
from app.services.estimator import UPLOAD_DIR
from app.routers import admin, estimate, train
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

app.include_router(estimate.router, prefix="/api")
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(train.router, prefix="/api", tags=["train"])

if Path(UPLOAD_DIR).exists():
    app.mount("/upload", StaticFiles(directory=UPLOAD_DIR), name="upload")
