import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

# api/ root (two levels up from this file: routers -> app -> api)
API_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = API_ROOT / "app" / "train.py"


@router.get("/train")
def train():
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data_dir", ".\\data",
        "--csv_file", ".\\data\\train.csv",
        "--epochs", "20",
        "--batch_size", "32",
        "--lr", "3e-4",
        "--img_size", "224",
        "--freeze_backbone",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(API_ROOT),
        capture_output=True,
        text=True,
    )

    return JSONResponse(
        status_code=200 if result.returncode == 0 else 500,
        content={
            "ok": result.returncode == 0,
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
