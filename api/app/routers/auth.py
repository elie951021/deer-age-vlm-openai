from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import check_password, create_token, require_auth

router = APIRouter()


class LoginRequest(BaseModel):
    password: str


@router.post("/login")
async def login(body: LoginRequest):
    if not check_password(body.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token, expires_at = create_token()
    return {"access_token": token, "token_type": "bearer", "expires_at": expires_at}


@router.get("/me", dependencies=[Depends(require_auth)])
async def me():
    return {"authenticated": True}
