"""Single-password auth with stateless HMAC-signed tokens (no DB, no extra deps)."""

import base64
import hashlib
import hmac
import json
import time
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _secret() -> bytes:
    # Changing AUTH_SECRET (or APP_PASSWORD when no secret is set) invalidates all tokens.
    secret = settings.auth_secret or f"pw:{settings.app_password}"
    return secret.encode()


def check_password(password: str) -> bool:
    if not settings.app_password:
        raise HTTPException(
            status_code=503,
            detail="Password login is not configured (set APP_PASSWORD in the API .env)",
        )
    return hmac.compare_digest(password.encode(), settings.app_password.encode())


def create_token() -> tuple[str, int]:
    expires_at = int(time.time()) + settings.auth_token_ttl_hours * 3600
    payload = _b64encode(json.dumps({"exp": expires_at}).encode())
    signature = _b64encode(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}", expires_at


def verify_token(token: str) -> bool:
    try:
        payload, signature = token.split(".", 1)
        expected = _b64encode(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return False
        data = json.loads(_b64decode(payload))
        return int(data.get("exp", 0)) > time.time()
    except Exception:
        return False


def require_auth(
    authorization: Optional[str] = Header(None),
    x_admin_key: Optional[str] = Header(None),
) -> None:
    """FastAPI dependency: accepts `Authorization: Bearer <token>` (from /api/auth/login)
    or `X-Admin-Key: <ADMIN_API_KEY>` (server-to-server)."""
    if not settings.app_password and not settings.admin_api_key:
        raise HTTPException(
            status_code=503,
            detail="Auth is not configured (set APP_PASSWORD and/or ADMIN_API_KEY in the API .env)",
        )

    if settings.admin_api_key and x_admin_key and hmac.compare_digest(
        x_admin_key.encode(), settings.admin_api_key.encode()
    ):
        return

    scheme, _, token = (authorization or "").partition(" ")
    if settings.app_password and scheme.lower() == "bearer" and verify_token(token.strip()):
        return

    raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
