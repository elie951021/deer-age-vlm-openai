from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"
    session_logging_enabled: bool = True
    # Server-to-server key (sent as X-Admin-Key, e.g. by the deer-age-admin Express server).
    admin_api_key: str = ""
    # Single shared login password. Auth-protected routes return 503 until it is set.
    app_password: str = ""
    # Secret used to sign login tokens. Optional; falls back to a value derived from app_password.
    auth_secret: str = ""
    auth_token_ttl_hours: int = 24

    class Config:
        env_file = ".env"
        extra = "ignore"  # tolerate leftover keys (e.g. old ADMIN_API_KEY) in .env


settings = Settings()
