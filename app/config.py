"""
Application configuration.

Settings are loaded from environment variables (or a .env file at startup).
No secrets are hardcoded here — sensitive values must be supplied at runtime.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level application settings."""

    # Service identity
    app_name: str = "DeployGuard Demo — Order Service"
    service_name: str = "deployguard-demo"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    # Network
    host: str = "127.0.0.1"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton — instantiated once at import time.
# All application code should import this object rather than re-instantiating.
settings = Settings()
