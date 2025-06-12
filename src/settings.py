from pathlib import Path

from dotenv import load_dotenv
from pydantic import StrictStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LOG_LEVEL: str = "INFO"

    MAX_RETRIES: int = 3

    LLM_API_KEY: StrictStr

    LLM_BASE_URL: str

    LLM_NAME: str

    MAX_CONCURRENT_LLM_CALLS: int = 10

    DISABLE_TRACING: bool = False

    FASTAPI_BASE_URL: str

    CALLBACK_URL: str


settings = Settings()
