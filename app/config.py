from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_BUCKET_NAME: str = "acne-analyzer-photos"
    AWS_REGION: str = "eu-central-1"

    LLM_API_URL: str = "http://localhost:1234/v1/chat/completions"
    LLM_MODEL: str = "local-model"
    RAG_PDF_PATH: str = "knowledge_base.pdf"

    MODEL_PATH: str = "best_model.pt"
    SUPPORT_SET_PATH: str = "support_set"


settings = Settings()
