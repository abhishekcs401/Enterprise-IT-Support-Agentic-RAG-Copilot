
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"

    # API Keys
    google_api_key: str = ""
    groq_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""

    # Pinecone Configuration
    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"

    # Embedding Configuration
    embedding_model: str = "gemini-embedding-001"

    # LLM Configuration
    llm_provider: str = "groq"
    groq_model: str = "llama-3.3-70b-versatile"

    # RAG Configuration
    top_k: int = 4
    max_retries: int = 1

    # Security
    admin_api_key: str = "change-me"

    # Application Paths
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
