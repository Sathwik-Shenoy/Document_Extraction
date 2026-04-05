import secrets

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Hackathon Document Analyzer"
    app_version: str = "1.0.0"
    environment: str = "dev"
    max_upload_size_mb: int = 20

    use_heavy_models: bool = False
    summarizer_primary: str = "facebook/bart-large-cnn"
    summarizer_fallback: str = "google/pegasus-xsum"

    ner_primary: str = "dslim/bert-base-NER"
    ner_fallback: str = "dbmdz/bert-large-cased-finetuned-conll03-english"

    sentiment_distilbert: str = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_mnli_bert: str = "textattack/bert-base-uncased-MNLI"
    sentiment_mnli_roberta: str = "roberta-large-mnli"

    redis_url: str = "redis://redis:6379/0"
    database_url: str = "postgresql+psycopg://postgres:postgres@db:5432/analyzer"

    api_key: str = Field(default_factory=lambda: secrets.token_urlsafe(24))
    require_api_key: bool = False
    rate_limit_per_minute: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
