"""Application settings loaded from environment variables.

Uses pydantic-settings BaseSettings for typed configuration.
MLSENTRY_API_KEY and DATABASE_URL are required; missing values
cause startup failure via pydantic ValidationError.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MLSentry application configuration.

    All settings are loaded from environment variables.
    Required fields (no default) trigger ValidationError if absent.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Required — fail-fast if missing (no default value)
    mlsentry_api_key: str = Field(..., min_length=1)
    database_url: str = Field(..., min_length=1)

    # External integrations
    github_token: str = Field(default="")
    github_repo_owner: str = Field(default="")
    github_repo_name: str = Field(default="")
    github_workflow_id: str = Field(default="retrain.yml")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")

    # Monitoring configuration
    monitoring_interval_minutes: int = Field(default=15, ge=1)
    log_anomaly_confidence_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
    )
    distilbert_checkpoint: str = Field(default="distilbert-base-uncased")

    # Timeout configuration (env-var-configurable per ND-01, ND-03)
    mlflow_timeout_ms: int = Field(default=5000)
    github_api_timeout_ms: int = Field(default=10000)

    # Application
    log_level: str = Field(default="INFO")
