"""
Configuration Module
====================

Application settings and configuration management using Pydantic.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Uses Pydantic for validation and type safety.
    """

    # ========== Application ==========
    app_name: str = Field(default="ticket-service", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")

    # ========== Server ==========
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port", ge=1, le=65535)

    # ========== Database ==========
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/tickets",
        description="PostgreSQL connection URL (async)"
    )
    db_pool_size: int = Field(default=5, description="Database connection pool size", ge=1)
    db_max_overflow: int = Field(default=10, description="Max overflow connections", ge=0)

    # ========== SLA Configuration ==========
    sla_config_path: Path = Field(
        default=Path("sla_config.yaml"),
        description="Path to SLA configuration YAML file"
    )
    sla_evaluation_interval: int = Field(
        default=60,
        description="Seconds between SLA evaluations",
        ge=10
    )

    # ========== Slack Integration ==========
    slack_webhook_url: Optional[str] = Field(
        default=None,
        description="Slack webhook URL for notifications"
    )
    slack_channel: str = Field(
        default="#all-customer-tickets",
        description="Slack channel for SLA notifications"
    )
    slack_timeout_seconds: float = Field(
        default=5.0,
        description="Timeout for Slack API calls",
        ge=0.1,
        le=30
    )

    # ========== Z.AI SDK (GLM 4.7) ==========
    zai_api_key: Optional[str] = Field(
        default=None,
        description="Z.AI API key for GLM 4.7"
    )
    mock_llm: bool = Field(
        default=False,
        description="Use mock LLM responses for testing (no API calls)"
    )

    # ========== Zilliz Cloud (Managed Milvus) Configuration ==========
    zilliz_uri: str = Field(
        default="",
        description="Zilliz Cloud cluster URI (get from console, e.g., https://inxxx.aws-us-west-2.vectordb-uat3.zillizcloud.com)"
    )
    zilliz_api_key: str = Field(
        default="",
        description="Zilliz Cloud API key"
    )
    zilliz_cluster_id: str = Field(
        default="",
        description="Zilliz Cloud cluster ID (deprecated - use zilliz_uri instead)"
    )
    milvus_collection_name: str = Field(
        default="tickets_docs",
        description="Milvus collection name"
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Embedding vector dimension",
        ge=128
    )
    top_k_results: int = Field(
        default=5,
        description="Number of documents to retrieve in RAG",
        ge=1,
        le=20
    )
    chunk_size: int = Field(
        default=1000,
        description="Character size for document chunks",
        ge=100
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between document chunks",
        ge=0
    )

    # ========== LLM Settings ==========
    llm_model: str = Field(
        default="glm-4.7",
        description="GLM model for generation and classification"
    )
    embedding_model: str = Field(
        default="embedding-2",
        description="Z.AI embedding model"
    )
    llm_temperature: float = Field(
        default=0.3,
        description="Default temperature for LLM",
        ge=0.0,
        le=1.0
    )
    llm_max_tokens: int = Field(
        default=1000,
        description="Default max tokens for LLM generation",
        ge=1,
        le=8000
    )

    # ========== CORS ==========
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )

    # ========== Rate Limiting ==========
    rate_limit_per_minute: int = Field(
        default=100,
        description="Max requests per minute per IP",
        ge=1
    )

    # ========== Grafana OTLP Metrics ==========
    grafana_host: Optional[str] = Field(
        default=None,
        description="Grafana OTLP gateway URL (e.g., https://otlp-gateway-prod-ap-south-1.grafana.net)"
    )
    grafana_api_key: Optional[str] = Field(
        default=None,
        description="Grafana API key for OTLP authentication"
    )
    grafana_instance_id: Optional[str] = Field(
        default=None,
        description="Grafana instance ID for OTLP authentication"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is one of allowed values."""
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Returns cached Settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


# ========== Constants ==========

class ProductCategory(str):
    """Product categories for ticket classification."""
    CASB = "CASB"           # Cloud Access Security Broker
    SWG = "SWG"             # Secure Web Gateway
    ZTNA = "ZTNA"           # Zero Trust Network Access
    DLP = "DLP"             # Data Loss Prevention
    SSPM = "SSPM"           # SaaS Security Posture Management
    CFW = "CFW"             # Cloud Firewall
    GENERAL = "GENERAL"     # General/Other


class UrgencyLevel(str):
    """Ticket urgency levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(str):
    """Ticket priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CustomerTier(str):
    """Customer tier levels."""
    ENTERPRISE = "enterprise"
    BUSINESS = "business"
    STANDARD = "standard"
    FREE = "free"


class TicketStatus(str):
    """Ticket lifecycle statuses."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    PENDING_VENDOR = "pending_vendor"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SLAType(str):
    """Types of SLA clocks."""
    RESPONSE = "response"
    RESOLUTION = "resolution"


class AlertType(str):
    """SLA alert types."""
    WARNING = "warning"
    BREACH = "breach"


class SLAState(str):
    """SLA status states."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    MET = "met"


# ========== Lists for validation ==========

PRODUCT_CATEGORIES = [
    ProductCategory.CASB, ProductCategory.SWG, ProductCategory.ZTNA,
    ProductCategory.DLP, ProductCategory.SSPM, ProductCategory.CFW,
    ProductCategory.GENERAL
]
URGENCY_LEVELS = [
    UrgencyLevel.CRITICAL, UrgencyLevel.HIGH,
    UrgencyLevel.MEDIUM, UrgencyLevel.LOW
]
VALID_PRIORITIES = [
    Priority.CRITICAL, Priority.HIGH,
    Priority.MEDIUM, Priority.LOW
]
VALID_CUSTOMER_TIERS = [
    CustomerTier.ENTERPRISE, CustomerTier.BUSINESS,
    CustomerTier.STANDARD, CustomerTier.FREE
]
VALID_STATUSES = [
    TicketStatus.OPEN, TicketStatus.IN_PROGRESS,
    TicketStatus.PENDING_CUSTOMER, TicketStatus.PENDING_VENDOR,
    TicketStatus.RESOLVED, TicketStatus.CLOSED
]
VALID_SLA_TYPES = [SLAType.RESPONSE, SLAType.RESOLUTION]
VALID_ALERT_TYPES = [AlertType.WARNING, AlertType.BREACH]
VALID_SLA_STATES = [SLAState.ON_TRACK, SLAState.AT_RISK, SLAState.BREACHED, SLAState.MET]
