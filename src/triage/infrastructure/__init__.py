"""
Triage Infrastructure Layer
============================

Infrastructure implementations for ticket triage module.

Contains:
- Models: SQLAlchemy ORM models
- Repositories: Data access implementations
- External: External service adapters (LLM, Vector Store)
"""

from src.triage.infrastructure.models import TriageTicketModel, ClassificationModel
from src.triage.infrastructure.repositories import (
    SQLAlchemyTriageTicketRepository,
    SQLAlchemyClassificationRepository
)
from src.triage.infrastructure.external import (
    LLMClientAdapter,
    VectorStoreAdapter,
    DocumentIngester
)

__all__ = [
    "TriageTicketModel",
    "ClassificationModel",
    "SQLAlchemyTriageTicketRepository",
    "SQLAlchemyClassificationRepository",
    "LLMClientAdapter",
    "VectorStoreAdapter",
    "DocumentIngester",
]
