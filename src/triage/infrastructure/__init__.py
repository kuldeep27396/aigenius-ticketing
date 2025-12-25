"""
Triage Infrastructure Layer
============================

Infrastructure implementations for ticket triage module.

Contains:
- Models: SQLAlchemy ORM models
- Repositories: Data access implementations
- External: External service adapters (LLM, Vector Store)
"""

from triage.infrastructure.models import TriageTicketModel, ClassificationModel
from triage.infrastructure.repositories import (
    SQLAlchemyTriageTicketRepository,
    SQLAlchemyClassificationRepository
)
from triage.infrastructure.external import (
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
