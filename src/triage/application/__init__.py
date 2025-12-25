"""
Triage Application Layer
=========================

Application layer for ticket triage module.

Contains:
- Services: Business logic orchestration
- DTOs: Data transfer objects for API serialization
"""

from triage.application.dto import (
    ClassifyRequest,
    RespondRequest,
    IngestDocumentsRequest,
    ClassificationResponse,
    RespondResponse,
    IngestResponse,
    StatsResponse,
    TriageTicketDTO,
    ClassificationInfo,
    CitationInfo
)
from triage.application.services import (
    ClassificationService,
    RAGService,
    ITriageTicketRepository,
    IClassificationRepository,
    IVectorStore,
    ILLMClient
)

__all__ = [
    # DTOs
    "ClassifyRequest",
    "RespondRequest",
    "IngestDocumentsRequest",
    "ClassificationResponse",
    "RespondResponse",
    "IngestResponse",
    "StatsResponse",
    "TriageTicketDTO",
    "ClassificationInfo",
    "CitationInfo",
    # Services
    "ClassificationService",
    "RAGService",
    # Repository Interfaces
    "ITriageTicketRepository",
    "IClassificationRepository",
    "IVectorStore",
    "ILLMClient",
]
