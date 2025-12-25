"""
Triage Domain Layer
===================

Domain layer for ticket triage module.

Contains:
- Entities: Core business objects (TriageTicket, ClassificationResult, RAGResult)
- Value Objects: Immutable objects (Citation, ClassificationPromptBuilder)

This layer is framework-agnostic and contains pure business logic.
"""

from src.triage.domain.entities import (
    ClassificationResult,
    TriageTicket,
    Citation,
    RAGResult,
    ClassificationPromptBuilder
)

__all__ = [
    "ClassificationResult",
    "TriageTicket",
    "Citation",
    "RAGResult",
    "ClassificationPromptBuilder",
]
