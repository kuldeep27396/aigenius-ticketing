"""
SLA Domain Layer
================

Domain layer for SLA monitoring module.

Contains:
- Entities: Core business objects with identity (Ticket, SLAMetrics, SLAAlert)
- Value Objects: Immutable objects defined by attributes (SLAConfig, SLADeadline)
- Domain Services: Stateless business logic (SLACalculator)

This layer has no dependencies on infrastructure - pure Python business logic.
"""

from src.sla.domain.entities import Ticket, SLAMetrics, SLAAlert
from src.sla.domain.value_objects import (
    SLACalculator,
    SLAConfig,
    EscalationLevelConfig,
    SLADeadline
)

__all__ = [
    # Entities
    "Ticket",
    "SLAMetrics",
    "SLAAlert",
    # Value Objects & Services
    "SLACalculator",
    "SLAConfig",
    "EscalationLevelConfig",
    "SLADeadline",
]
