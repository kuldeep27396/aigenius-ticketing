"""
SLA Application Layer
======================

Application layer for SLA monitoring module.

Contains:
- Services: Orchestrate business logic and coordinate with repositories
- DTOs: Data transfer objects for API serialization

This layer depends on the domain layer and repository interfaces,
but not on concrete infrastructure implementations.
"""

from src.sla.application.dto import (
    TicketIngestRequest,
    TicketCreateDTO,
    TicketUpdateDTO,
    DashboardQueryDTO,
    SLAStatusResponse,
    TicketSLAResponse,
    AlertResponse,
    DashboardResponse,
    DashboardSummary,
    IngestResponse,
    TicketEntityDTO,
)
from src.sla.application.services import (
    SLAService,
    SLAEvaluationService,
    ITicketRepository,
    ISLAAlertRepository,
    ISLAConfigProvider,
)

__all__ = [
    # DTOs
    "TicketIngestRequest",
    "TicketCreateDTO",
    "TicketUpdateDTO",
    "DashboardQueryDTO",
    "SLAStatusResponse",
    "TicketSLAResponse",
    "AlertResponse",
    "DashboardResponse",
    "DashboardSummary",
    "IngestResponse",
    "TicketEntityDTO",
    # Services
    "SLAService",
    "SLAEvaluationService",
    # Repository Interfaces
    "ITicketRepository",
    "ISLAAlertRepository",
    "ISLAConfigProvider",
]
