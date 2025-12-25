"""
SLA Controllers (API Routes)
=============================

FastAPI routes for SLA monitoring endpoints.

Controllers are thin - they delegate to application services.
"""

import time
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database import get_session
from sla.application import (
    SLAService,
    TicketIngestRequest, TicketCreateDTO,
    DashboardQueryDTO, TicketSLAResponse,
    DashboardResponse, DashboardSummary,
    IngestResponse, TicketEntityDTO,
    SLAStatusResponse, AlertResponse
)
from sla.infrastructure import (
    SQLAlchemyTicketRepository,
    SQLAlchemyAlertRepository,
    YAMLConfigProvider
)
from config import Priority, CustomerTier, TicketStatus, SLAState

from shared.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/sla", tags=["SLA Monitoring"])


# ========== Example payloads for Swagger ==========

TICKET_CREATE_EXAMPLE = {
    "id": "TICKET-001",
    "priority": "high",
    "customer_tier": "enterprise",
    "status": "open",
    "subject": "CASB Salesforce sync not working",
    "content": "Our CASB integration with Salesforce has stopped syncing data since yesterday.",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:00Z"
}

INGEST_RESPONSE_EXAMPLE = {
    "created": 1,
    "updated": 0,
    "failed": 0,
    "errors": []
}

TICKET_SLA_RESPONSE_EXAMPLE = {
    "ticket_id": "123e4567-e89b-12d3-a456-426614174000",
    "external_id": "TICKET-001",
    "priority": "high",
    "customer_tier": "enterprise",
    "status": "open",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:00Z",
    "response_sla": {
        "deadline": "2024-01-15T10:30:00Z",
        "remaining_seconds": 1800,
        "percentage_remaining": 50.0,
        "is_breached": False,
        "state": "on_track",
        "met_at": None
    },
    "resolution_sla": {
        "deadline": "2024-01-17T10:00:00Z",
        "remaining_seconds": 172800,
        "percentage_remaining": 100.0,
        "is_breached": False,
        "state": "on_track",
        "met_at": None
    },
    "overall_state": "on_track",
    "next_deadline": "2024-01-15T10:30:00Z",
    "active_alerts": []
}

DASHBOARD_RESPONSE_EXAMPLE = {
    "tickets": [],
    "total_count": 0,
    "summary": {
        "total_tickets": 0,
        "breached_count": 0,
        "at_risk_count": 0,
        "on_track_count": 0,
        "met_count": 0,
        "breach_rate": 0.0
    }
}


# ========== Dependencies ==========

async def get_sla_service(
    session: AsyncSession = Depends(get_session)
) -> SLAService:
    """Get SLA service instance."""
    ticket_repo = SQLAlchemyTicketRepository(session)
    config_provider = YAMLConfigProvider("sla_config.yaml")
    return SLAService(ticket_repo, config_provider)


async def get_evaluation_service(
    session: AsyncSession = Depends(get_session)
):
    """Get SLA evaluation service instance."""
    ticket_repo = SQLAlchemyTicketRepository(session)
    alert_repo = SQLAlchemyAlertRepository(session)
    config_provider = YAMLConfigProvider("sla_config.yaml")

    from sla.application import SLAEvaluationService
    return SLAEvaluationService(ticket_repo, alert_repo, config_provider)


# ========== Route Handlers ==========

@router.post(
    "/tickets",
    response_model=IngestResponse,
    summary="Ingest tickets for SLA tracking",
    description="""
    Ingest a batch of tickets for SLA monitoring and tracking.

    **Idempotent**: Tickets are identified by `id` (external ticket ID). If a ticket
    already exists and the new `updated_at` is newer, the ticket is updated.

    **Priority Levels**: `critical`, `high`, `medium`, `low`

    **Customer Tiers**: `enterprise`, `business`, `standard`, `free`

    **Ticket Status**: `open`, `in_progress`, `pending_customer`, `pending_vendor`, `resolved`, `closed`

    **SLA Calculation**:
    - Response SLA: Time to first response
    - Resolution SLA: Time to complete resolution

    **Example Request**:
    ```json
    {
        "tickets": [
            {
                "id": "TICKET-001",
                "priority": "high",
                "customer_tier": "enterprise",
                "status": "open",
                "subject": "CASB Salesforce sync not working",
                "content": "Our CASB integration with Salesforce has stopped syncing data since yesterday.",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:00:00Z"
            }
        ]
    }
    ```
    """,
    responses={
        200: {
            "description": "Tickets ingested successfully",
            "content": {
                "application/json": {
                    "example": INGEST_RESPONSE_EXAMPLE
                }
            }
        }
    }
)
async def ingest_tickets(
    request: TicketIngestRequest,
    session: AsyncSession = Depends(get_session)
):
    start_time = time.perf_counter()

    created = 0
    updated = 0
    failed = 0
    errors = []

    ticket_repo = SQLAlchemyTicketRepository(session)

    for ticket_dto in request.tickets:
        try:
            # Convert to internal DTO
            entity_dto = TicketEntityDTO(
                id=None,
                external_id=ticket_dto.id,
                priority=ticket_dto.priority,
                customer_tier=ticket_dto.customer_tier,
                status=ticket_dto.status,
                subject=ticket_dto.subject,
                content=ticket_dto.content,
                created_at=ticket_dto.created_at,
                updated_at=ticket_dto.updated_at,
                first_response_at=ticket_dto.first_response_at,
                resolved_at=ticket_dto.resolved_at,
                closed_at=ticket_dto.closed_at
            )

            # Check if exists
            existing = await ticket_repo.get_by_external_id(ticket_dto.id)

            if existing:
                # Update if newer
                if ticket_dto.updated_at > existing.updated_at:
                    await ticket_repo.update(entity_dto)
                    updated += 1
                else:
                    # Skip if not newer
                    pass
            else:
                # Create new
                await ticket_repo.create(entity_dto)
                created += 1

        except Exception as e:
            failed += 1
            errors.append(f"{ticket_dto.id}: {str(e)}")
            logger.error(f"Failed to ingest ticket {ticket_dto.id}: {e}")

    await session.commit()

    total_time = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "Ticket ingestion complete",
        extra={
            "tickets_created": created,
            "tickets_updated": updated,
            "tickets_failed": failed,
            "processing_time_ms": total_time
        }
    )

    return IngestResponse(
        created=created,
        updated=updated,
        failed=failed,
        errors=errors
    )


@router.get(
    "/tickets/{ticket_id}",
    response_model=TicketSLAResponse,
    summary="Get ticket SLA status",
    description="""
    Get detailed SLA information for a single ticket.

    Returns:
        - Response and resolution SLA status
        - Current state (on_track, at_risk, breached, met)
        - Active alerts
    """,
    responses={
        200: {
            "description": "Ticket SLA information",
            "content": {
                "application/json": {
                    "example": TICKET_SLA_RESPONSE_EXAMPLE
                }
            }
        },
        404: {
            "description": "Ticket not found"
        }
    }
)
async def get_ticket_sla(
    ticket_id: str,
    session: AsyncSession = Depends(get_session),
    sla_service: SLAService = Depends(get_sla_service)
):
    ticket_repo = SQLAlchemyTicketRepository(session)
    alert_repo = SQLAlchemyAlertRepository(session)

    # Get ticket
    ticket_data = await ticket_repo.get_by_id(ticket_id)
    if not ticket_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket {ticket_id} not found"
        )

    # Calculate SLA metrics
    metrics = await sla_service.calculate_sla_metrics(ticket_id)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not calculate SLA for ticket {ticket_id}"
        )

    # Get active alerts
    pending_alerts = await alert_repo.get_pending_alerts(ticket_id)

    # Build response
    response_sla = SLAStatusResponse(
        deadline=metrics.response_deadline,
        remaining_seconds=metrics.response_remaining_seconds,
        percentage_remaining=metrics.response_percentage_remaining,
        is_breached=metrics.response_is_breached,
        state=metrics.response_state,
        met_at=metrics.response_met_at
    )

    resolution_sla = SLAStatusResponse(
        deadline=metrics.resolution_deadline,
        remaining_seconds=metrics.resolution_remaining_seconds,
        percentage_remaining=metrics.resolution_percentage_remaining,
        is_breached=metrics.resolution_is_breached,
        state=metrics.resolution_state,
        met_at=metrics.resolution_met_at
    )

    alerts = [
        AlertResponse(
            id=alert.id,
            ticket_id=alert.ticket_id,
            ticket_external_id=alert.ticket_external_id,
            sla_type=alert.sla_type,
            alert_type=alert.alert_type,
            triggered_at=alert.triggered_at,
            deadline=alert.deadline,
            remaining_seconds=alert.remaining_seconds,
            state=alert.state,
            escalation_level=alert.escalation_level,
            notification_sent=alert.notification_sent,
            channels=alert.channels
        )
        for alert in pending_alerts
    ]

    return TicketSLAResponse(
        ticket_id=str(ticket_data.id),
        external_id=ticket_data.external_id,
        priority=ticket_data.priority,
        customer_tier=ticket_data.customer_tier,
        status=ticket_data.status,
        created_at=ticket_data.created_at,
        updated_at=ticket_data.updated_at,
        response_sla=response_sla,
        resolution_sla=resolution_sla,
        overall_state=metrics.most_urgent_state,
        next_deadline=metrics.next_deadline,
        active_alerts=alerts
    )


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Get SLA dashboard",
    description="""
    Get dashboard view with SLA status for all tickets.

    **Query Parameters:**
    - `customer_tier`: Filter by customer tier (enterprise, business, standard, free)
    - `priority`: Filter by priority (critical, high, medium, low)
    - `status`: Filter by ticket status (open, in_progress, pending_customer, pending_vendor, resolved, closed)
    - `sla_state`: Filter by SLA state (breached, at_risk, on_track, met)
    - `limit`: Results per page (default: 100, max: 1000)
    - `offset`: Page offset for pagination (default: 0)

    **Response includes:**
    - List of tickets with SLA metrics (response_sla, resolution_sla)
    - Overall SLA state and next deadline
    - Summary statistics (total, breached, at_risk, on_track, met)
    - Breach rate percentage

    **SLA States:**
    - `breached`: SLA deadline has passed
    - `at_risk`: Warning threshold exceeded (default: 80% of SLA used)
    - `on_track`: SLA is within safe limits
    - `met`: SLA was successfully achieved

    **Example Response**:
    ```json
    {
        "tickets": [
            {
                "ticket_id": "uuid",
                "external_id": "TICKET-001",
                "priority": "high",
                "customer_tier": "enterprise",
                "status": "open",
                "response_sla": {
                    "deadline": "2024-01-15T10:30:00Z",
                    "remaining_seconds": 1800,
                    "percentage_remaining": 50.0,
                    "is_breached": false,
                    "state": "on_track"
                },
                "resolution_sla": { ... },
                "overall_state": "on_track",
                "next_deadline": "2024-01-15T10:30:00Z"
            }
        ],
        "total_count": 1,
        "summary": {
            "total_tickets": 1,
            "breached_count": 0,
            "at_risk_count": 0,
            "on_track_count": 1,
            "met_count": 0,
            "breach_rate": 0.0
        }
    }
    ```
    """,
    responses={
        200: {
            "description": "Dashboard data",
            "content": {
                "application/json": {
                    "example": DASHBOARD_RESPONSE_EXAMPLE
                }
            }
        }
    }
)
async def get_dashboard(
    customer_tier: Optional[str] = Query(None, description="Filter by customer tier (enterprise, business, standard, free)"),
    priority: Optional[str] = Query(None, description="Filter by priority (critical, high, medium, low)"),
    ticket_status: Optional[str] = Query(None, alias="status", description="Filter by ticket status"),
    sla_state: Optional[str] = Query(None, description="Filter by SLA state (breached, at_risk, on_track, met)"),
    limit: int = Query(100, ge=1, le=1000, description="Results per page"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: AsyncSession = Depends(get_session),
    sla_service: SLAService = Depends(get_sla_service)
):
    ticket_repo = SQLAlchemyTicketRepository(session)

    # Build filters
    filters = {}
    if customer_tier:
        filters["customer_tier"] = customer_tier
    if priority:
        filters["priority"] = priority
    if ticket_status:
        filters["status"] = ticket_status

    # Get tickets
    tickets = await ticket_repo.list(filters, limit=limit, offset=offset)

    # Calculate SLA for all tickets
    results = []
    state_counts = {"breached": 0, "at_risk": 0, "on_track": 0, "met": 0}

    for ticket_data in tickets:
        metrics = await sla_service.calculate_sla_metrics(str(ticket_data.id))
        if not metrics:
            continue

        # Filter by SLA state if requested
        if sla_state and metrics.most_urgent_state != sla_state:
            continue

        state_counts[metrics.most_urgent_state] += 1

        response_sla = SLAStatusResponse(
            deadline=metrics.response_deadline,
            remaining_seconds=metrics.response_remaining_seconds,
            percentage_remaining=metrics.response_percentage_remaining,
            is_breached=metrics.response_is_breached,
            state=metrics.response_state,
            met_at=metrics.response_met_at
        )

        resolution_sla = SLAStatusResponse(
            deadline=metrics.resolution_deadline,
            remaining_seconds=metrics.resolution_remaining_seconds,
            percentage_remaining=metrics.resolution_percentage_remaining,
            is_breached=metrics.resolution_is_breached,
            state=metrics.resolution_state,
            met_at=metrics.resolution_met_at
        )

        results.append(TicketSLAResponse(
            ticket_id=str(ticket_data.id),
            external_id=ticket_data.external_id,
            priority=ticket_data.priority,
            customer_tier=ticket_data.customer_tier,
            status=ticket_data.status,
            created_at=ticket_data.created_at,
            updated_at=ticket_data.updated_at,
            response_sla=response_sla,
            resolution_sla=resolution_sla,
            overall_state=metrics.most_urgent_state,
            next_deadline=metrics.next_deadline,
            active_alerts=[]
        ))

    total_count = len(results)
    breach_rate = (state_counts["breached"] / total_count * 100) if total_count > 0 else 0

    return DashboardResponse(
        tickets=results,
        total_count=total_count,
        summary=DashboardSummary(
            total_tickets=total_count,
            breached_count=state_counts["breached"],
            at_risk_count=state_counts["at_risk"],
            on_track_count=state_counts["on_track"],
            met_count=state_counts["met"],
            breach_rate=breach_rate
        )
    )


# Export router for inclusion in main app
sla_router = router
