"""
SLA Application DTOs
=====================

Data Transfer Objects for SLA API layer.

These Pydantic models handle serialization/deserialization and validation
for API requests and responses. Following YAGNI - only what's needed.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Literal
from datetime import datetime


# ========== Type Aliases for Literals ==========
PriorityStr = Literal["critical", "high", "medium", "low"]
CustomerTierStr = Literal["enterprise", "business", "standard", "free"]
TicketStatusStr = Literal["open", "in_progress", "pending_customer", "pending_vendor", "resolved", "closed"]
SLATypeStr = Literal["response", "resolution"]
SLAStateStr = Literal["on_track", "at_risk", "breached", "met"]
AlertTypeStr = Literal["warning", "breach"]


# ========== Request DTOs ==========

class TicketIngestRequest(BaseModel):
    """Request model for ticket ingestion."""
    tickets: List["TicketCreateDTO"] = Field(
        ...,
        description="List of tickets to ingest"
    )


class TicketCreateDTO(BaseModel):
    """DTO for creating a single ticket."""
    id: str = Field(..., min_length=1, description="Unique ticket ID")
    priority: PriorityStr = Field(..., description="Ticket priority")
    customer_tier: CustomerTierStr = Field(..., description="Customer tier")
    status: TicketStatusStr = Field(default="open", description="Ticket status")
    subject: str = Field(..., min_length=1, description="Ticket subject")
    content: str = Field(..., min_length=1, description="Ticket content")
    created_at: datetime = Field(..., description="Ticket creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    first_response_at: Optional[datetime] = Field(None, description="First response time")
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")
    closed_at: Optional[datetime] = Field(None, description="Close time")

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, v: datetime, info) -> datetime:
        """Ensure updated_at is not before created_at."""
        if "created_at" in info.data and v < info.data["created_at"]:
            raise ValueError("updated_at cannot be before created_at")
        return v


class TicketUpdateDTO(BaseModel):
    """DTO for updating a ticket."""
    status: Optional[TicketStatusStr] = None
    subject: Optional[str] = Field(None, min_length=1)
    content: Optional[str] = Field(None, min_length=1)
    updated_at: datetime = Field(..., description="Update timestamp")
    first_response_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class DashboardQueryDTO(BaseModel):
    """Query parameters for dashboard endpoint."""
    customer_tier: Optional[CustomerTierStr] = None
    priority: Optional[PriorityStr] = None
    status: Optional[TicketStatusStr] = None
    sla_state: Optional[SLAStateStr] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# ========== Response DTOs ==========

class SLAStatusResponse(BaseModel):
    """Response model for SLA status of a single clock."""
    deadline: datetime = Field(..., description="SLA deadline")
    remaining_seconds: float = Field(..., description="Time remaining (0 if breached/met)")
    percentage_remaining: float = Field(..., description="Percentage of time remaining")
    is_breached: bool = Field(..., description="Whether SLA is breached")
    state: SLAStateStr = Field(..., description="Current SLA state")
    met_at: Optional[datetime] = Field(None, description="When SLA was met")


class TicketSLAResponse(BaseModel):
    """Response model for ticket SLA information."""
    ticket_id: str = Field(..., description="Internal ticket UUID")
    external_id: str = Field(..., description="External ticket ID")
    priority: PriorityStr
    customer_tier: CustomerTierStr
    status: TicketStatusStr
    created_at: datetime
    updated_at: datetime

    # SLA information
    response_sla: SLAStatusResponse = Field(..., description="Response SLA status")
    resolution_sla: SLAStatusResponse = Field(..., description="Resolution SLA status")
    overall_state: SLAStateStr = Field(..., description="Overall SLA state")
    next_deadline: datetime = Field(..., description="Next upcoming deadline")

    # Alerts
    active_alerts: List["AlertResponse"] = Field(
        default_factory=list,
        description="Active SLA alerts"
    )


class AlertResponse(BaseModel):
    """Response model for SLA alert."""
    id: str = Field(..., description="Alert ID")
    ticket_id: str = Field(..., description="Ticket UUID")
    ticket_external_id: str = Field(..., description="External ticket ID")
    sla_type: SLATypeStr
    alert_type: AlertTypeStr
    triggered_at: datetime
    deadline: datetime
    remaining_seconds: float
    state: SLAStateStr
    escalation_level: int = 1
    notification_sent: bool = False
    channels: List[str] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    """Response model for dashboard."""
    tickets: List[TicketSLAResponse] = Field(..., description="List of tickets")
    total_count: int = Field(..., description="Total number of tickets matching filter")
    summary: "DashboardSummary" = Field(..., description="Summary statistics")


class DashboardSummary(BaseModel):
    """Summary statistics for dashboard."""
    total_tickets: int
    breached_count: int
    at_risk_count: int
    on_track_count: int
    met_count: int
    breach_rate: float = Field(..., description="Percentage of tickets breached")


class IngestResponse(BaseModel):
    """Response model for ticket ingestion."""
    created: int = Field(..., description="Number of new tickets created")
    updated: int = Field(..., description="Number of existing tickets updated")
    failed: int = Field(default=0, description="Number of failed ingestions")
    errors: List[str] = Field(default_factory=list, description="Error messages")


# Forward references
TicketIngestRequest.model_rebuild()
TicketSLAResponse.model_rebuild()
DashboardResponse.model_rebuild()
AlertResponse.model_rebuild()


# ========== Domain Entity DTOs (for internal use) ==========

class TicketEntityDTO(BaseModel):
    """
    DTO representing ticket as passed between application and domain layers.

    This bridges the gap between domain entities and infrastructure models.
    """
    id: Optional[str]  # UUID, None for new tickets
    external_id: str
    priority: PriorityStr
    customer_tier: CustomerTierStr
    status: TicketStatusStr
    subject: str
    content: str
    created_at: datetime
    updated_at: datetime
    first_response_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    def to_domain(self) -> Any:
        """Convert to domain entity."""
        from src.sla.domain import Ticket
        from src.config import Priority, CustomerTier, TicketStatus

        # Convert string to enum
        priority_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW
        }
        tier_map = {
            "enterprise": CustomerTier.ENTERPRISE,
            "business": CustomerTier.BUSINESS,
            "standard": CustomerTier.STANDARD,
            "free": CustomerTier.FREE
        }
        status_map = {
            "open": TicketStatus.OPEN,
            "in_progress": TicketStatus.IN_PROGRESS,
            "pending_customer": TicketStatus.PENDING_CUSTOMER,
            "pending_vendor": TicketStatus.PENDING_VENDOR,
            "resolved": TicketStatus.RESOLVED,
            "closed": TicketStatus.CLOSED
        }

        return Ticket(
            id=self.id or "",
            external_id=self.external_id,
            priority=priority_map[self.priority],
            customer_tier=tier_map[self.customer_tier],
            status=status_map[self.status],
            subject=self.subject,
            content=self.content,
            created_at=self.created_at,
            updated_at=self.updated_at,
            first_response_at=self.first_response_at,
            resolved_at=self.resolved_at,
            closed_at=self.closed_at
        )

    @classmethod
    def from_domain(cls, ticket: Any) -> "TicketEntityDTO":
        """Create from domain entity."""
        return cls(
            id=ticket.id,
            external_id=ticket.external_id,
            priority=ticket.priority.value,
            customer_tier=ticket.customer_tier.value,
            status=ticket.status.value,
            subject=ticket.subject,
            content=ticket.content,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
            first_response_at=ticket.first_response_at,
            resolved_at=ticket.resolved_at,
            closed_at=ticket.closed_at
        )
