"""
SLA Domain Entities
====================

Pure Python domain entities for SLA monitoring.

Following Domain-Driven Design principles, these entities contain
business logic and are free of infrastructure concerns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from enum import Enum

from src.config import (
    SLAType, SLAState, AlertType,
    Priority, CustomerTier, TicketStatus
)


@dataclass
class Ticket:
    """
    Ticket entity representing a support ticket.

    This is a domain entity that encapsulates the core business object
    for SLA tracking. Contains only domain logic, no infrastructure.
    """

    # Core attributes
    id: str
    external_id: str
    priority: Priority
    customer_tier: CustomerTier
    status: TicketStatus
    subject: str
    content: str

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Optional SLA tracking fields
    first_response_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate ticket on initialization."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")

        if self.first_response_at and self.first_response_at < self.created_at:
            raise ValueError("first_response_at cannot be before created_at")

        if self.resolved_at and self.resolved_at < self.created_at:
            raise ValueError("resolved_at cannot be before created_at")

    @property
    def age_minutes(self) -> int:
        """Get ticket age in minutes."""
        return int((datetime.now(timezone.utc) - self.created_at).total_seconds() / 60)

    @property
    def is_open(self) -> bool:
        """Check if ticket is still open."""
        return self.status in (TicketStatus.OPEN, TicketStatus.IN_PROGRESS,
                              TicketStatus.PENDING_CUSTOMER, TicketStatus.PENDING_VENDOR)

    @property
    def is_resolved(self) -> bool:
        """Check if ticket has been resolved."""
        return self.status in (TicketStatus.RESOLVED, TicketStatus.CLOSED)

    def mark_first_response(self, timestamp: Optional[datetime] = None) -> None:
        """Mark first response time."""
        if self.first_response_at is not None:
            return  # Already has first response
        self.first_response_at = timestamp or datetime.now(timezone.utc)

    def mark_resolved(self, timestamp: Optional[datetime] = None) -> None:
        """Mark ticket as resolved."""
        self.resolved_at = timestamp or datetime.now(timezone.utc)
        self.status = TicketStatus.RESOLVED
        self.updated_at = self.resolved_at

    def mark_closed(self, timestamp: Optional[datetime] = None) -> None:
        """Mark ticket as closed."""
        if not self.resolved_at:
            self.mark_resolved(timestamp)
        self.closed_at = timestamp or datetime.now(timezone.utc)
        self.status = TicketStatus.CLOSED
        self.updated_at = self.closed_at


@dataclass
class SLAMetrics:
    """
    SLA metrics for a ticket.

    Contains calculated SLA information including deadlines,
    remaining time, and breach status.
    """

    # Ticket reference
    ticket_id: str

    # Response SLA
    response_deadline: datetime
    response_remaining_seconds: float
    response_percentage_remaining: float
    response_is_breached: bool
    response_state: SLAState

    # Resolution SLA
    resolution_deadline: datetime
    resolution_remaining_seconds: float
    resolution_percentage_remaining: float
    resolution_is_breached: bool
    resolution_state: SLAState

    # Optional fields (with defaults must come last)
    response_met_at: Optional[datetime] = None
    resolution_met_at: Optional[datetime] = None

    # Overall status (computed field)
    is_any_breached: bool = field(init=False)

    def __post_init__(self):
        """Calculate overall breach status."""
        self.is_any_breached = self.response_is_breached or self.resolution_is_breached

    @property
    def most_urgent_state(self) -> SLAState:
        """Get the most urgent SLA state."""
        if self.is_any_breached:
            return SLAState.BREACHED
        if (self.response_state == SLAState.AT_RISK or
            self.resolution_state == SLAState.AT_RISK):
            return SLAState.AT_RISK
        if (self.response_state == SLAState.MET and
            self.resolution_state == SLAState.MET):
            return SLAState.MET
        return SLAState.ON_TRACK

    @property
    def next_deadline(self) -> datetime:
        """Get the next upcoming deadline."""
        if self.response_state not in (SLAState.MET, SLAState.BREACHED):
            return self.response_deadline
        if self.resolution_state not in (SLAState.MET, SLAState.BREACHED):
            return self.resolution_deadline
        # Both met or breached, return the earlier one
        return min(self.response_deadline, self.resolution_deadline)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "ticket_id": self.ticket_id,
            "response": {
                "deadline": self.response_deadline.isoformat(),
                "remaining_seconds": self.response_remaining_seconds,
                "percentage_remaining": self.response_percentage_remaining,
                "is_breached": self.response_is_breached,
                "state": self.response_state.value,
                "met_at": self.response_met_at.isoformat() if self.response_met_at else None
            },
            "resolution": {
                "deadline": self.resolution_deadline.isoformat(),
                "remaining_seconds": self.resolution_remaining_seconds,
                "percentage_remaining": self.resolution_percentage_remaining,
                "is_breached": self.resolution_is_breached,
                "state": self.resolution_state.value,
                "met_at": self.resolution_met_at.isoformat() if self.resolution_met_at else None
            },
            "overall": {
                "state": self.most_urgent_state.value,
                "is_any_breached": self.is_any_breached,
                "next_deadline": self.next_deadline.isoformat()
            }
        }


@dataclass
class SLAAlert:
    """
    SLA alert entity.

    Represents a notification that needs to be sent when SLA
    thresholds are crossed.
    """

    id: Optional[str]
    ticket_id: str
    ticket_external_id: str
    sla_type: SLAType
    alert_type: AlertType
    triggered_at: datetime
    deadline: datetime
    remaining_seconds: float
    state: SLAState

    # Escalation info
    escalation_level: int = 1
    notification_sent: bool = False
    notification_sent_at: Optional[datetime] = None

    # Channels
    channels: List[str] = field(default_factory=list)

    @property
    def is_notification_pending(self) -> bool:
        """Check if notification still needs to be sent."""
        return not self.notification_sent

    def mark_notification_sent(self, timestamp: Optional[datetime] = None) -> None:
        """Mark notification as sent."""
        self.notification_sent = True
        self.notification_sent_at = timestamp or datetime.now(timezone.utc)
