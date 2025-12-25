"""
SLA Infrastructure Models
==========================

SQLAlchemy ORM models for SLA module.

These are the database representations of our domain entities.
They belong in the infrastructure layer, not the domain layer.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, DateTime, Boolean, Integer, Float, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database import Base
from src.config import Priority, CustomerTier, TicketStatus, SLAType, SLAState, AlertType


class TicketModel(Base):
    """
    Database model for Ticket entity.

    Maps to the 'tickets' table.
    """
    __tablename__ = "tickets"

    # Primary key
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # Business identifier (external ticket ID)
    external_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)

    # SLA attributes
    priority: Mapped[Priority] = mapped_column(String(50), nullable=False, default=Priority.MEDIUM)
    customer_tier: Mapped[CustomerTier] = mapped_column(String(50), nullable=False, default=CustomerTier.STANDARD)
    status: Mapped[TicketStatus] = mapped_column(String(50), nullable=False, default=TicketStatus.OPEN)

    # Ticket content
    subject: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # SLA tracking
    first_response_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Unique constraint on (external_id, updated_at) for idempotent ingest
    __table_args__ = (
        # This is handled at application level for SQLAlchemy 2.0
    )


class AlertModel(Base):
    """
    Database model for SLA Alert entity.

    Maps to the 'sla_alerts' table.
    """
    __tablename__ = "sla_alerts"

    # Primary key
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # Ticket reference
    ticket_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    ticket_external_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Alert details
    sla_type: Mapped[SLAType] = mapped_column(String(50), nullable=False)  # response or resolution
    alert_type: Mapped[AlertType] = mapped_column(String(50), nullable=False)  # warning or breach
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    remaining_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    state: Mapped[SLAState] = mapped_column(String(50), nullable=False)

    # Escalation
    escalation_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Notification tracking
    notification_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notification_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
