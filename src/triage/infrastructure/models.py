"""
Triage Infrastructure Models
=============================

SQLAlchemy ORM models for the triage module.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, DateTime, Integer, Float, Text, Uuid, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from infrastructure.database import Base
from config import ProductCategory, UrgencyLevel


class TriageTicketModel(Base):
    """
    Database model for TriageTicket entity.

    Stores ticket content for classification and RAG operations.
    """
    __tablename__ = "triage_tickets"

    # Primary key
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # Business identifier (optional, for linking to external tickets)
    external_id: Mapped[Optional[str]] = mapped_column(String(255), index=True, nullable=True)

    # Ticket content
    subject: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class ClassificationModel(Base):
    """
    Database model for ClassificationResult entity.

    Stores classification history for tickets.
    """
    __tablename__ = "classifications"

    # Primary key
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # Foreign key to ticket
    ticket_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("triage_tickets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Classification results
    product: Mapped[ProductCategory] = mapped_column(String(50), nullable=False)
    urgency: Mapped[UrgencyLevel] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
