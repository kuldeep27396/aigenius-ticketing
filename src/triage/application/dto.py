"""
Triage Application DTOs
========================

Data Transfer Objects for Triage API layer.

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

from triage.domain import Citation


# ========== Type Aliases for Literals ==========
ProductCategoryStr = Literal["CASB", "SWG", "ZTNA", "DLP", "SSPM", "CFW", "GENERAL"]
UrgencyLevelStr = Literal["critical", "high", "medium", "low"]


# ========== Request DTOs ==========

class ClassifyRequest(BaseModel):
    """Request model for ticket classification."""
    ticket_id: Optional[str] = Field(None, description="External ticket ID")
    subject: str = Field(..., min_length=1, description="Ticket subject")
    content: str = Field(..., min_length=1, description="Ticket content")

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        """Ensure content is not too long for LLM."""
        if len(v) > 10000:
            raise ValueError("Content too long (max 10000 characters)")
        return v


class RespondRequest(BaseModel):
    """Request model for RAG response generation."""
    ticket_id: Optional[str] = Field(None, description="External ticket ID")
    query: str = Field(..., min_length=1, description="Customer query")

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        """Ensure query is not too long."""
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 characters)")
        return v


class IngestDocumentsRequest(BaseModel):
    """Request model for document ingestion."""
    url: str = Field(..., description="URL to ingest documents from")
    max_pages: int = Field(default=10, ge=1, le=100, description="Maximum pages to process")


# ========== Response DTOs ==========

class ClassificationInfo(BaseModel):
    """Classification result information."""
    product: ProductCategoryStr
    urgency: UrgencyLevelStr
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class ClassificationResponse(BaseModel):
    """Response model for ticket classification."""
    ticket_id: str = Field(..., description="Internal ticket UUID")
    classification: ClassificationInfo
    processing_time_ms: int


class CitationInfo(BaseModel):
    """Citation information in API response."""
    index: int
    url: str
    title: str
    snippet: str


class RespondResponse(BaseModel):
    """Response model for RAG generation."""
    ticket_id: Optional[str]
    response: str
    citations: List[CitationInfo]
    sources_used: int
    processing_time_ms: int


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    message: str
    documents_processed: int = 0
    chunks_created: int = 0
    url: Optional[str] = None


class StatsResponse(BaseModel):
    """Response model for triage statistics."""
    classifications_total: int
    documents_indexed: int
    product_distribution: Dict[str, int]


# ========== Domain DTOs ==========

class TriageTicketDTO(BaseModel):
    """DTO for passing ticket data between layers."""
    id: Optional[str]
    external_id: Optional[str]
    subject: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    def to_domain(self) -> Any:
        """Convert to domain entity."""
        from triage.domain import TriageTicket
        return TriageTicket(
            id=self.id,
            external_id=self.external_id,
            subject=self.subject,
            content=self.content,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

    @classmethod
    def from_domain(cls, ticket: Any) -> "TriageTicketDTO":
        """Create from domain entity."""
        return cls(
            id=ticket.id,
            external_id=ticket.external_id,
            subject=ticket.subject,
            content=ticket.content,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at
        )
