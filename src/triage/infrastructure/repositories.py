"""
Triage Infrastructure Repositories
====================================

SQLAlchemy implementations of triage repositories.
"""

from typing import List, Optional, Any
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.triage.application import ITriageTicketRepository, IClassificationRepository, TriageTicketDTO
from src.triage.domain import ClassificationResult
from src.core import RepositoryException


class SQLAlchemyTriageTicketRepository(ITriageTicketRepository):
    """SQLAlchemy implementation for triage tickets."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_external_id(self, external_id: str) -> Optional[Any]:
        """Get ticket by external ID."""
        from src.triage.infrastructure.models import TriageTicketModel

        stmt = select(TriageTicketModel).where(
            TriageTicketModel.external_id == external_id
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, ticket_id: str) -> Optional[Any]:
        """Get ticket by internal ID."""
        from src.triage.infrastructure.models import TriageTicketModel

        try:
            ticket_uuid = UUID(ticket_id)
        except ValueError:
            return None

        stmt = select(TriageTicketModel).where(TriageTicketModel.id == ticket_uuid)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, ticket_dto: TriageTicketDTO) -> Any:
        """Create new ticket."""
        from src.triage.infrastructure.models import TriageTicketModel

        model = TriageTicketModel(
            id=uuid4(),
            external_id=ticket_dto.external_id,
            subject=ticket_dto.subject,
            content=ticket_dto.content,
            created_at=ticket_dto.created_at,
            updated_at=ticket_dto.updated_at
        )

        self._session.add(model)
        await self._session.flush()

        return model

    async def exists_by_external_id(self, external_id: str) -> bool:
        """Check if ticket exists by external ID."""
        from src.triage.infrastructure.models import TriageTicketModel

        stmt = select(TriageTicketModel.id).where(
            TriageTicketModel.external_id == external_id
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None


class SQLAlchemyClassificationRepository(IClassificationRepository):
    """SQLAlchemy implementation for classification results."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, result: ClassificationResult, ticket_id: str) -> Any:
        """Store classification result."""
        from src.triage.infrastructure.models import ClassificationModel

        try:
            ticket_uuid = UUID(ticket_id)
        except ValueError:
            raise RepositoryException(f"Invalid ticket ID: {ticket_id}")

        model = ClassificationModel(
            id=uuid4(),
            ticket_id=ticket_uuid,
            product=result.product,
            urgency=result.urgency,
            confidence=result.confidence,
            reasoning=result.reasoning,
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            created_at=result.timestamp
        )

        self._session.add(model)
        await self._session.flush()

        return model
