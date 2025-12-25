"""
SLA Infrastructure Repositories
=================================

Concrete implementations of repository interfaces using SQLAlchemy.

This layer contains the data access logic - how we store and retrieve
entities from the database.
"""

from typing import List, Optional, Any
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from sla.application import (
    ITicketRepository, ISLAAlertRepository, ISLAConfigProvider,
    TicketEntityDTO
)
from sla.domain import SLAAlert
from sla.domain.value_objects import SLAConfig, EscalationLevelConfig
from config import TicketStatus
from core import RepositoryException


class SQLAlchemyTicketRepository(ITicketRepository):
    """
    SQLAlchemy implementation of ticket repository.

    Handles persistence of Ticket entities using async SQLAlchemy.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_external_id(self, external_id: str) -> Optional[Any]:
        """Get ticket by external ID."""
        from sla.infrastructure.models import TicketModel

        stmt = select(TicketModel).where(TicketModel.external_id == external_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, ticket_id: str) -> Optional[Any]:
        """Get ticket by internal ID."""
        from sla.infrastructure.models import TicketModel

        try:
            ticket_uuid = UUID(ticket_id)
        except ValueError:
            return None

        stmt = select(TicketModel).where(TicketModel.id == ticket_uuid)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, ticket_dto: TicketEntityDTO) -> Any:
        """Create new ticket."""
        from sla.infrastructure.models import TicketModel

        model = TicketModel(
            id=uuid4(),
            external_id=ticket_dto.external_id,
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

        self._session.add(model)
        await self._session.flush()

        return model

    async def update(self, ticket_dto: TicketEntityDTO) -> Any:
        """Update existing ticket."""
        from sla.infrastructure.models import TicketModel

        model = await self.get_by_external_id(ticket_dto.external_id)
        if not model:
            raise RepositoryException(f"Ticket {ticket_dto.external_id} not found")

        # Update fields
        model.priority = ticket_dto.priority
        model.customer_tier = ticket_dto.customer_tier
        model.status = ticket_dto.status
        model.subject = ticket_dto.subject
        model.content = ticket_dto.content
        model.updated_at = ticket_dto.updated_at
        model.first_response_at = ticket_dto.first_response_at
        model.resolved_at = ticket_dto.resolved_at
        model.closed_at = ticket_dto.closed_at

        await self._session.flush()

        return model

    async def exists_by_external_id(self, external_id: str) -> bool:
        """Check if ticket exists by external ID."""
        from sla.infrastructure.models import TicketModel

        stmt = select(TicketModel.id).where(TicketModel.external_id == external_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def list(
        self,
        filters: dict,
        limit: int = 100,
        offset: int = 0
    ) -> List[Any]:
        """List tickets with filters."""
        from sla.infrastructure.models import TicketModel

        stmt = select(TicketModel)

        # Apply filters
        conditions = []
        if "status" in filters:
            status_list = filters["status"]
            if isinstance(status_list, list):
                conditions.append(TicketModel.status.in_(status_list))
            else:
                conditions.append(TicketModel.status == status_list)

        if "customer_tier" in filters:
            conditions.append(TicketModel.customer_tier == filters["customer_tier"])

        if "priority" in filters:
            conditions.append(TicketModel.priority == filters["priority"])

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Order by created_at descending
        stmt = stmt.order_by(TicketModel.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        return result.scalars().all()


class SQLAlchemyAlertRepository(ISLAAlertRepository):
    """
    SQLAlchemy implementation of SLA alert repository.

    Handles persistence of SLAAlert entities.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, alert: SLAAlert) -> SLAAlert:
        """Create new alert."""
        from sla.infrastructure.models import AlertModel

        model = AlertModel(
            id=uuid4() if not alert.id else UUID(alert.id),
            ticket_id=UUID(alert.ticket_id),
            ticket_external_id=alert.ticket_external_id,
            sla_type=alert.sla_type,
            alert_type=alert.alert_type,
            triggered_at=alert.triggered_at,
            deadline=alert.deadline,
            remaining_seconds=alert.remaining_seconds,
            state=alert.state,
            escalation_level=alert.escalation_level,
            notification_sent=alert.notification_sent,
            notification_sent_at=alert.notification_sent_at
        )

        self._session.add(model)
        await self._session.flush()

        # Update alert with generated ID
        alert.id = str(model.id)

        return alert

    async def get_pending_alerts(self, ticket_id: Optional[str] = None) -> List[SLAAlert]:
        """Get alerts that haven't been sent yet."""
        from sla.infrastructure.models import AlertModel

        stmt = select(AlertModel).where(AlertModel.notification_sent == False)

        if ticket_id:
            try:
                ticket_uuid = UUID(ticket_id)
                stmt = stmt.where(AlertModel.ticket_id == ticket_uuid)
            except ValueError:
                return []

        stmt = stmt.order_by(AlertModel.triggered_at.asc())

        result = await self._session.execute(stmt)
        models = result.scalars().all()

        alerts = []
        for model in models:
            alerts.append(SLAAlert(
                id=str(model.id),
                ticket_id=str(model.ticket_id),
                ticket_external_id=model.ticket_external_id,
                sla_type=model.sla_type,
                alert_type=model.alert_type,
                triggered_at=model.triggered_at,
                deadline=model.deadline,
                remaining_seconds=model.remaining_seconds,
                state=model.state,
                escalation_level=model.escalation_level,
                notification_sent=model.notification_sent,
                notification_sent_at=model.notification_sent_at
            ))

        return alerts

    async def mark_sent(self, alert_id: str, sent_at: datetime) -> None:
        """Mark alert as sent."""
        from sla.infrastructure.models import AlertModel

        try:
            alert_uuid = UUID(alert_id)
        except ValueError:
            raise RepositoryException(f"Invalid alert ID: {alert_id}")

        stmt = select(AlertModel).where(AlertModel.id == alert_uuid)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            raise RepositoryException(f"Alert {alert_id} not found")

        model.notification_sent = True
        model.notification_sent_at = sent_at
        await self._session.flush()


class YAMLConfigProvider(ISLAConfigProvider):
    """
    SLA configuration provider that loads from YAML.

    Watches the file for changes and reloads automatically.
    """

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._config: Optional[SLAConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        import yaml
        from pathlib import Path

        config_file = Path(self._config_path)

        if not config_file.exists():
            # Return default config
            self._config = SLAConfig()
            return

        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)

        # Convert to SLAConfig
        escalation_levels = [
            EscalationLevelConfig(**e)
            for e in data.get('escalation_levels', [{'level': 1, 'notify': ['#support-alerts']}])
        ]

        self._config = SLAConfig(
            sla_targets=data.get('sla_targets', {}),
            customer_tier_multipliers=data.get('customer_tier_multipliers', {}),
            escalation_thresholds=data.get('escalation_thresholds', {'warning': 15, 'breach': 0}),
            escalation_levels=escalation_levels
        )

    def get_config(self) -> SLAConfig:
        """Get current SLA configuration."""
        return self._config

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
