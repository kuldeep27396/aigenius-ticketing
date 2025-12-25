"""
SLA Application Services
=========================

Application services orchestrate business logic and coordinate between
domain entities and repositories.

Following SOLID principles:
- Single Responsibility: Each service has one clear purpose
- Dependency Inversion: Depend on abstractions (repositories), not concrete implementations
"""

from datetime import datetime, timezone
from typing import List, Optional, Any
from abc import ABC, abstractmethod

from sla.domain import (
    Ticket, SLAMetrics, SLAAlert,
    SLACalculator, SLAConfig
)
from config import Priority, CustomerTier, SLAType, SLAState, TicketStatus


# ========== Repository Interfaces (Dependency Inversion) ==========

class ITicketRepository(ABC):
    """Interface for ticket data access."""

    @abstractmethod
    async def get_by_external_id(self, external_id: str) -> Optional[Any]:
        """Get ticket by external ID."""

    @abstractmethod
    async def get_by_id(self, ticket_id: str) -> Optional[Any]:
        """Get ticket by internal ID."""

    @abstractmethod
    async def create(self, ticket_dto: Any) -> Any:
        """Create new ticket."""

    @abstractmethod
    async def update(self, ticket_dto: Any) -> Any:
        """Update existing ticket."""

    @abstractmethod
    async def exists_by_external_id(self, external_id: str) -> bool:
        """Check if ticket exists by external ID."""

    @abstractmethod
    async def list(
        self,
        filters: dict,
        limit: int = 100,
        offset: int = 0
    ) -> List[Any]:
        """List tickets with filters."""


class ISLAAlertRepository(ABC):
    """Interface for SLA alert data access."""

    @abstractmethod
    async def create(self, alert: SLAAlert) -> SLAAlert:
        """Create new alert."""

    @abstractmethod
    async def get_pending_alerts(self, ticket_id: Optional[str] = None) -> List[SLAAlert]:
        """Get alerts that haven't been sent yet."""

    @abstractmethod
    async def mark_sent(self, alert_id: str, sent_at: datetime) -> None:
        """Mark alert as sent."""


class ISLAConfigProvider(ABC):
    """Interface for SLA configuration access."""

    @abstractmethod
    def get_config(self) -> SLAConfig:
        """Get current SLA configuration."""


# ========== Application Services ==========

class SLAService:
    """
    Service for SLA calculations and ticket tracking.

    Coordinates between domain logic and data access.
    """

    def __init__(
        self,
        ticket_repository: Optional[ITicketRepository],
        config_provider: ISLAConfigProvider
    ):
        self._ticket_repo = ticket_repository
        self._config_provider = config_provider

    async def calculate_sla_metrics(
        self,
        ticket_id: str
    ) -> Optional[SLAMetrics]:
        """
        Calculate SLA metrics for a ticket.

        Args:
            ticket_id: Internal ticket UUID

        Returns:
            SLAMetrics or None if ticket not found
        """
        if self._ticket_repo is None:
            raise ValueError("Ticket repository not configured")

        ticket_data = await self._ticket_repo.get_by_id(ticket_id)
        if not ticket_data:
            return None

        return await self._calculate_metrics_for_ticket(ticket_data)

    async def calculate_sla_metrics_for_tickets(
        self,
        ticket_ids: List[str]
    ) -> dict[str, SLAMetrics]:
        """
        Calculate SLA metrics for multiple tickets.

        Args:
            ticket_ids: List of ticket UUIDs

        Returns:
            Dict mapping ticket_id to SLAMetrics
        """
        results = {}

        for ticket_id in ticket_ids:
            metrics = await self.calculate_sla_metrics(ticket_id)
            if metrics:
                results[ticket_id] = metrics

        return results

    async def _calculate_metrics_for_ticket(self, ticket_data: Any) -> SLAMetrics:
        """Calculate SLA metrics for a single ticket."""
        config = self._config_provider.get_config()
        current_time = datetime.now(timezone.utc)

        # Get ticket attributes
        ticket_id = getattr(ticket_data, 'id', '')
        priority = getattr(ticket_data, 'priority', Priority.MEDIUM)
        customer_tier = getattr(ticket_data, 'customer_tier', CustomerTier.STANDARD)
        created_at = getattr(ticket_data, 'created_at', current_time)
        first_response_at = getattr(ticket_data, 'first_response_at', None)
        resolved_at = getattr(ticket_data, 'resolved_at', None)

        # Calculate response SLA
        response_minutes = config.get_sla_minutes(
            priority,
            customer_tier,
            SLAType.RESPONSE
        )
        response_deadline = SLACalculator.calculate_deadline(
            created_at, priority, customer_tier, SLAType.RESPONSE, response_minutes
        )

        response_remaining, response_pct, response_breached = SLACalculator.calculate_remaining_metrics(
            created_at, response_deadline, current_time, first_response_at
        )

        response_state = SLACalculator.calculate_status(
            created_at, response_deadline, current_time, first_response_at,
            config.get_warning_threshold()
        )

        # Calculate resolution SLA
        resolution_minutes = config.get_sla_minutes(
            priority,
            customer_tier,
            SLAType.RESOLUTION
        )
        resolution_deadline = SLACalculator.calculate_deadline(
            created_at, priority, customer_tier, SLAType.RESOLUTION, resolution_minutes
        )

        resolution_remaining, resolution_pct, resolution_breached = SLACalculator.calculate_remaining_metrics(
            created_at, resolution_deadline, current_time, resolved_at
        )

        resolution_state = SLACalculator.calculate_status(
            created_at, resolution_deadline, current_time, resolved_at,
            config.get_warning_threshold()
        )

        return SLAMetrics(
            ticket_id=ticket_id,
            response_deadline=response_deadline,
            response_remaining_seconds=response_remaining,
            response_percentage_remaining=response_pct,
            response_is_breached=response_breached,
            response_state=response_state,
            response_met_at=first_response_at,
            resolution_deadline=resolution_deadline,
            resolution_remaining_seconds=resolution_remaining,
            resolution_percentage_remaining=resolution_pct,
            resolution_is_breached=resolution_breached,
            resolution_state=resolution_state,
            resolution_met_at=resolved_at
        )


class SLAEvaluationService:
    """
    Service for evaluating SLA compliance and generating alerts.

    Run periodically to check all tickets and create alerts as needed.
    """

    def __init__(
        self,
        ticket_repository: ITicketRepository,
        alert_repository: ISLAAlertRepository,
        config_provider: ISLAConfigProvider
    ):
        self._ticket_repo = ticket_repository
        self._alert_repo = alert_repository
        self._config_provider = config_provider

    async def evaluate_all_tickets(self) -> List[SLAAlert]:
        """
        Evaluate SLA for all open tickets and create alerts if needed.

        Returns:
            List of new alerts created
        """
        # Get all open tickets
        open_tickets = await self._ticket_repo.list({
            "status": [TicketStatus.OPEN, TicketStatus.IN_PROGRESS,
                      TicketStatus.PENDING_CUSTOMER, TicketStatus.PENDING_VENDOR]
        })

        new_alerts = []

        for ticket_data in open_tickets:
            ticket_id = getattr(ticket_data, 'id', '')
            alerts = await self._evaluate_ticket(ticket_data)
            new_alerts.extend(alerts)

        return new_alerts

    async def _evaluate_ticket(self, ticket_data: Any) -> List[SLAAlert]:
        """Evaluate a single ticket and create alerts if needed."""
        from uuid import uuid4

        config = self._config_provider.get_config()
        current_time = datetime.now(timezone.utc)

        ticket_id = getattr(ticket_data, 'id', '')
        external_id = getattr(ticket_data, 'external_id', '')
        priority = getattr(ticket_data, 'priority', Priority.MEDIUM)
        customer_tier = getattr(ticket_data, 'customer_tier', CustomerTier.STANDARD)
        created_at = getattr(ticket_data, 'created_at', current_time)
        first_response_at = getattr(ticket_data, 'first_response_at', None)
        resolved_at = getattr(ticket_data, 'resolved_at', None)

        alerts = []

        # Evaluate response SLA
        response_minutes = config.get_sla_minutes(
            priority, customer_tier, SLAType.RESPONSE
        )
        response_deadline = SLACalculator.calculate_deadline(
            created_at, priority, customer_tier, SLAType.RESPONSE, response_minutes
        )
        response_state = SLACalculator.calculate_status(
            created_at, response_deadline, current_time, first_response_at,
            config.get_warning_threshold()
        )

        alert = self._should_create_alert(
            ticket_id, external_id, SLAType.RESPONSE,
            response_state, response_deadline, config
        )
        if alert:
            alerts.append(alert)

        # Evaluate resolution SLA
        resolution_minutes = config.get_sla_minutes(
            priority, customer_tier, SLAType.RESOLUTION
        )
        resolution_deadline = SLACalculator.calculate_deadline(
            created_at, priority, customer_tier, SLAType.RESOLUTION, resolution_minutes
        )
        resolution_state = SLACalculator.calculate_status(
            created_at, resolution_deadline, current_time, resolved_at,
            config.get_warning_threshold()
        )

        alert = self._should_create_alert(
            ticket_id, external_id, SLAType.RESOLUTION,
            resolution_state, resolution_deadline, config
        )
        if alert:
            alerts.append(alert)

        return alerts

    def _should_create_alert(
        self,
        ticket_id: str,
        external_id: str,
        sla_type: SLAType,
        state: SLAState,
        deadline: datetime,
        config: SLAConfig
    ) -> Optional[SLAAlert]:
        """Determine if alert should be created."""
        from uuid import uuid4
        from config import AlertType

        if state == SLAState.MET:
            return None

        current_time = datetime.now(timezone.utc)
        remaining = (deadline - current_time).total_seconds()
        remaining = max(0, remaining)

        if state == SLAState.BREACHED:
            alert_type = AlertType.BREACH
        elif state == SLAState.AT_RISK:
            alert_type = AlertType.WARNING
        else:
            return None

        return SLAAlert(
            id=str(uuid4()),
            ticket_id=ticket_id,
            ticket_external_id=external_id,
            sla_type=sla_type,
            alert_type=alert_type,
            triggered_at=current_time,
            deadline=deadline,
            remaining_seconds=remaining,
            state=state,
            channels=config.get_channels_for_level(1)
        )
