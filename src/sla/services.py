"""
SLA Services
============

Legacy SLA services for evaluation and alerting.

This file contains services that coordinate SLA evaluation and
Slack notifications. These services work with the new clean architecture
structure while maintaining compatibility with existing code.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from sla.domain import SLAAlert, SLAConfig
from sla.infrastructure.external import SlackClient, SlackMessage
from sla.domain.value_objects import SLACalculator
from config import (
    Priority, CustomerTier, SLAType, SLAState, TicketStatus, AlertType
)


class SLAEvaluator:
    """
    Evaluates SLA compliance for all tickets and generates alerts.

    This service:
    1. Queries all open tickets
    2. Calculates current SLA status
    3. Creates alerts for at-risk or breached tickets
    4. Sends Slack notifications for alerts
    """

    def __init__(
        self,
        config_manager,  # SLAConfigManager from external.py
        slack_client: SlackClient
    ):
        self._config_manager = config_manager
        self._slack_client = slack_client

    async def evaluate(self, session: AsyncSession) -> dict:
        """
        Evaluate all open tickets and send alerts.

        Args:
            session: Database session

        Returns:
            Summary of evaluation results
        """
        from sla.infrastructure.models import TicketModel, AlertModel
        from sqlalchemy import select

        config = self._config_manager.config
        current_time = datetime.now(timezone.utc)

        # Get all open tickets
        stmt = select(TicketModel).where(
            TicketModel.status.in_([
                TicketStatus.OPEN,
                TicketStatus.IN_PROGRESS,
                TicketStatus.PENDING_CUSTOMER,
                TicketStatus.PENDING_VENDOR
            ])
        )
        result = await session.execute(stmt)
        tickets = result.scalars().all()

        alerts_created = 0
        notifications_sent = 0

        for ticket in tickets:
            # Evaluate and create alerts
            ticket_alerts = await self._evaluate_ticket(ticket, config, current_time)

            for alert_data in ticket_alerts:
                # Create alert in database
                alert = AlertModel(
                    id=uuid4(),
                    ticket_id=ticket.id,
                    ticket_external_id=ticket.external_id,
                    sla_type=alert_data["sla_type"],
                    alert_type=alert_data["alert_type"],
                    triggered_at=current_time,
                    deadline=alert_data["deadline"],
                    remaining_seconds=alert_data["remaining_seconds"],
                    state=alert_data["state"],
                    escalation_level=1
                )
                session.add(alert)
                alerts_created += 1

                # Send Slack notification
                if await self._send_slack_notification(ticket, alert_data):
                    notifications_sent += 1

        await session.commit()

        return {
            "tickets_evaluated": len(tickets),
            "alerts_created": alerts_created,
            "notifications_sent": notifications_sent
        }

    async def _evaluate_ticket(
        self,
        ticket,
        config: SLAConfig,
        current_time: datetime
    ) -> List[dict]:
        """Evaluate a single ticket and return alert data if needed."""
        alerts = []

        # Evaluate response SLA
        response_minutes = config.get_sla_minutes(
            ticket.priority, ticket.customer_tier, SLAType.RESPONSE
        )
        response_deadline = SLACalculator.calculate_deadline(
            ticket.created_at, ticket.priority, ticket.customer_tier,
            SLAType.RESPONSE, response_minutes
        )
        response_state = SLACalculator.calculate_status(
            ticket.created_at, response_deadline, current_time,
            ticket.first_response_at, config.get_warning_threshold()
        )

        if response_state != SLAState.ON_TRACK:
            remaining = max(0, (response_deadline - current_time).total_seconds())
            alerts.append({
                "sla_type": SLAType.RESPONSE,
                "alert_type": AlertType.BREACH if response_state == SLAState.BREACHED else AlertType.WARNING,
                "deadline": response_deadline,
                "remaining_seconds": remaining,
                "state": response_state
            })

        # Evaluate resolution SLA
        resolution_minutes = config.get_sla_minutes(
            ticket.priority, ticket.customer_tier, SLAType.RESOLUTION
        )
        resolution_deadline = SLACalculator.calculate_deadline(
            ticket.created_at, ticket.priority, ticket.customer_tier,
            SLAType.RESOLUTION, resolution_minutes
        )
        resolution_state = SLACalculator.calculate_status(
            ticket.created_at, resolution_deadline, current_time,
            ticket.resolved_at, config.get_warning_threshold()
        )

        if resolution_state != SLAState.ON_TRACK:
            remaining = max(0, (resolution_deadline - current_time).total_seconds())
            alerts.append({
                "sla_type": SLAType.RESOLUTION,
                "alert_type": AlertType.BREACH if resolution_state == SLAState.BREACHED else AlertType.WARNING,
                "deadline": resolution_deadline,
                "remaining_seconds": remaining,
                "state": resolution_state
            })

        return alerts

    async def _send_slack_notification(
        self,
        ticket,
        alert_data: dict
    ) -> bool:
        """Send Slack notification for an alert."""
        sla_type = alert_data["sla_type"]
        alert_type = alert_data["alert_type"]
        deadline = alert_data["deadline"]
        remaining = alert_data["remaining_seconds"]

        # Calculate remaining percentage
        total_minutes = (deadline - ticket.created_at).total_seconds() / 60
        remaining_pct = (remaining / total_minutes * 100) if total_minutes > 0 else 0

        message = SlackMessage(
            ticket_id=str(ticket.id),
            priority=ticket.priority,
            customer_tier=ticket.customer_tier,
            sla_type=sla_type,
            alert_type=alert_type,
            remaining_percentage=remaining_pct,
            escalation_level=1,
            status=ticket.status,
            created_at=ticket.created_at.isoformat(),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        return await self._slack_client.send_alert(message)


class SLAService:
    """
    SLA Service for compatibility with existing code.

    Delegates to the new architecture's SLAService.
    """

    def __init__(self, config_manager):
        self._config_manager = config_manager

    @property
    def config(self) -> SLAConfig:
        """Get SLA configuration."""
        return self._config_manager.config
