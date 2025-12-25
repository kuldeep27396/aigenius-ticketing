"""
SLA Value Objects
==================

Immutable value objects for SLA domain.

Value objects are defined by their attributes rather than an identity.
They are immutable and can be freely shared.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from config import (
    Priority, CustomerTier, SLAType, SLAState, AlertType,
    VALID_PRIORITIES, VALID_CUSTOMER_TIERS, VALID_SLA_TYPES
)


class SLACalculator:
    """
    Pure functions for SLA calculations.

    Stateless utility class following DRY principle -
    all SLA calculation logic in one place.
    """

    @staticmethod
    def calculate_deadline(
        created_at: datetime,
        priority: Priority,
        customer_tier: CustomerTier,
        sla_type: SLAType,
        sla_minutes: int
    ) -> datetime:
        """
        Calculate SLA deadline for a ticket.

        Args:
            created_at: When ticket was created
            priority: Ticket priority
            customer_tier: Customer tier
            sla_type: Type of SLA (response/resolution)
            sla_minutes: Base SLA in minutes (already adjusted for tier)

        Returns:
            The SLA deadline
        """
        return created_at + timedelta(minutes=sla_minutes)

    @staticmethod
    def calculate_status(
        created_at: datetime,
        deadline: datetime,
        current_time: datetime,
        met_at: Optional[datetime] = None,
        warning_threshold_percent: int = 15
    ) -> SLAState:
        """
        Calculate current SLA state.

        Args:
            created_at: When ticket was created
            deadline: The SLA deadline
            current_time: Current time for evaluation
            met_at: When SLA was met (first response/resolution)
            warning_threshold_percent: Percentage threshold for "at_risk"

        Returns:
            SLAState: Current SLA state
        """
        # If SLA was already met
        if met_at and met_at <= deadline:
            return SLAState.MET

        # Calculate remaining time
        remaining = (deadline - current_time).total_seconds()
        total = (deadline - created_at).total_seconds()

        if total <= 0:
            percentage = 0
        else:
            percentage = (remaining / total) * 100

        is_breached = remaining <= 0

        # Determine status
        if is_breached:
            return SLAState.BREACHED
        elif percentage <= warning_threshold_percent:
            return SLAState.AT_RISK
        else:
            return SLAState.ON_TRACK

    @staticmethod
    def calculate_remaining_metrics(
        created_at: datetime,
        deadline: datetime,
        current_time: datetime,
        met_at: Optional[datetime] = None
    ) -> tuple[float, float, bool]:
        """
        Calculate remaining time metrics.

        Returns:
            Tuple of (remaining_seconds, percentage_remaining, is_breached)
        """
        if met_at and met_at <= deadline:
            return 0.0, 0.0, False

        remaining = (deadline - current_time).total_seconds()
        total = (deadline - created_at).total_seconds()

        if total <= 0:
            percentage = 0.0
        else:
            percentage = max(0, min(100, (remaining / total) * 100))

        is_breached = remaining <= 0

        return max(0, remaining), percentage, is_breached

    @staticmethod
    def should_alert(
        current_state: SLAState,
        previous_state: Optional[SLAState] = None,
        warning_threshold: int = 15
    ) -> Optional[AlertType]:
        """
        Determine if an alert should be created.

        Args:
            current_state: Current SLA state
            previous_state: Previous SLA state (if any)
            warning_threshold: Warning threshold percentage

        Returns:
            AlertType if alert needed, None otherwise
        """
        if current_state == SLAState.BREACHED:
            # Alert on breach or state change to breach
            if previous_state != SLAState.BREACHED:
                return AlertType.BREACH
        elif current_state == SLAState.AT_RISK:
            # Alert on entering at_risk state
            if previous_state not in (SLAState.AT_RISK, SLAState.BREACHED):
                return AlertType.WARNING

        return None


class EscalationLevelConfig(BaseModel):
    """Configuration for a single escalation level."""
    level: int = Field(ge=1, description="Escalation level (1-based)")
    notify: List[str] = Field(default_factory=list, description="Slack channels")


class SLAConfig(BaseModel):
    """
    SLA Configuration loaded from YAML.

    Effective SLA = Base SLA × Customer Tier Multiplier

    This is a value object - immutable and defined by its attributes.
    """
    sla_targets: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="SLA targets in minutes by priority"
    )
    customer_tier_multipliers: Dict[str, float] = Field(
        default_factory=dict,
        description="SLA multipliers by customer tier"
    )
    escalation_thresholds: Dict[str, int] = Field(
        default={"warning": 15, "breach": 0},
        description="Percentage thresholds for warning/breach"
    )
    escalation_levels: List[EscalationLevelConfig] = Field(
        default_factory=lambda: [EscalationLevelConfig(level=1, notify=["#support-alerts"])],
        description="Notification config per escalation level"
    )

    @field_validator("sla_targets")
    @classmethod
    def validate_sla_targets(cls, v: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        """Validate SLA targets have required priorities."""
        # Use VALID_PRIORITIES and VALID_SLA_TYPES lists instead of iterating
        required_priorities = VALID_PRIORITIES
        required_sla_types = VALID_SLA_TYPES

        for priority in required_priorities:
            if priority not in v:
                v[priority] = {"response": 60, "resolution": 480}

            for sla_type in required_sla_types:
                if sla_type not in v[priority]:
                    v[priority][sla_type] = 60 if sla_type == "response" else 480

        return v

    @field_validator("customer_tier_multipliers")
    @classmethod
    def validate_tier_multipliers(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate customer tier multipliers."""
        required_tiers = VALID_CUSTOMER_TIERS

        for tier in required_tiers:
            if tier not in v:
                # Default multipliers
                defaults = {
                    "enterprise": 0.5,
                    "business": 0.75,
                    "standard": 1.0,
                    "free": 1.5
                }
                v[tier] = defaults.get(tier, 1.0)

        return v

    def get_sla_minutes(
        self,
        priority: str,
        customer_tier: str,
        sla_type: str
    ) -> int:
        """
        Calculate SLA target in minutes.

        Formula: Base SLA (from priority) × Customer Tier Multiplier

        Example:
            Priority "high" response = 60 minutes
            Customer tier "enterprise" multiplier = 0.5
            Effective SLA = 60 × 0.5 = 30 minutes
        """
        base_minutes = self.sla_targets.get(priority, {}).get(sla_type, 60)
        multiplier = self.customer_tier_multipliers.get(customer_tier, 1.0)
        return int(base_minutes * multiplier)

    def get_warning_threshold(self) -> int:
        """Get percentage threshold for warning alerts."""
        return self.escalation_thresholds.get("warning", 15)

    def get_breach_threshold(self) -> int:
        """Get percentage threshold for breach (usually 0)."""
        return self.escalation_thresholds.get("breach", 0)

    def get_channels_for_level(self, level: int) -> List[str]:
        """Get Slack channels to notify for given escalation level."""
        for esc in self.escalation_levels:
            if esc.level == level:
                return esc.notify
        return []


@dataclass(frozen=True)
class SLADeadline:
    """
    Immutable value object representing an SLA deadline.

    Can be used as a domain event or for passing deadline information.
    """
    ticket_id: str
    sla_type: SLAType
    deadline: datetime
    created_at: datetime
    priority: Priority
    customer_tier: CustomerTier

    @property
    def is_past(self) -> bool:
        """Check if deadline is in the past."""
        return datetime.now(self.deadline.tzinfo) > self.deadline

    @property
    def minutes_until_deadline(self) -> float:
        """Get minutes until deadline (negative if past)."""
        delta = self.deadline - datetime.now(self.deadline.tzinfo)
        return delta.total_seconds() / 60
