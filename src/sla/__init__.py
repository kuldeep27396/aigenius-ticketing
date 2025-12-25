"""
SLA Monitoring Module
=====================

Bounded Context for Service Level Agreement monitoring and escalation.

Responsibilities:
- Calculate SLA deadlines based on priority and customer tier
- Evaluate tickets for SLA compliance
- Generate alerts when SLAs are at risk or breached
- Escalate via Slack notifications
- Provide dashboard API for SLA visibility

Functional Requirements Implemented:
- FR-1: POST /tickets (ticket ingestion)
- FR-2: PostgreSQL persistence with SLA fields
- FR-3: Background SLA evaluation engine
- FR-4: Slack escalation workflow
- FR-5: Config hot-reload via watchdog
- FR-6: GET /tickets/{id}, GET /dashboard
"""

__version__ = "1.0.0"
