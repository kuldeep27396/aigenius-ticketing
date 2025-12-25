"""
Shared Kernel Module
====================

This module contains shared infrastructure and domain elements used across
all bounded contexts (SLA Monitoring and Ticket Triage).

Architecture Pattern: Modular Monolith
- Each module (sla, triage) is a bounded context
- Shared kernel contains only generic infrastructure
- Domain models are extended within each module

DO NOT add business logic from SLA or Triage to shared kernel.
"""

__version__ = "1.0.0"
