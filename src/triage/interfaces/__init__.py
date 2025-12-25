"""
Triage Interfaces Layer
========================

Interface adapters (controllers) for ticket triage module.

Contains:
- Controllers: FastAPI route handlers
"""

from triage.interfaces.controllers import triage_router

__all__ = ["triage_router"]
