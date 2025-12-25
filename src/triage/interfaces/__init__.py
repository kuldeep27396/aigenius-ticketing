"""
Triage Interfaces Layer
========================

Interface adapters (controllers) for ticket triage module.

Contains:
- Controllers: FastAPI route handlers
"""

from src.triage.interfaces.controllers import triage_router

__all__ = ["triage_router"]
