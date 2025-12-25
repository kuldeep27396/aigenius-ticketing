"""
SLA Interfaces Layer
====================

Interface adapters (controllers) for SLA monitoring module.

Contains:
- Controllers: FastAPI route handlers

This is the outermost layer - handles HTTP requests/responses and
delegates to application services.
"""

from sla.interfaces.controllers import sla_router

__all__ = ["sla_router"]
