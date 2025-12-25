"""
Core Exceptions
================

Custom exceptions for the application following clean architecture principles.

These exceptions define domain-specific errors that can be caught and handled
appropriately at the application boundaries.
"""

from typing import Optional, Any
from dataclasses import dataclass


class ApplicationException(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class DomainException(ApplicationException):
    """Base exception for domain logic violations."""


class RepositoryException(ApplicationException):
    """Base exception for repository/data access errors."""


class ValidationException(ApplicationException):
    """Exception for validation errors."""


class ResourceNotFoundException(ApplicationException):
    """Exception when a requested resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[dict] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type}"
        if resource_id:
            message += f" with id '{resource_id}'"
        message += " not found"
        super().__init__(message, details)


class ConfigurationException(ApplicationException):
    """Exception for configuration errors."""


class ExternalServiceException(ApplicationException):
    """Base exception for external service failures."""

    def __init__(
        self,
        service_name: str,
        message: str,
        details: Optional[dict] = None
    ):
        self.service_name = service_name
        super().__init__(f"{service_name}: {message}", details)


class LLMException(ExternalServiceException):
    """Exception for LLM API failures."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__("LLM Service", message, details)


class VectorStoreException(ExternalServiceException):
    """Exception for vector store failures."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__("Vector Store", message, details)


class SLABreachException(DomainException):
    """Exception raised when an SLA breach is detected."""

    def __init__(
        self,
        ticket_id: str,
        sla_type: str,
        deadline: Any,
        details: Optional[dict] = None
    ):
        self.ticket_id = ticket_id
        self.sla_type = sla_type
        self.deadline = deadline
        super().__init__(
            f"SLA {sla_type} breached for ticket {ticket_id}",
            details or {"ticket_id": ticket_id, "sla_type": sla_type}
        )
