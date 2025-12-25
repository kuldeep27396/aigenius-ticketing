"""
Core Module
============

Shared core utilities and abstractions used across the application.

This module contains framework-agnostic code that defines the fundamental
building blocks of the system.
"""

from src.core.exceptions import (
    ApplicationException,
    DomainException,
    RepositoryException,
    ValidationException,
    ResourceNotFoundException,
    ConfigurationException,
    ExternalServiceException,
    LLMException,
    VectorStoreException,
    SLABreachException,
)

__all__ = [
    "ApplicationException",
    "DomainException",
    "RepositoryException",
    "ValidationException",
    "ResourceNotFoundException",
    "ConfigurationException",
    "ExternalServiceException",
    "LLMException",
    "VectorStoreException",
    "SLABreachException",
]
