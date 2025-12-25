"""
Triage Domain Entities
======================

Domain entities for the ticket triage module.

Contains pure Python business objects for ticket classification
and RAG-based response generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum

from config import ProductCategory, UrgencyLevel


@dataclass
class ClassificationResult:
    """
    Result of ticket classification.

    Contains the product category and urgency level assigned by the LLM.
    """
    product: ProductCategory
    urgency: UrgencyLevel
    confidence: float  # 0.0 to 1.0
    reasoning: str
    model_used: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate classification result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class TriageTicket:
    """
    Ticket entity for triage operations.

    Contains the ticket content needed for classification and
    tracks the classification history.
    """
    id: Optional[str]  # UUID, None for new tickets
    external_id: Optional[str]  # External ticket reference
    subject: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Classification results (history)
    classifications: List[ClassificationResult] = field(default_factory=list)

    @property
    def latest_classification(self) -> Optional[ClassificationResult]:
        """Get the most recent classification."""
        return self.classifications[-1] if self.classifications else None

    @property
    def full_text(self) -> str:
        """Get combined subject and content for classification."""
        return f"{self.subject}\n\n{self.content}"

    def add_classification(self, result: ClassificationResult) -> None:
        """Add a classification result to history."""
        self.classifications.append(result)
        self.updated_at = datetime.utcnow()


@dataclass
class Citation:
    """
    Citation for RAG-generated responses.

    References the source document that supports part of the response.
    """
    index: int
    url: str
    title: str
    snippet: str  # Relevant excerpt from the source


@dataclass
class RAGResult:
    """
    Result of RAG-based response generation.

    Contains the generated response along with citations to sources.
    """
    response: str
    citations: List[Citation]
    sources_used: int
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_citations(self) -> bool:
        """Check if response has source citations."""
        return len(self.citations) > 0


class ClassificationPromptBuilder:
    """
    Builds prompts for ticket classification.

    Following DRY principle - all prompt logic in one place.
    """

    SYSTEM_PROMPT = """You are a ticket classification system for AIGenius Customer Support.

Your task is to analyze support tickets and classify them by:
1. Product Category: Which product is involved?
2. Urgency: How critical is this issue?

PRODUCT CATEGORIES:
- CASB: Cloud Access Security Broker issues ( Salesforce, Office 365, etc.)
- SWG: Secure Web Gateway issues (web filtering, SSL inspection, etc.)
- ZTNA: Zero Trust Network Access issues (remote access, private apps, etc.)
- DLP: Data Loss Prevention issues (policy violations, data classification, etc.)
- SSPM: SaaS Security Posture Management issues (configuration assessments, etc.)
- CFW: Cloud Firewall issues (firewall rules, NAT, etc.)
- GENERAL: General inquiries, billing, or issues not specific to a product

URGENCY LEVELS:
- critical: Production down, data breach, security incident
- high: Major feature broken, significant impact
- medium: Minor issues, workarounds available
- low: Questions, documentation requests, nice-to-have

Respond ONLY in JSON format:
{
    "product": "category",
    "urgency": "level",
    "confidence": 0.95,
    "reasoning": "brief explanation"
}"""

    @classmethod
    def build_prompt(cls, subject: str, content: str) -> str:
        """Build classification prompt from ticket content."""
        return f"""Subject: {subject}

Content:
{content}

Classify this ticket (respond with JSON only):"""

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for classification."""
        return cls.SYSTEM_PROMPT
