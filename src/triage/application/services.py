"""
Triage Application Services
============================

Application services for ticket classification and RAG.

Orchestrates business logic between domain entities and repositories.
"""

import time
import json
from typing import Optional, List, Any
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from triage.domain import (
    ClassificationResult, TriageTicket, RAGResult, Citation,
    ClassificationPromptBuilder
)
from config import ProductCategory, UrgencyLevel
from core import LLMException, VectorStoreException


# ========== Repository Interfaces ==========

class ITriageTicketRepository(ABC):
    """Interface for triage ticket data access."""

    @abstractmethod
    async def get_by_external_id(self, external_id: str) -> Optional[Any]:
        """Get ticket by external ID."""

    @abstractmethod
    async def get_by_id(self, ticket_id: str) -> Optional[Any]:
        """Get ticket by internal ID."""

    @abstractmethod
    async def create(self, ticket_dto: Any) -> Any:
        """Create new ticket."""

    @abstractmethod
    async def exists_by_external_id(self, external_id: str) -> bool:
        """Check if ticket exists."""


class IClassificationRepository(ABC):
    """Interface for classification result storage."""

    @abstractmethod
    async def create(self, result: ClassificationResult, ticket_id: str) -> Any:
        """Store classification result."""


class IVectorStore(ABC):
    """Interface for vector store operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""

    @abstractmethod
    async def get_document_count(self) -> int:
        """Get number of documents."""

    @abstractmethod
    async def search(self, query: str, top_k: int) -> List[dict]:
        """Search for similar documents."""


class ILLMClient(ABC):
    """Interface for LLM operations."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        operation: str = "chat_completion"
    ) -> Any:
        """Generate chat completion."""


# ========== Application Services ==========

class ClassificationService:
    """
    Service for ticket classification using LLM.

    Coordinates between LLM client and repositories.
    """

    def __init__(self, llm_client: ILLMClient):
        self._llm = llm_client

    async def classify(
        self,
        subject: str,
        content: str,
        ticket_id: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a ticket by product and urgency.

        Args:
            subject: Ticket subject
            content: Ticket content
            ticket_id: Optional ticket ID for logging

        Returns:
            ClassificationResult with product, urgency, confidence
        """
        start_time = time.perf_counter()

        # Build prompt
        user_prompt = ClassificationPromptBuilder.build_prompt(subject, content)

        messages = [
            {"role": "system", "content": ClassificationPromptBuilder.get_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM
        try:
            response = await self._llm.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                operation="classification"
            )

            # Parse JSON response
            content_text = response.content
            # Try to extract JSON from response
            if "```json" in content_text:
                content_text = content_text.split("```json")[1].split("```")[0].strip()
            elif "```" in content_text:
                content_text = content_text.split("```")[1].split("```")[0].strip()

            result_data = json.loads(content_text)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return ClassificationResult(
                product=ProductCategory(result_data.get("product", ProductCategory.GENERAL)),
                urgency=UrgencyLevel(result_data.get("urgency", UrgencyLevel.MEDIUM)),
                confidence=float(result_data.get("confidence", 0.5)),
                reasoning=result_data.get("reasoning", ""),
                model_used=response.model,
                latency_ms=latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                timestamp=datetime.now(timezone.utc)
            )

        except json.JSONDecodeError as e:
            raise LLMException(f"Failed to parse classification response: {e}")
        except Exception as e:
            raise LLMException(f"Classification failed: {e}")


class RAGService:
    """
    Service for RAG-based response generation.

    Orchestrates vector search and LLM generation.
    """

    def __init__(
        self,
        llm_client: ILLMClient,
        vector_store: IVectorStore
    ):
        self._llm = llm_client
        self._vector_store = vector_store

    async def generate_response(
        self,
        query: str,
        top_k: int = 5
    ) -> RAGResult:
        """
        Generate a response to a query using RAG.

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            RAGResult with response and citations
        """
        start_time = time.perf_counter()

        # Search for relevant documents
        search_results = await self._vector_store.search(query, top_k=top_k)

        if not search_results:
            return RAGResult(
                response="I couldn't find relevant documentation to answer your question.",
                citations=[],
                sources_used=0,
                latency_ms=int((time.perf_counter() - start_time) * 1000)
            )

        # Build context from results
        context = self._build_context(search_results)

        # Generate response with LLM
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Context from documentation:

{context}

---

User Question: {query}

Please answer the question based on the context above. Include citation numbers [1], [2], etc. to reference the sources."""}
        ]

        try:
            response = await self._llm.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                operation="rag"
            )

            # Build citations
            citations = self._build_citations(search_results)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return RAGResult(
                response=response.content,
                citations=citations,
                sources_used=len(citations),
                latency_ms=latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                timestamp=datetime.now(timezone.utc)
            )

        except Exception as e:
            raise LLMException(f"RAG generation failed: {e}")

    def _get_system_prompt(self) -> str:
        """Get system prompt for RAG generation."""
        return """You are a helpful customer support assistant for AIGenius Ticketing System.

Your role is to answer questions based on the provided documentation context. Follow these guidelines:

1. ONLY use information from the provided context to answer questions
2. Include citation numbers [1], [2], etc. to reference your sources
3. If the context doesn't contain enough information, say so honestly
4. Be concise but thorough
5. Use clear, professional language
6. If multiple sources support a point, cite all of them
7. Format your response clearly with proper paragraphs

Remember: Accuracy is more important than completeness. Never make up information."""

    def _build_context(self, results: List[dict]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            content = result.get("content", "")
            part = f"[{i}] {metadata.get('title', 'Untitled')} ({metadata.get('url', 'unknown')})\n{content}"
            context_parts.append(part)

        return "\n\n".join(context_parts)

    def _build_citations(self, results: List[dict]) -> List[Citation]:
        """Build citation list from search results."""
        citations = []

        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            snippet = content[:200] + "..." if len(content) > 200 else content
            metadata = result.get("metadata", {})

            citations.append(Citation(
                index=i,
                url=metadata.get("url", ""),
                title=metadata.get("title", ""),
                snippet=snippet
            ))

        return citations
