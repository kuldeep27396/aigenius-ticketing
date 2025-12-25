"""
LLM Client Infrastructure
==========================

Wrapper for LLM providers (OpenAI, Z.AI) providing clean interface for LLM operations.

This module abstracts the LLM client implementation following the
Dependency Inversion Principle - the domain layer depends on abstractions,
not concrete implementations.
"""

import time
from typing import List, Optional, Any
from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from zai import ZaiClient

from src.config import settings
from src.core import LLMException, ConfigurationException
from src.shared.infrastructure.grafana import get_grafana_exporter


class EmbeddingResult:
    """Result of an embedding generation."""

    def __init__(self, embedding: List[float], model: str):
        self.embedding = embedding
        self.model = model
        self.dimension = len(embedding)


class ChatCompletionResult:
    """Result of a chat completion."""

    def __init__(
        self,
        content: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int
    ):
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.latency_ms = latency_ms


class ILLMClient(ABC):
    """
    Interface for LLM client operations.

    Following Interface Segregation Principle - only methods
    actually needed by the application are defined.
    """

    @abstractmethod
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for text."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        operation: str = "chat_completion"
    ) -> ChatCompletionResult:
        """Generate chat completion."""


class ZAIILLMClient(ILLMClient):
    """
    Z.AI SDK client implementation for GLM 4.7.

    Provides async wrapper around Z.AI SDK operations.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.zai_api_key
        if not self._api_key:
            raise ConfigurationException("Z.AI API key not configured")

        self._client = ZaiClient(api_key=self._api_key)
        self._model = settings.llm_model
        self._embedding_model = settings.embedding_model

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for text using Z.AI embedding model.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector

        Raises:
            LLMException: If embedding generation fails
        """
        try:
            response = self._client.embeddings.create(
                model=self._embedding_model,
                input=text
            )
            return EmbeddingResult(
                embedding=response.data[0].embedding,
                model=self._embedding_model
            )
        except Exception as e:
            raise LLMException(f"Embedding generation failed: {str(e)}")

    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        operation: str = "chat_completion"
    ) -> ChatCompletionResult:
        """
        Generate chat completion using GLM 4.7.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            operation: Operation type for metrics (chat_completion, classification, rag)

        Returns:
            ChatCompletionResult with generated text

        Raises:
            LLMException: If completion fails
        """
        start_time = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            content = response.choices[0].message.content

            # Z.AI doesn't return token usage, so we estimate
            prompt_tokens = len(str(messages))
            completion_tokens = len(content)

            result = ChatCompletionResult(
                content=content,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms
            )

            # Export metrics to Grafana
            exporter = get_grafana_exporter()
            if exporter and exporter.is_enabled():
                await exporter.export_llm_metrics(
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    operation=operation
                )

            return result

        except Exception as e:
            raise LLMException(f"Chat completion failed: {str(e)}")


class OpenAILLMClient(ILLMClient):
    """
    OpenAI client implementation for GPT models.

    Provides async wrapper around OpenAI SDK operations.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.openai_api_key
        if not self._api_key:
            raise ConfigurationException("OpenAI API key not configured")

        self._client = AsyncOpenAI(api_key=self._api_key)
        self._model = settings.llm_model
        self._embedding_model = settings.embedding_model

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for text using OpenAI embedding model.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector

        Raises:
            LLMException: If embedding generation fails
        """
        try:
            response = await self._client.embeddings.create(
                model=self._embedding_model,
                input=text
            )
            return EmbeddingResult(
                embedding=response.data[0].embedding,
                model=self._embedding_model
            )
        except Exception as e:
            raise LLMException(f"Embedding generation failed: {str(e)}")

    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        operation: str = "chat_completion"
    ) -> ChatCompletionResult:
        """
        Generate chat completion using OpenAI GPT.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            operation: Operation type for metrics (chat_completion, classification, rag)

        Returns:
            ChatCompletionResult with generated text

        Raises:
            LLMException: If completion fails
        """
        start_time = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            result = ChatCompletionResult(
                content=content,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms
            )

            # Export metrics to Grafana
            exporter = get_grafana_exporter()
            if exporter and exporter.is_enabled():
                await exporter.export_llm_metrics(
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    operation=operation
                )

            return result

        except Exception as e:
            raise LLMException(f"Chat completion failed: {str(e)}")


class GroqLLMClient(ILLMClient):
    """
    Groq client implementation for Llama models.

    Groq is OpenAI-compatible with ultra-fast inference.
    Base URL: https://api.groq.com/openai/v1
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.groq_api_key
        if not self._api_key:
            raise ConfigurationException("Groq API key not configured")

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self._model = settings.llm_model
        self._embedding_model = settings.embedding_model

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for text.

        Note: Groq doesn't provide embedding API, so we use a mock implementation
        or fallback to OpenAI embeddings if configured.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        # For now, return mock embeddings since Groq doesn't have embedding API
        dimension = settings.embedding_dimension
        import hashlib
        # Create deterministic pseudo-embedding based on text hash
        hash_obj = hashlib.sha256(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        import random
        random.seed(seed)
        embedding = [random.uniform(-1, 1) for _ in range(dimension)]
        return EmbeddingResult(embedding=embedding, model="groq-mock-embedding")

    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        operation: str = "chat_completion"
    ) -> ChatCompletionResult:
        """
        Generate chat completion using Groq Llama models.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            operation: Operation type for metrics (chat_completion, classification, rag)

        Returns:
            ChatCompletionResult with generated text

        Raises:
            LLMException: If completion fails
        """
        start_time = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            result = ChatCompletionResult(
                content=content,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms
            )

            # Export metrics to Grafana
            exporter = get_grafana_exporter()
            if exporter and exporter.is_enabled():
                await exporter.export_llm_metrics(
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    operation=operation
                )

            return result

        except Exception as e:
            raise LLMException(f"Chat completion failed: {str(e)}")


class MockLLMClient(ILLMClient):
    """
    Mock LLM client for testing.

    Returns predictable responses without calling external APIs.
    """

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Return mock embedding (zero vector)."""
        dimension = settings.embedding_dimension
        return EmbeddingResult(
            embedding=[0.0] * dimension,
            model="mock-embedding"
        )

    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        operation: str = "chat_completion"
    ) -> ChatCompletionResult:
        """Return mock response based on operation type."""

        # Check if this is a classification request
        user_content = str(messages[-1].get("content", "")) if messages else ""

        if "classification" in operation.lower() or "classify" in operation.lower() or "product" in user_content.lower():
            # Mock classification response
            import json
            mock_response = {
                "product": "CASB",
                "urgency": "high",
                "confidence": 0.92,
                "reasoning": "Mock: Ticket mentions CASB product with high urgency based on content analysis."
            }
            content = f"```json\n{json.dumps(mock_response, indent=2)}\n```"
        elif "rag" in operation.lower() or "response" in operation.lower():
            # Mock RAG response
            content = """Based on the documentation, here's how to configure CASB for Salesforce integration:

1. Navigate to Settings > CASB > Salesforce
2. Enter your Salesforce credentials
3. Configure sync preferences (real-time or scheduled)
4. Test the connection [1]

For troubleshooting, check the integration logs in the CASB dashboard."""
        else:
            # Generic mock response
            content = "This is a mock LLM response for testing purposes."

        return ChatCompletionResult(
            content=content,
            model="mock-model",
            prompt_tokens=100,
            completion_tokens=len(content.split()),
            latency_ms=100
        )
