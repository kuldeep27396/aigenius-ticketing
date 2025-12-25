"""
Triage External Service Adapters
==================================

Adapters for external services (LLM, Vector Store) used by the triage module.

Implements the interfaces defined in the application layer using concrete
external service implementations.
"""

from typing import List, Any
from dataclasses import dataclass

from triage.application import IVectorStore, ILLMClient
from infrastructure.llm import ZAIILLMClient, EmbeddingResult, ChatCompletionResult
from infrastructure.vectorstore import MilvusVectorStore, SearchResult, Document
from core import LLMException, VectorStoreException


class LLMClientAdapter(ILLMClient):
    """
    Adapter that wraps the infrastructure LLM client.

    Implements the application layer ILLMClient interface using
    the infrastructure layer ZAIILLMClient.
    """

    def __init__(self, api_key: str | None = None):
        self._client = ZAIILLMClient(api_key)

    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        operation: str = "chat_completion"
    ) -> Any:
        """Generate chat completion."""
        return await self._client.chat_completion(messages, temperature, max_tokens, operation)


class VectorStoreAdapter(IVectorStore):
    """
    Adapter that wraps the infrastructure vector store.

    Implements the application layer IVectorStore interface using
    the infrastructure layer MilvusVectorStore.
    """

    def __init__(self, collection_name: str | None = None):
        self._store = MilvusVectorStore(collection_name=collection_name)

    async def initialize(self) -> None:
        """Initialize the vector store."""
        await self._store.initialize()

    async def get_document_count(self) -> int:
        """Get number of documents in the collection."""
        return await self._store.get_document_count()

    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search for similar documents.

        First generates an embedding for the query, then searches.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of search result dicts with content and metadata
        """
        from infrastructure.llm import ZAIILLMClient
        from config import settings

        # Generate query embedding
        llm_client = ZAIILLMClient(settings.zai_api_key)
        embedding_result = await llm_client.generate_embedding(query)

        # Search vector store
        search_results = await self._store.search(embedding_result.embedding, top_k)

        # Convert to dict format expected by application layer
        return [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score
            }
            for r in search_results
        ]


class DocumentIngester:
    """
    Service for ingesting documents into the vector store.

    Handles:
    - Web scraping (placeholder for P0)
    - Text chunking
    - Embedding generation
    - Vector storage
    """

    def __init__(self, vector_store: MilvusVectorStore):
        self._vector_store = vector_store

    async def ingest_url(
        self,
        url: str,
        max_pages: int = 10
    ) -> dict:
        """
        Ingest documents from a URL.

        For P0, this is a placeholder. Full implementation would:
        1. Fetch pages from the URL
        2. Extract text content
        3. Chunk the text
        4. Generate embeddings
        5. Store in vector database

        Args:
            url: Base URL to scrape
            max_pages: Maximum number of pages to process

        Returns:
            Ingestion statistics
        """
        # TODO: Implement actual web scraping
        # For now, return placeholder
        return {
            "url": url,
            "status": "not_implemented",
            "documents_processed": 0,
            "chunks_created": 0,
            "message": "Web scraping not yet implemented for P0"
        }

    async def ingest_text(
        self,
        text: str,
        metadata: dict,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> dict:
        """
        Ingest plain text directly.

        Useful for testing or direct document uploads.

        Args:
            text: Text to ingest
            metadata: Metadata dict (url, title, etc.)
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            Ingestion statistics
        """
        import uuid
        from infrastructure.llm import ZAIILLMClient
        from config import settings

        # Chunk the text
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)

        if not chunks:
            return {
                "status": "success",
                "chunks_created": 0,
                "message": "No content to chunk"
            }

        # Generate embeddings
        llm_client = ZAIILLMClient(settings.zai_api_key)
        embeddings = []
        for chunk in chunks:
            result = await llm_client.generate_embedding(chunk)
            embeddings.append(result.embedding)

        # Create documents
        documents = [
            Document(
                id=str(uuid.uuid4()),
                text=chunk,
                embedding=emb,
                metadata=metadata
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        # Store in vector store
        await self._vector_store.add_documents(documents)

        return {
            "status": "success",
            "chunks_created": len(chunks),
            "message": f"Successfully ingested {len(chunks)} chunks"
        }

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Split text into chunks for embedding.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            # Try to find a good break point
            if end < text_length:
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Look for line break
                    line_break = text.rfind("\n", start, end)
                    if line_break > start + chunk_size // 2:
                        end = line_break + 1
                    else:
                        # Look for sentence break
                        sentence_break = text.rfind(". ", start, end)
                        if sentence_break > start + chunk_size // 2:
                            end = sentence_break + 2

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - chunk_overlap if end < text_length else text_length

        return chunks
