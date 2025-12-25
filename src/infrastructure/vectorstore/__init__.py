"""
Vector Store Infrastructure
============================

Milvus vector store implementation for document storage and retrieval.

This module provides a clean interface for vector operations following
the Repository pattern.
"""

import time
from typing import List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pymilvus import MilvusClient

from src.config import settings
from src.core import VectorStoreException


@dataclass
class Document:
    """Document for vector storage."""
    id: str
    text: str
    embedding: List[float]
    metadata: dict


@dataclass
class SearchResult:
    """Result from vector search."""
    content: str
    metadata: dict
    score: float
    id: Optional[str] = None


class IVectorStore(ABC):
    """
    Interface for vector store operations.

    Following Interface Segregation and Dependency Inversion principles.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""

    @abstractmethod
    async def get_document_count(self) -> int:
        """Get number of documents in the collection."""

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents."""

    @abstractmethod
    async def delete_by_source(self, source_url: str) -> int:
        """Delete all documents from a specific source."""


class MilvusVectorStore(IVectorStore):
    """
    Zilliz Cloud (Managed Milvus) implementation of vector store.

    Uses Zilliz Cloud service for vector storage and retrieval.

    For correct connection, you need to find your cluster's Public Endpoint
    in the Zilliz Cloud Console. It should look like:
    https://inxxxxxxxxxxxxxxxxx.aws-us-west-2.vectordb-uat3.zillizcloud.com
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        uri: Optional[str] = None
    ):
        self._collection_name = collection_name or settings.milvus_collection_name
        self._dimension = settings.embedding_dimension
        self._client: Optional[MilvusClient] = None
        self._initialized = False

        # For Zilliz Cloud connection:
        # 1. Use provided URI (if explicitly set)
        # 2. Use environment variable ZILLIZ_URI (if available)
        # 3. Fall back to public API endpoint (works with API key only for some clusters)
        if uri:
            self._uri = uri
        elif hasattr(settings, 'zilliz_uri') and settings.zilliz_uri:
            self._uri = settings.zilliz_uri
        elif settings.zilliz_api_key:
            # Use public endpoint - may not work for all cluster types
            self._uri = "https://api.zillizcloud.com"
        else:
            self._uri = "https://inference-api.zhipuai.cn/v1"

        self._api_key = settings.zilliz_api_key

    async def initialize(self) -> None:
        """Initialize Zilliz Cloud client and collection."""
        if self._initialized:
            return

        if not self._api_key:
            raise VectorStoreException("ZILLIZ_API_KEY not configured")

        try:
            self._client = MilvusClient(
                uri=self._uri,
                token=self._api_key
            )

            # Create collection if not exists
            if not self._client.has_collection(self._collection_name):
                self._client.create_collection(
                    collection_name=self._collection_name,
                    dimension=self._dimension
                )

            self._initialized = True

        except Exception as e:
            raise VectorStoreException(f"Failed to initialize Milvus: {str(e)}")

    async def get_document_count(self) -> int:
        """Get number of documents in the collection."""
        if not self._initialized:
            await self.initialize()

        if not self._client:
            return 0

        try:
            # Use query to get count - num_entities doesn't exist in some versions
            # For Zilliz Cloud serverless, use num_rows or query
            res = self._client.query(
                collection_name=self._collection_name,
                filter='',
                output_fields=["primary_key"],
                limit=1
            )
            return len(res)
        except Exception:
            return 0

    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects with embeddings

        Raises:
            VectorStoreException: If add operation fails
        """
        if not self._initialized:
            await self.initialize()

        if not self._client:
            raise VectorStoreException("Vector store not initialized")

        try:
            # Prepare data for Milvus
            data = []
            for doc in documents:
                data.append({
                    "id": doc.id,
                    "vector": doc.embedding,
                    "text": doc.text,
                    "url": doc.metadata.get("url", ""),
                    "title": doc.metadata.get("title", "")
                })

            # Insert into Milvus
            self._client.insert(
                collection_name=self._collection_name,
                data=data
            )

            # Flush to ensure data is persisted
            self._client.flush(self._collection_name)

        except Exception as e:
            raise VectorStoreException(f"Failed to add documents: {str(e)}")

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            VectorStoreException: If search fails
        """
        if not self._initialized:
            await self.initialize()

        if not self._client:
            raise VectorStoreException("Vector store not initialized")

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "url", "title"]
            )

            # Format results
            formatted_results = []
            if results and len(results) > 0 and len(results[0]) > 0:
                for hit in results[0]:
                    formatted_results.append(SearchResult(
                        content=hit["entity"]["text"],
                        metadata={
                            "url": hit["entity"].get("url", ""),
                            "title": hit["entity"].get("title", "")
                        },
                        score=hit["distance"],
                        id=hit.get("id")
                    ))

            return formatted_results

        except Exception as e:
            raise VectorStoreException(f"Search failed: {str(e)}")

    async def delete_by_source(self, source_url: str) -> int:
        """
        Delete all documents from a specific source URL.

        Note: This requires a more complex setup with Milvus.
        For P0, returning 0.

        Args:
            source_url: URL of the source to delete

        Returns:
            Number of documents deleted
        """
        # TODO: Implement with Milvus delete expressions
        return 0
