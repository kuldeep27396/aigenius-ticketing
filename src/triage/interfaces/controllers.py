"""
Triage Controllers (API Routes)
================================

FastAPI routes for ticket triage endpoints.

Controllers delegate to application services.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_session
from src.triage.application import (
    ClassificationService, RAGService,
    ClassifyRequest, ClassificationResponse,
    RespondRequest, RespondResponse,
    IngestDocumentsRequest, IngestResponse,
    StatsResponse, TriageTicketDTO,
    ITriageTicketRepository, IClassificationRepository
)
from src.triage.infrastructure import (
    SQLAlchemyTriageTicketRepository,
    SQLAlchemyClassificationRepository,
    LLMClientAdapter,
    VectorStoreAdapter,
    DocumentIngester
)
from src.shared.infrastructure.logging import get_logger
from src.config import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/triage", tags=["Ticket Triage"])


# ========== Example payloads for Swagger ==========

CLASSIFY_REQUEST_EXAMPLE = {
    "ticket_id": "TICKET-001",
    "subject": "CASB Salesforce sync issue",
    "content": "Our CASB integration with Salesforce has stopped syncing data since yesterday."
}

CLASSIFY_RESPONSE_EXAMPLE = {
    "ticket_id": "123e4567-e89b-12d3-a456-426614174000",
    "classification": {
        "product": "CASB",
        "urgency": "high",
        "confidence": 0.92,
        "reasoning": "Ticket mentions CASB and Salesforce integration - core product issue with sync failure affecting operations."
    },
    "processing_time_ms": 1500
}

RESPOND_REQUEST_EXAMPLE = {
    "ticket_id": "TICKET-001",
    "query": "How do I configure CASB for Salesforce integration?"
}

RESPOND_RESPONSE_EXAMPLE = {
    "ticket_id": "TICKET-001",
    "response": "To configure CASB for Salesforce integration, follow these steps:\n\n1. Navigate to Settings > CASB > Salesforce\n2. Enter your Salesforce credentials\n3. Configure sync preferences\n4. Test the connection [1]",
    "citations": [
        {
            "index": 1,
            "url": "https://docs.example.com/casb-salesforce",
            "title": "CASB Salesforce Integration Guide",
            "snippet": "Navigate to Settings > CASB > Salesforce to configure the integration..."
        }
    ],
    "sources_used": 1,
    "processing_time_ms": 2500
}

STATS_RESPONSE_EXAMPLE = {
    "classifications_total": 150,
    "documents_indexed": 500,
    "product_distribution": {
        "CASB": 60,
        "SWG": 30,
        "ZTNA": 25,
        "DLP": 20,
        "SSPM": 10,
        "CFW": 5,
        "GENERAL": 0
    }
}


# ========== Dependencies ==========

def get_classification_service() -> ClassificationService:
    """Get classification service (singleton for now)."""
    from src.main import classification_service
    if classification_service is None:
        raise HTTPException(
            status_code=503,
            detail="Classification service not available - LLM not configured"
        )
    return classification_service


async def get_rag_service(request: Request) -> RAGService:
    """Get RAG service with vector store from app state."""
    from src.main import llm_client

    if not hasattr(request.app.state, "vector_store"):
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized"
        )

    if llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM client not configured"
        )

    vector_store_adapter = VectorStoreAdapter()

    return RAGService(llm_client, vector_store_adapter)


async def get_vector_store(request: Request):
    """Get vector store from app state."""
    if not hasattr(request.app.state, "vector_store"):
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized"
        )
    return request.app.state.vector_store


# ========== Route Handlers ==========

@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify a ticket by product and urgency",
    description="""
    Classify a support ticket using GLM 4.7 LLM to determine:
    - **Product**: Which product category (CASB, SWG, ZTNA, DLP, SSPM, CFW, GENERAL)
    - **Urgency**: How critical (critical, high, medium, low)

    **Products**:
    - `CASB` - Cloud Access Security Broker
    - `SWG` - Secure Web Gateway
    - `ZTNA` - Zero Trust Network Access
    - `DLP` - Data Loss Prevention
    - `SSPM` - SaaS Security Posture Management
    - `CFW` - Cloud Firewall
    - `GENERAL` - General inquiries

    **Example Request**:
    ```json
    {
        "ticket_id": "TICKET-001",
        "subject": "CASB Salesforce sync issue",
        "content": "Our CASB integration with Salesforce has stopped syncing data since yesterday."
    }
    ```

    **Example Response**:
    ```json
    {
        "ticket_id": "uuid",
        "classification": {
            "product": "CASB",
            "urgency": "high",
            "confidence": 0.92,
            "reasoning": "Ticket mentions CASB and Salesforce integration - core product issue with sync failure affecting operations."
        },
        "processing_time_ms": 1500
    }
    ```
    """,
    responses={
        200: {
            "description": "Ticket classified successfully",
            "content": {
                "application/json": {
                    "example": CLASSIFY_RESPONSE_EXAMPLE
                }
            }
        },
        503: {
            "description": "LLM service not available"
        }
    }
)
async def classify_ticket(
    request: Request,
    payload: ClassifyRequest,
    db: AsyncSession = Depends(get_session),
    service: ClassificationService = Depends(get_classification_service)
):
    start_time = time.perf_counter()
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.info(
        "Classifying ticket",
        extra={
            "correlation_id": correlation_id,
            "has_ticket_id": payload.ticket_id is not None
        }
    )

    # Get or create ticket record
    ticket_uuid = None
    ticket_repo = SQLAlchemyTriageTicketRepository(db)
    classification_repo = SQLAlchemyClassificationRepository(db)

    if payload.ticket_id:
        # Look up existing ticket
        ticket = await ticket_repo.get_by_external_id(payload.ticket_id)

        if ticket:
            ticket_uuid = str(ticket.id)
        else:
            # Create new ticket record
            ticket_dto = TriageTicketDTO(
                id=None,
                external_id=payload.ticket_id,
                subject=payload.subject,
                content=payload.content,
                created_at=datetime.now(timezone.utc)
            )
            ticket = await ticket_repo.create(ticket_dto)
            ticket_uuid = str(ticket.id)
    else:
        # Create ticket without external ID
        ticket_dto = TriageTicketDTO(
            id=None,
            external_id=None,
            subject=payload.subject,
            content=payload.content,
            created_at=datetime.now(timezone.utc)
        )
        ticket = await ticket_repo.create(ticket_dto)
        ticket_uuid = str(ticket.id)

    # Run classification
    result = await service.classify(
        subject=payload.subject or "",
        content=payload.content,
        ticket_id=payload.ticket_id
    )

    # Store classification result
    await classification_repo.create(result, ticket_uuid)

    await db.commit()

    total_time = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "Ticket classified",
        extra={
            "correlation_id": correlation_id,
            "ticket_uuid": ticket_uuid,
            "product": result.product,
            "urgency": result.urgency,
            "confidence": result.confidence
        }
    )

    from src.triage.application.dto import ClassificationInfo
    return ClassificationResponse(
        ticket_id=ticket_uuid,
        classification=ClassificationInfo(
            product=result.product,
            urgency=result.urgency,
            confidence=result.confidence,
            reasoning=result.reasoning
        ),
        processing_time_ms=total_time
    )


@router.post(
    "/respond",
    response_model=RespondResponse,
    summary="Generate response (RAG)",
    description="""
    Generate a response to a support query using RAG with Milvus vector store.

    The endpoint:
    1. Searches the vector database for relevant documentation
    2. Uses retrieved context to generate a response with GLM 4.7
    3. Includes citations to source documents

    **Citations**: Response includes [1], [2] markers that reference the citations list.

    Note: Requires documents to be indexed in the vector store.
    """,
    responses={
        200: {
            "description": "Response generated successfully",
            "content": {
                "application/json": {
                    "example": RESPOND_RESPONSE_EXAMPLE
                }
            }
        },
        503: {
            "description": "Vector store or LLM not available"
        }
    }
)
async def generate_response(
    request: Request,
    payload: RespondRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    start_time = time.perf_counter()
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.info(
        "Generating RAG response",
        extra={
            "correlation_id": correlation_id,
            "query_preview": payload.query[:100]
        }
    )

    # Check if we have documents
    vector_store_adapter = VectorStoreAdapter()
    await vector_store_adapter.initialize()
    doc_count = await vector_store_adapter.get_document_count()

    if doc_count == 0:
        return RespondResponse(
            ticket_id=payload.ticket_id,
            response="No documentation has been indexed yet. "
                     "Please use the POST /ingest endpoint to add documents first.",
            citations=[],
            sources_used=0,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000)
        )

    # Generate response
    try:
        result = await rag_service.generate_response(payload.query)

        # Convert citations
        from src.triage.application.dto import CitationInfo
        citations = [
            CitationInfo(
                index=c.index,
                url=c.url,
                title=c.title,
                snippet=c.snippet
            )
            for c in result.citations
        ]

        total_time = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "RAG response generated",
            extra={
                "correlation_id": correlation_id,
                "sources_used": result.sources_used,
                "latency_ms": total_time
            }
        )

        return RespondResponse(
            ticket_id=payload.ticket_id,
            response=result.response,
            citations=citations,
            sources_used=result.sources_used,
            processing_time_ms=total_time
        )

    except Exception as e:
        logger.error(
            "RAG generation failed",
            extra={
                "correlation_id": correlation_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Response generation failed: {str(e)}"
        )


@router.post(
    "/ingest",
    summary="Ingest documents for RAG",
    description="""
    Ingest documentation from a URL for use in RAG responses.

    This endpoint:
    1. Fetches documents from the provided URL
    2. Chunks the text for embedding
    3. Stores in the Milvus vector database for retrieval

    **Note**: For P0, web scraping is a placeholder. Full implementation
    would include web scraping, PDF parsing, etc.
    """,
    responses={
        200: {
            "description": "Ingestion status",
            "content": {
                "application/json": {
                    "example": {
                        "status": "not_implemented",
                        "message": "Web scraping not yet implemented for P0",
                        "documents_processed": 0,
                        "url": "https://docs.example.com"
                    }
                }
            }
        }
    }
)
async def ingest_documents(
    request: Request,
    ingest_request: IngestDocumentsRequest,
    vector_store = Depends(get_vector_store)
):
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.info(
        "Document ingestion requested",
        extra={
            "correlation_id": correlation_id,
            "url": ingest_request.url,
            "max_pages": ingest_request.max_pages
        }
    )

    ingester = DocumentIngester(vector_store)
    result = await ingester.ingest_url(ingest_request.url, ingest_request.max_pages)

    return IngestResponse(**result)


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get triage statistics",
    description="""
    Get statistics about the triage system:
    - Number of classifications performed
    - Document count in vector store
    - Classification distribution by product
    """,
    responses={
        200: {
            "description": "Statistics",
            "content": {
                "application/json": {
                    "example": STATS_RESPONSE_EXAMPLE
                }
            }
        }
    }
)
async def get_stats(
    request: Request,
    db: AsyncSession = Depends(get_session),
    vector_store = Depends(get_vector_store)
):
    from sqlalchemy import func, select
    from src.triage.infrastructure.models import ClassificationModel

    # Get classification counts
    stmt = select(func.count(ClassificationModel.id))
    result = await db.execute(stmt)
    classification_count = result.scalar()

    # Get document count
    doc_count = await vector_store.get_document_count()

    # Get product distribution
    stmt = select(
        ClassificationModel.product,
        func.count(ClassificationModel.id)
    ).group_by(ClassificationModel.product)
    result = await db.execute(stmt)
    product_distribution = {row[0]: row[1] for row in result.all()}

    return StatsResponse(
        classifications_total=classification_count,
        documents_indexed=doc_count,
        product_distribution=product_distribution
    )


# Export router for inclusion in main app
triage_router = router
