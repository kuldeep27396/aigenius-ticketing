"""
AIGenius Ticketing - Main Application
=========================================

AI-powered Customer Support Ticketing System.

Modules:
- SLA Monitoring: Track service level agreements and escalations
- AI Triage: Classify tickets and generate RAG-based responses

Clean Architecture Layers:
- Interfaces: FastAPI controllers
- Application: Services and DTOs
- Domain: Entities and value objects
- Infrastructure: Database, LLM, vector store
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Configuration and Core
from config import settings
from core import ApplicationException

# Infrastructure
from infrastructure.database import init_database, close_database, create_tables
from infrastructure.llm import ZAIILLMClient
from infrastructure.vectorstore import MilvusVectorStore

# SLA Module - External services
from sla.infrastructure.external import (
    SLAConfigManager, SlackClient, SLAScheduler
)
from sla.infrastructure.repositories import SQLAlchemyTicketRepository
from sla.application import SLAService

# SLA Services for evaluation
from sla.services import SLAEvaluator

# Module Routers
from sla.interfaces import sla_router
from triage.interfaces import triage_router

# Logging
from shared.infrastructure.logging import setup_logging, get_logger
from shared.infrastructure.grafana import init_grafana_exporter

logger = get_logger(__name__)

# Global service instances
sla_config_manager = None
sla_service = None
classification_service = None
sla_scheduler = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.

    STARTUP:
    1. Setup structured logging
    2. Initialize database
    3. Create database tables
    4. Load SLA configuration
    5. Start SLA scheduler
    6. Initialize LLM client
    7. Initialize vector store

    SHUTDOWN:
    1. Stop SLA scheduler
    2. Close Slack client
    3. Close database connections
    """
    global sla_config_manager, sla_service, classification_service, sla_scheduler, llm_client

    # === STARTUP ===
    setup_logging()
    logger.info("Starting Ticket Service", extra={
        "version": settings.app_version,
        "environment": settings.environment
    })

    # Initialize database
    logger.info("Initializing database")
    init_database()

    # Create tables (for development - use Alembic in production)
    # Note: If database is not available, the server will start but
    # database-dependent endpoints will fail
    logger.info("Creating database tables")
    try:
        await create_tables()
    except Exception as e:
        logger.warning(f"Database not available - running in degraded mode: {e}")
        logger.warning("Please start PostgreSQL to enable full functionality")

    # Initialize SLA configuration manager
    logger.info("Loading SLA configuration")
    sla_config_manager = SLAConfigManager()
    sla_config_manager.load(settings.sla_config_path)
    sla_config_manager.start_watching()

    # Initialize SLA service
    sla_service = SLAService(None, sla_config_manager)

    # Initialize Slack client
    slack_client = SlackClient()

    # Initialize SLA evaluator and scheduler (optional, requires database)
    try:
        sla_evaluator = SLAEvaluator(sla_config_manager, slack_client)

        async def sla_evaluation_job():
            """Background SLA evaluation job."""
            from infrastructure.database import get_session_context
            async with get_session_context() as session:
                await sla_evaluator.evaluate(session)

        sla_scheduler = SLAScheduler(interval_seconds=settings.sla_evaluation_interval)
        await sla_scheduler.start(sla_evaluation_job)
    except Exception as e:
        logger.warning(f"SLA scheduler not started: {e}")
        sla_scheduler = None

    # Initialize LLM client
    logger.info("Initializing LLM client")
    try:
        llm_client = ZAIILLMClient(settings.zai_api_key)
    except Exception as e:
        logger.warning(f"LLM client initialization failed: {e}")
        llm_client = None

    # Initialize Grafana OTLP metrics exporter
    logger.info("Initializing Grafana OTLP exporter")
    try:
        if settings.grafana_host and settings.grafana_api_key and settings.grafana_instance_id:
            grafana_exporter = init_grafana_exporter(
                host=settings.grafana_host,
                api_key=settings.grafana_api_key,
                instance_id=settings.grafana_instance_id
            )
            logger.info("Grafana OTLP exporter initialized successfully")
        else:
            logger.info("Grafana OTLP exporter not configured - metrics will not be exported")
    except Exception as e:
        logger.warning(f"Grafana exporter initialization failed: {e}")

    # Initialize classification service
    from triage.application import ClassificationService
    from triage.infrastructure import LLMClientAdapter
    if llm_client or settings.zai_api_key:
        llm_adapter = LLMClientAdapter(settings.zai_api_key)
        classification_service = ClassificationService(llm_adapter)
    else:
        classification_service = None
        logger.warning("Classification service not available - no LLM client")

    # Initialize vector store for triage (Milvus) - optional
    logger.info("Initializing Milvus vector store")
    try:
        vector_store = MilvusVectorStore()
        await vector_store.initialize()
        app.state.vector_store = vector_store
    except Exception as e:
        logger.warning(f"Vector store not available: {e}")
        app.state.vector_store = None

    # Store services in app state for dependency injection
    app.state.llm_client = llm_client
    app.state.classification_service = classification_service

    logger.info("Ticket Service started successfully")

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info("Shutting down Ticket Service")

    # Stop SLA scheduler
    if sla_scheduler:
        await sla_scheduler.stop()

    # Stop config watcher
    if sla_config_manager:
        sla_config_manager.stop_watching()

    # Close Slack client
    await slack_client.close()

    # Close database
    await close_database()

    logger.info("Ticket Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AIGenius Ticketing API",
    description="""
    ## AI-Powered Customer Support Ticketing System

    An intelligent automation platform for SLA monitoring, AI classification, and RAG-based response generation.

    ---

    ### 📊 SLA Monitoring Module

    **Endpoints:**
    - `POST /sla/tickets` - Ingest tickets for SLA tracking
    - `GET /sla/dashboard` - View all tickets with SLA status
    - `GET /sla/tickets/{id}` - Get detailed SLA information for a ticket

    **Features:**
    - Real-time SLA tracking (Response & Resolution)
    - Automated breach detection and alerting
    - Priority-based escalation (critical, high, medium, low)
    - Customer tier support (enterprise, business, standard, free)
    - Background evaluation job (every 60 seconds)

    ---

    ### 🤖 Triage Module

    **Endpoints:**
    - `POST /triage/classify` - Classify ticket by product and urgency (LLM)
    - `POST /triage/respond` - Generate AI-powered responses (RAG)
    - `POST /triage/ingest` - Index documentation for RAG
    - `GET /triage/stats` - View classification statistics

    **Features:**
    - GLM 4.7 LLM for intelligent classification
    - Products: CASB, SWG, ZTNA, DLP, SSPM, CFW, GENERAL
    - Milvus vector store for semantic search
    - Retrieval Augmented Generation (RAG) for responses

    ---

    ### 🔧 Configuration

    **SLA Time Limits (Minutes):**

    | Priority | Enterprise | Business | Standard | Free |
    |----------|-----------|----------|----------|------|
    | Critical | 15 / 120  | 30 / 240 | 60 / 480 | 120 / 960 |
    | High     | 30 / 240  | 60 / 480 | 120 / 960 | 240 / 1920 |
    | Medium   | 60 / 480  | 120 / 960 | 240 / 1920 | 480 / 3840 |
    | Low      | 120 / 960 | 240 / 1920 | 480 / 3840 | 960 / 7680 |

    *Format: Response SLA / Resolution SLA*

    ---

    ### 🏗️ Architecture

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                     Interfaces Layer                        │
    │              (FastAPI Controllers / Routes)                 │
    └──────────────────────────┬──────────────────────────────────┘
                               │
    ┌──────────────────────────┴──────────────────────────────────┐
    │                   Application Layer                          │
    │              (Services, DTOs, Use Cases)                    │
    └──────────────────────────┬──────────────────────────────────┘
                               │
    ┌──────────────────────────┴──────────────────────────────────┐
    │                     Domain Layer                             │
    │           (Entities, Value Objects, Business Logic)         │
    └──────────────────────────┬──────────────────────────────────┘
                               │
    ┌──────────────────────────┴──────────────────────────────────┐
    │                  Infrastructure Layer                        │
    │    (Database, LLM, Vector Store, External APIs)            │
    └─────────────────────────────────────────────────────────────┘
    ```

    ---
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Custom Middleware (from shared) ===
from shared.api.middleware import (
    CorrelationIDMiddleware,
    MetricsMiddleware,
    LoggingMiddleware,
    global_exception_handler
)

app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_exception_handler(Exception, global_exception_handler)

# === Include Module Routers ===
app.include_router(sla_router)
app.include_router(triage_router)

# === Health Check Endpoint ===

@app.get("/health", tags=["Health"], responses={
    200: {
        "description": "Service is healthy",
        "content": {
            "application/json": {
                "example": {
                    "status": "healthy",
                    "version": "1.0.0",
                    "environment": "development",
                    "checks": {
                        "database": "connected",
                        "sla_config": "loaded",
                        "sla_scheduler": "running",
                        "llm_client": "available",
                        "vector_store": "available (0 documents)"
                    }
                }
            }
        }
    }
})
async def health_check(request: Request):
    """
    Health check endpoint for load balancers and orchestrators.

    Returns service health status including:
    - Database connectivity
    - SLA configuration status
    - Scheduler state
    - LLM client availability
    - Vector store status
    """
    checks = {
        "database": "connected",
        "sla_config": "loaded",
        "sla_scheduler": "running" if sla_scheduler and sla_scheduler.is_running else "stopped",
        "llm_client": "available" if llm_client else "not_configured",
        "vector_store": "initializing"
    }

    # Check vector store document count
    if hasattr(request.app.state, "vector_store"):
        try:
            count = await request.app.state.vector_store.get_document_count()
            checks["vector_store"] = f"available ({count} documents)"
        except Exception as e:
            checks["vector_store"] = f"error: {str(e)}"

    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
        "checks": checks
    }


@app.get("/", tags=["Root"], responses={
    200: {
        "description": "API information",
        "content": {
            "application/json": {
                "example": {
                    "service": "Ticket Service",
                    "version": "1.0.0",
                    "architecture": "Clean Architecture / Modular Monolith",
                    "docs": "/docs",
                    "health": "/health"
                }
            }
        }
    }
})
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Ticket Service",
        "version": settings.app_version,
        "architecture": "Clean Architecture / Modular Monolith",
        "docs": "/docs",
        "health": "/health",
        "modules": {
            "sla": {
                "prefix": "/sla",
                "endpoints": [
                    "POST /sla/tickets - Ingest ticket batch",
                    "GET /sla/tickets/{id} - Get ticket SLA status",
                    "GET /sla/dashboard - Get dashboard"
                ]
            },
            "triage": {
                "prefix": "/triage",
                "endpoints": [
                    "POST /triage/classify - Classify ticket",
                    "POST /triage/respond - Generate RAG response",
                    "POST /triage/ingest - Ingest documents",
                    "GET /triage/stats - Get triage statistics"
                ]
            }
        }
    }


# === Grafana Metrics Test Endpoint ===

@app.post("/test/metrics", tags=["Testing"])
async def test_grafana_metrics(request: Request):
    """
    Test endpoint to send metrics to Grafana.

    Useful for verifying Grafana OTLP integration is working.
    """
    from shared.infrastructure.grafana import get_grafana_exporter

    exporter = get_grafana_exporter()

    if not exporter or not exporter.is_enabled():
        return {
            "status": "error",
            "message": "Grafana exporter not configured. Set GRAFANA_HOST, GRAFANA_API_KEY, and GRAFANA_INSTANCE_ID."
        }

    # Send test metrics
    success = await exporter.export_llm_metrics(
        model="glm-4.7",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=1500,
        operation="test",
        attributes={"test": "manual_trigger"}
    )

    if success:
        return {
            "status": "success",
            "message": "Test metrics sent to Grafana successfully",
            "grafana_host": settings.grafana_host,
            "instance_id": settings.grafana_instance_id
        }
    else:
        return {
            "status": "error",
            "message": "Failed to send metrics to Grafana. Check logs for details.",
            "grafana_host": settings.grafana_host,
            "instance_id": settings.grafana_instance_id
        }


# === Development Entry Point ===

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level="info"
    )
