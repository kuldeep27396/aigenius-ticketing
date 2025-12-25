# Technical Interview Preparation Guide

**AIGenius Ticketing System - Code Review & Design Decisions**

*Prepared for technical interview preparation. This document simulates a senior engineer's code review with tough questions and detailed answers.*

---

## Table of Contents

1. [High-Level Design (HLD) Questions](#hld-questions)
2. [Low-Level Design (LLD) Questions](#lld-questions)
3. [FastAPI Fundamentals](#fastapi-fundamentals)
4. [Design Decisions & Trade-offs](#design-decisions)
5. [Code Quality Review](#code-review)
6. [What You Did Well](#strengths)
7. [Areas for Improvement](#improvements)

---

# High-Level Design (HLD) Questions

## Q1: Why did you choose a **Modular Monolith** over microservices?

**Answer:**

**Decision Factors:**
- **Team Size**: Solo project initially - microservices add operational overhead
- **Domain Complexity**: Two bounded contexts (SLA, Triage) - simple enough for one service
- **Deployment**: Single Render service vs managing multiple containers
- **Data Consistency**: Shared PostgreSQL database avoids distributed transactions
- **Development Speed**: Faster to iterate with monorepo

**Architecture:**
```
src/
├── sla/           # Bounded Context 1
├── triage/        # Bounded Context 2
├── shared/        # Cross-cutting concerns
└── config/        # Configuration
```

**Trade-offs:**
| Aspect | Monolith | Microservices |
|--------|----------|---------------|
| Development | ✅ Faster | ❌ Slower |
| Deployment | ✅ Simple | ❌ Complex |
| Scaling | ❌ All-or-nothing | ✅ Per-service |
| Fault Isolation | ❌ Shared runtime | ✅ Isolated |
| Data Consistency | ✅ ACID transactions | ❌ Distributed complexity |

**When to split:**
- If SLA evaluation becomes CPU-intensive → separate service
- If RAG retrieval needs different scale → independent deployment
- If team grows → bounded contexts become microservices

---

## Q2: How do the two modules (SLA and Triage) communicate?

**Answer:**

Currently they're **decoupled** - no direct inter-module communication:

```python
# SLA Module - src/sla/
POST /sla/tickets          # Ingest tickets
GET  /sla/tickets/{id}     # Query SLA status

# Triage Module - src/triage/
POST /triage/classify      # Classify ticket
POST /triage/respond       # Generate response
```

**Why this design:**
1. **Separation of Concerns**: SLA tracks time, Triage analyzes content
2. **Independent Deployment**: Can scale each module independently
3. **Flexibility**: Could integrate with existing ticketing systems

**Optional Integration (not implemented):**
```python
# Future: SLA could consume Triage classifications
# src/sla/application/services.py

async def create_ticket_with_classification(
    external_id: str,
    subject: str,
    content: str
):
    # Create ticket
    ticket = await ticket_repo.create(...)

    # Call Triage API (or import service directly)
    classification = await triage_service.classify(
        ClassificationRequest(
            ticket_id=str(ticket.id),
            subject=subject,
            content=content
        )
    )

    # Could use classification to set initial priority
    ticket.priority = map_classification_to_priority(classification)
    await ticket_repo.update(ticket)
```

**Reference**: [`src/sla/application/services.py:100`](../src/sla/application/services.py#L100)

---

## Q3: Why did you use **SQLAlchemy 2.0** with async instead of synchronous?

**Answer:**

**Key Difference - Blocking vs Non-blocking:**

```python
# ❌ Synchronous (blocks entire request thread)
from sqlalchemy import select

def get_ticket(ticket_id: str):
    # While DB query runs, entire thread is blocked
    stmt = select(TicketModel).where(TicketModel.id == ticket_id)
    result = await session.execute(stmt)  # Blocks here!
    return result.scalar_one()

# ✅ Asynchronous (thread released during I/O)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

async def get_ticket(ticket_id: str):
    # While DB query runs, thread handles other requests
    stmt = select(TicketModel).where(TicketModel.id == ticket_id)
    result = await session.execute(stmt)  # Yields control
    return result.scalar_one()
```

**Why Async Matters in FastAPI:**

```python
# Without async: 100 concurrent requests = 100 threads
# With async: 100 concurrent requests = 1 thread handling all

@app.get("/sla/dashboard")
async def get_dashboard():
    # This async function yields control during:
    # 1. Database queries
    # 2. LLM API calls
    # 3. HTTP requests (Slack webhooks)

    tickets = await ticket_repo.list({})  # Yields to event loop
    results = []

    for ticket in tickets:
        metrics = await sla_service.calculate(ticket.id)  # Yields
        results.append(metrics)

    return results
```

**Throughput Comparison:**
- **Synchronous**: ~10 req/sec (100 threads / 10ms per query)
- **Asynchronous**: ~1000 req/sec (single thread / 1ms overhead between queries)

**Reference**: [`src/infrastructure/database/__init__.py`](../src/infrastructure/database/__init__.py)

```python
# Using asyncpg driver (non-blocking PostgreSQL)
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug
)
```

---

# Low-Level Design (LLD) Questions

## Q4: How does idempotent ticket ingestion work?

**Answer:**

**The Problem:** Clients may retry requests or send duplicate events.

**Solution:** Identify tickets by `(external_id, updated_at)` tuple.

```python
# src/sla/interfaces/controllers.py:176
async def ingest_tickets(request: TicketIngestRequest):
    created = updated = failed = 0
    errors = []

    ticket_repo = SQLAlchemyTicketRepository(session)

    for ticket_dto in request.tickets:
        try:
            # 1. Check if ticket exists
            existing = await ticket_repo.get_by_external_id(ticket_dto.id)

            if existing:
                # 2. Update only if newer
                if ticket_dto.updated_at > existing.updated_at:
                    await ticket_repo.update(entity_dto)
                    updated += 1
                else:
                    # 3. Skip if not newer (idempotent)
                    pass
            else:
                # 4. Create new ticket
                await ticket_repo.create(entity_dto)
                created += 1

        except Exception as e:
            failed += 1
            errors.append(f"{ticket_dto.id}: {str(e)}")

    await session.commit()

    return IngestResponse(
        created=created,
        updated=updated,
        failed=failed,
        errors=errors
    )
```

**Database Schema Support:**
```sql
-- src/sla/infrastructure/models.py:34
external_id VARCHAR(255) UNIQUE NOT NULL  -- Business key
updated_at TIMESTAMPTZ NOT NULL           -- Version comparison
```

**Idempotency Scenarios:**
| Scenario | Request 1 | Request 2 (same) | Result |
|----------|-----------|------------------|--------|
| New ticket | `created=1` | `updated=1` | ✅ One ticket |
| Old timestamp | `created=1` | Skipped | ✅ No change |
| New timestamp | `created=1` | `updated=1` | ✅ Updated |

---

## Q5: How does the SLA evaluation loop work without a cron job?

**Answer:**

Used **APScheduler** - in-process scheduler with background threads.

```python
# src/main.py:109 (lifespan context manager)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SLA scheduler")
    sla_scheduler = SLAScheduler(settings.sla_evaluation_interval)
    await sla_scheduler.start(sla_evaluation_job)

    yield

    # Shutdown
    logger.info("Stopping SLA scheduler")
    await sla_scheduler.stop()
```

**SLA Scheduler Implementation:**
```python
# src/sla/infrastructure/external.py:355
class SLAScheduler:
    """Wrapper for APScheduler for background SLA evaluation."""

    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._running = False

    async def start(self, job_func) -> None:
        """Start the scheduler with the given job function."""
        self._scheduler = AsyncIOScheduler()

        self._scheduler.add_job(
            job_func,
            "interval",  # Run every X seconds
            seconds=self.interval_seconds,
            id="sla_evaluation",
            name="SLA Evaluation Job",
            misfire_grace_time=60,  # Catch up if missed
            max_instances=1,        # Prevent overlap
            replace_existing=True
        )

        self._scheduler.start()
        self._running = True
```

**Evaluation Job:**
```python
# src/sla/application/services.py
async def sla_evaluation_job(
    session: AsyncSession = Depends(get_session),
    alert_repo: ISLAAlertRepository = Depends(get_alert_repository),
    ...
):
    """Run every 60 seconds to check SLA status."""

    # 1. Get all open tickets
    open_tickets = await ticket_repo.list(
        {"status": "open"},
        limit=10000  # No limit for evaluation
    )

    # 2. Calculate SLA for each
    for ticket in open_tickets:
        metrics = await sla_calculator.calculate(ticket, config)

        # 3. Check thresholds
        if metrics.response_percentage_remaining <= 0.15:
            # At risk or breached
            await handle_alert(ticket, metrics, alert_repo)

    # 4. Send notifications
    pending_alerts = await alert_repo.get_pending_alerts()
    for alert in pending_alerts:
        await slack_client.send_alert(alert)
        await alert_repo.mark_sent(alert.id, datetime.now())
```

**Why APScheduler over Cron?**
| Feature | APScheduler | Cron |
|---------|-------------|------|
| Dynamic scheduling | ✅ Can change interval | ❌ Requires crontab edit |
| In-process | ✅ Same Python process | ❌ Separate process |
| Error handling | ✅ Python try/except | ❌ Log parsing |
| Testing | ✅ Unit testable | ❌ System test only |
| Deployment | ✅ No extra config | ❌ Kubernetes CronJob |

---

## Q6: How do you handle LLM failures? What's the fallback strategy?

**Answer:**

**Multi-Provider LLM Client (Priority Order):**

```python
# src/main.py:48
logger.info("Initializing LLM client")
try:
    from src.infrastructure.llm import (
        GroqLLMClient, OpenAILLMClient, ZAIILLMClient
    )

    # Priority: Groq (free, fast) → OpenAI → Z.AI
    if settings.groq_api_key:
        llm_client = GroqLLMClient(settings.groq_api_key)
        logger.info(f"Using Groq: {settings.llm_model}")

    elif settings.openai_api_key:
        llm_client = OpenAILLMClient(settings.openai_api_key)
        logger.info(f"Using OpenAI: {settings.llm_model}")

    elif settings.zai_api_key:
        llm_client = ZAIILLMClient(settings.zai_api_key)
        logger.info(f"Using Z.AI: {settings.llm_model}")

    else:
        llm_client = None
        logger.warning("No LLM API key configured")

except Exception as e:
    logger.warning(f"LLM client initialization failed: {e}")
    llm_client = None
```

**Triage Adapter Fallback:**
```python
# src/triage/infrastructure/external.py:19
class LLMClientAdapter(ILLMClient):
    def __init__(self, api_key: str | None = None):
        # Try Groq first (free, fast), then OpenAI, then Z.AI
        if settings.groq_api_key or api_key:
            self._client = GroqLLMClient(api_key or settings.groq_api_key)

        elif settings.openai_api_key:
            self._client = OpenAILLMClient(settings.openai_api_key)

        elif settings.zai_api_key:
            self._client = ZAIILLMClient(api_key or settings.zai_api_key)

        else:
            raise LLMException(
                "No LLM API key configured. "
                "Set GROQ_API_KEY, OPENAI_API_KEY, or ZAI_API_KEY."
            )

    async def chat_completion(self, messages, **kwargs):
        try:
            return await self._client.chat_completion(messages, **kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Could retry with next provider here
            raise
```

**Error Handling in Service Layer:**
```python
# src/triage/application/services.py:49
async def classify_ticket(self, request: ClassificationRequest) -> ClassificationResult:
    try:
        # Build prompt
        prompt = self._build_classification_prompt(request)

        # Call LLM
        response = await self._llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Parse response
        result = self._parse_classification(response.content)

        # Track metrics
        return ClassificationResult(
            product=result["product"],
            urgency=result["urgency"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            model_used=settings.llm_model,
            latency_ms=int((time.time() - start) * 1000)
        )

    except LLMException as e:
        # LLM unavailable - use mock/fallback
        logger.warning(f"LLM unavailable, using defaults: {e}")
        return ClassificationResult(
            product=ProductCategory.GENERAL,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.0,
            reasoning="LLM unavailable - default classification"
        )
```

---

# FastAPI Fundamentals

## Q7: What is `@router.post` and `Depends`? How do they work?

**Answer:**

### `@router.post` - Route Decorator

```python
@router.post("/tickets")
async def ingest_tickets(request: TicketIngestRequest):
    return {"status": "ok"}
```

**What's happening:**
1. `@router.post` registers the function with FastAPI's routing system
2. HTTP POST to `/sla/tickets` triggers this function
3. FastAPI validates request body against `TicketIngestRequest` Pydantic model
4. Returns JSON response automatically

**Under the hood:**
```python
# FastAPI does this automatically:
class Request:
    method: str = "POST"
    path: str = "/sla/tickets"
    headers: dict
    body: bytes

# Deserializes JSON:
request_body = b'{"tickets": [{"id": "TICKET-100", ...}]}'
data = json.loads(request_body)

# Validates with Pydantic:
validated = TicketIngestRequest(**data)  # Raises ValidationError if invalid

# Calls your function:
result = await ingest_tickets(validated)

# Serializes response:
response = JSONResponse(content=result)
```

### `Depends` - Dependency Injection

```python
async def get_ticket(
    ticket_id: str,
    session: AsyncSession = Depends(get_session)  # ← Dependency
):
    ticket = await session.get(Ticket, ticket_id)
    return ticket
```

**What's happening:**
1. `Depends(get_session)` tells FastAPI to call `get_session()` first
2. Result is passed to your function as `session` parameter
3. Same instance is reused across the request (singleton per request)

**Full example with multiple dependencies:**

```python
# src/sla/interfaces/controllers.py:104
async def get_sla_service(
    session: AsyncSession = Depends(get_session)
) -> SLAService:
    """Dependency provider - creates SLA service with database session."""
    ticket_repo = SQLAlchemyTicketRepository(session)
    config_provider = YAMLConfigProvider("sla_config.yaml")
    return SLAService(ticket_repo, config_provider)

@router.get("/tickets/{ticket_id}")
async def get_ticket_sla(
    ticket_id: str,
    session: AsyncSession = Depends(get_session),
    sla_service: SLAService = Depends(get_sla_service)  # ← Uses dependency
):
    # Both session and sla_service are injected
    ticket_data = await ticket_repo.get_by_external_id(ticket_id)
    metrics = await sla_service.calculate_sla_metrics(str(ticket_data.id))
    return metrics
```

**Dependency Lifecycle:**

```python
# FastAPI creates dependency cache per request
# Request 1:
get_session() called once → reused for all dependencies in request

# Request 2:
get_session() called again → new session, new transaction

# Equivalent to:
with get_db_session() as session:
    service1 = SLAService(session)
    service2 = SLAService(session)  # Same session
    # Handle request
# Session closed automatically
```

---

## Q8: What's the difference between `response_model` and the return type annotation?

**Answer:**

```python
@router.post(
    "/tickets",
    response_model=IngestResponse  # ← For Swagger docs & validation
)
async def ingest_tickets(
    request: TicketIngestRequest
) -> IngestResponse:  # ← For Python type checking
    return IngestResponse(created=1, ...)
```

**`response_model` (FastAPI-specific):**
- Controls what's shown in **OpenAPI/Swagger** documentation
- **Filters output** - only includes fields in the model
- Validates response before sending to client
- Enables **response serialization**

**Return type annotation (Python):**
- Used by **IDE/Pyright** for type checking
- No runtime enforcement
- Doesn't affect API behavior

**Example - Filtering:**

```python
from pydantic import BaseModel

class FullTicketData(BaseModel):
    id: str
    external_id: str
    priority: str
    internal_notes: str  # Sensitive data
    created_at: datetime

class PublicTicket(BaseModel):
    external_id: str
    priority: str
    created_at: datetime
    # Note: internal_notes NOT included

@router.get("/tickets/{id}", response_model=PublicTicket)
async def get_ticket(id: str) -> FullTicketData:
    # Full data available in function
    ticket = await db.get_ticket(id)
    return ticket  # But only PublicTicket fields sent to client!
```

**Why this matters:**
```python
# Client receives:
{
  "external_id": "TICKET-100",
  "priority": "high",
  "created_at": "2025-12-25T10:00:00Z"
}

# Even though you returned:
{
  "id": "uuid-here",
  "external_id": "TICKET-100",
  "priority": "high",
  "internal_notes": "Customer is VIP",
  "created_at": "2025-12-25T10:00:00Z"
}
```

---

## Q9: How does `asyncio` work in FastAPI? When should I use `async def` vs `def`?

**Answer:**

**Rule of Thumb:**
- Use `async def` for **I/O operations** (database, HTTP, external APIs)
- Use regular `def` for **CPU-intensive operations** (data processing, encryption)

**FastAPI Concurrency Model:**

```python
# ✅ ASYNC: Good for I/O
@router.get("/tickets/{id}")
async def get_ticket(id: str):
    # While waiting for DB, thread handles other requests
    ticket = await db.get_ticket(id)
    return ticket

# ❌ SYNC: Blocks thread during DB call
@router.get("/tickets/{id}")
def get_ticket(id: str):
    # Thread blocks here, can't handle other requests
    ticket = db.get_ticket(id)  # Synchronous call
    return ticket
```

**When to use `def` instead of `async def`:**

```python
# ✅ DEF: OK for CPU-bound (non-blocking operations)
@router.get("/stats")
def get_stats():
    # Fast computation, no I/O
    # Def is fine because it's quick
    return {"total": 1000, "active": 50}

# ✅ DEF: Also OK for non-async operations
@router.get("/health")
def health_check():
    return {"status": "healthy"}
```

**Why `async def` matters in your code:**

```python
# src/sla/interfaces/controllers.py:176
async def ingest_tickets(request: TicketIngestRequest):
    # Multiple async operations can run concurrently

    # Without async: 10 tickets × 50ms DB latency = 500ms
    # With async: All 10 insertions can happen in ~50ms

    for ticket_dto in request.tickets:
        await ticket_repo.create(entity_dto)  # Non-blocking
        # CPU is free to handle other API requests while DB inserts
```

**Performance Comparison:**

```python
# Synchronous - Single-threaded blocking
def ingest_tickets_sync(request):
    for ticket in request.tickets:
        ticket_repo.create(ticket)  # Blocks for 50ms each
    # 10 tickets = 500ms total
    # During these 500ms, server can't handle other requests!

# Asynchronous - Event loop
async def ingest_tickets_async(request):
    for ticket in request.tickets:
        await ticket_repo.create(ticket)  # Yields control
        # Other requests can be processed during DB I/O!
    # 10 tickets might complete in 50-100ms total
```

---

## Q10: What's `Annotated` used for in your config?

**Answer:**

```python
# src/config/__init__.py:15
from pydantic import Field, field_validator
from typing import Annotated
```

**`Annotated` adds metadata to type hints:**

```python
# BEFORE (Python 3.9 style):
def get_tickets(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    pass

# AFTER (Annotated - Python 3.10+):
from typing import Annotated

def get_tickets(
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0
):
    pass
```

**Benefits:**
1. **Type checker friendly** - IDE understands both `int` and constraints
2. **Reusable annotations** - Define once, use everywhere
3. **Better IDE autocomplete** - Shows validation rules
4. **Pydantic V3 style** - Modern FastAPI pattern

**In your code:**

```python
# src/config/__init__.py:42
sla_config_path: Annotated[Path, Field(
    default=Path("sla_config.yaml"),
    description="Path to SLA configuration YAML file"
)]
```

**Equivalent without Annotated:**
```python
sla_config_path: Path = Field(
    default=Path("sla_config.yaml"),
    description="Path to SLA configuration YAML file"
)
```

**Why Pydantic `Field`?**
- Generates **OpenAPI/Swagger documentation**
- Adds **validation rules**
- Provides **default values**
- **Description** appears in UI

---

# Design Decisions & Trade-offs

## Q11: Why use **Pydantic** for configuration instead of environment variables directly?

**Answer:**

**Your approach** - [`src/config/__init__.py:15`](../src/config/__init__.py#L15):

```python
class Settings(BaseSettings):
    app_name: str = Field(default="ticket-service")
    database_url: str = Field(...)
    groq_api_key: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # DATABASE_URL == database_url
        extra="ignore"  # Ignore extra env vars
    )
```

**Benefits:**

1. **Type Safety:**
```python
settings = get_settings()

# IDE knows types
database_url: str = settings.database_url  # ✅ Type checked
port: int = settings.port  # ✅ Type checked

# No runtime surprises
port = settings.port * 2  # ✅ int
port = settings.port + "invalid"  # ❌ Type error
```

2. **Validation:**
```python
class Settings(BaseSettings):
    port: int = Field(default=8000, ge=1, le=65535)
    #                          ^^^^^^^^^^^^
    #                          Validation rules!
```

3. **Defaults & Documentation:**
```python
class Settings(BaseSettings):
    sla_config_path: Path = Field(
        default=Path("sla_config.yaml"),
        description="Path to SLA configuration YAML file"
    )
```

4. **Environment Variable Loading:**
```bash
# .env file
DATABASE_URL=postgresql+asyncpg://...
GROQ_API_KEY=gsk_xxx

# Loaded automatically!
settings = Settings()  # Reads .env automatically
```

**Alternative (not used) - Direct env vars:**
```python
import os

# ❌ No validation, no defaults, no type safety
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL required")
# What type is it? String? What format?
```

---

## Q12: Why did you use **UUID** instead of auto-increment integers for primary keys?

**Answer:**

**Your choice:**
```python
# src/sla/infrastructure/models.py:31
id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
external_id: Mapped[str] = mapped_column(String(255), unique=True)
```

**Pros of UUID:**
1. **Distributed-friendly** - Multiple services can generate IDs without coordination
2. **Security** - Don't expose record count (GET /tickets/100 = 101st ticket)
3. **No lock contention** - `uuid4()` is random, no sequence needed
4. **Globally unique** - Can merge data from different systems

**Pros of Auto-Increment:**
1. **Smaller index** - 8 bytes vs 16 bytes
2. **Sortable** - Natural ordering by creation time
3. **Human-readable** - Ticket 100 vs Ticket 550e8400-e29b-41d4-a716-446655440000

**Hybrid approach (what you did):**
```python
id: UUID           # Internal primary key
external_id: str  # Business identifier (TICKET-100)
```

**Why this works:**
- External users see `TICKET-100` (friendly)
- Internal joins use UUID (fast, unique)
- Can correlate with external systems via `external_id`

---

## Q13: How do you ensure database consistency during SLA evaluation?

**Answer:**

**Potential Race Condition:**
```python
# Time: 10:00:00
scheduler_tick_1() sees ticket has 5 minutes remaining

# Time: 10:00:01 (1ms later!)
scheduler_tick_2() also sees ticket has 5 minutes remaining

# Both create alerts → Duplicate alerts!
```

**Solutions:**

**1. APScheduler `max_instances=1`:**
```python
# src/sla/infrastructure/external.py:377
self._scheduler.add_job(
    job_func,
    "interval",
    seconds=60,
    max_instances=1,  # ← Prevents overlapping runs
    replace_existing=True
)
```

**2. Idempotent Alert Creation:**
```python
# src/sla/application/services.py
async def create_alert_if_needed(
    ticket_id: str,
    sla_type: SLAType,
    alert_type: AlertType
):
    # Check if alert already exists for this ticket/SLA
    existing = await alert_repo.get_pending_alerts(ticket_id)

    for alert in existing:
        if alert.sla_type == sla_type:
            # Alert already exists, skip
            return

    # Create new alert
    await alert_repo.create(alert)
```

**3. Database Constraints:**
```sql
-- Could add unique constraint to prevent duplicates
CREATE UNIQUE INDEX idx_sla_alerts_unique
ON sla_alerts(ticket_id, sla_type, state)
WHERE state IN ('on_track', 'at_risk');
```

---

## Q14: How does hot-reload of SLA configuration work?

**Answer:**

**Using Watchdog file observer:**

```python
# src/sla/infrastructure/external.py:30
class ConfigFileHandler(FileSystemEventHandler):
    """Watchdog event handler for SLA config file changes."""

    def __init__(self, config_manager: "SLAConfigManager", config_path: Path):
        self.config_manager = config_manager
        self.config_path = config_path

    def on_modified(self, event):
        """Handle file modification event."""
        if Path(event.src_path).resolve() == self.config_path.resolve():
            logger.info(f"Config file changed: {event.src_path}")
            self.config_manager.reload()  # Reload YAML
```

**Scheduler Integration:**
```python
# src/sla/infrastructure/external.py:93
def start_watching(self) -> None:
    """Start watching configuration file for changes."""

    if not self._path.exists():
        # Production: File might not exist
        return

    try:
        self._observer = Observer()
        handler = ConfigFileHandler(self, self._path)
        self._observer.schedule(
            handler,
            str(self._path.parent),  # Watch directory
            recursive=False
        )
        self._observer.start()
        logger.info(f"Started watching config file: {self._path}")

    except (OSError, FileNotFoundError) as e:
        # Docker: Inotify doesn't work in containers
        logger.warning(f"File watching not available: {e}")
```

**Reload Logic:**
```python
# src/sla/infrastructure/external.py:78
def reload(self) -> bool:
    """Reload configuration from file."""
    try:
        new_config = self._load_from_file(self._path)
        with self._lock:  # Thread-safe
            self._config = new_config
        logger.info("SLA configuration reloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to reload SLA config: {e}")
        return False
```

**Why `threading.Lock`?**
- Watchdog runs in separate thread
- Main application reads config from another thread
- Lock prevents read-during-write corruption

---

# Code Quality Review

## What You Did Well ✅

### 1. Clean Architecture
```
domain/      ← No dependencies on infrastructure
application/ ← Depends on domain interfaces
infrastructure/ ← Implements interfaces
interfaces/   ← FastAPI controllers (thin)
```

**Why this matters:**
- Testable: Can mock infrastructure in unit tests
- Swappable: Change database without touching business logic
- Maintainable: Clear separation of concerns

### 2. Dependency Inversion Principle
```python
# src/sla/application/services.py:26
class ITicketRepository(ABC):
    @abstractmethod
    async def get_by_external_id(self, external_id: str) -> Optional[Any]:
        """Interface - contract, not implementation"""

class SQLAlchemyTicketRepository(ITicketRepository):
    """Concrete implementation - can be swapped"""
    async def get_by_external_id(self, external_id: str):
        # SQLAlchemy code here
```

### 3. Type Safety with Pydantic
```python
# Request validation
class TicketCreateDTO(BaseModel):
    id: str = Field(..., min_length=1, max_length=255)
    priority: Priority  # Enum validation
    customer_tier: CustomerTier  # Enum validation
```

### 4. Comprehensive Logging
```python
# Structured JSON logging
logger.info(
    "Ticket ingestion complete",
    extra={
        "tickets_created": created,
        "tickets_updated": updated,
        "tickets_failed": failed,
        "processing_time_ms": total_time
    }
)
```

### 5. Idempotent Operations
```python
# Check if exists, compare timestamps
if existing:
    if ticket_dto.updated_at > existing.updated_at:
        await ticket_repo.update(entity_dto)
```

---

# Areas for Improvement 🔧

## 1. Add Database Migrations

**Current:** SQLAlchemy `create_all()`
```python
# src/main.py
async def create_database_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

**Better:** Alembic migrations
```bash
# Create migration
alembic revision --autogenerate -m "Add tickets table"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

**Why:**
- Version controlled schema changes
- Rollback support
- Zero-downtime deployments

## 2. Add Request/Response Validation Middleware

**Current:** Basic validation in endpoints

**Add:** Global middleware
```python
@app.middleware("http")
async def validate_content_type(request: Request, call_next):
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type"
            )
    return await call_next(request)
```

## 3. Add Circuit Breaker Pattern

**Current:** Basic retry logic in Slack client

**Improve:** Full circuit breaker
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def call_llm(messages):
    return await llm_client.chat_completion(messages)

# Opens after 5 failures
# Attempts recovery after 60 seconds
```

## 4. Add Distributed Tracing

**Current:** Correlation IDs in logs

**Add:** OpenTelemetry tracing
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("ingest_tickets")
async def ingest_tickets(request):
    with tracer.start_as_span("process_ticket") as span:
        span.set_attribute("ticket_count", len(request.tickets))
        # ... process tickets
```

## 5. Add API Versioning

**Current:** No versioning

**Add:** Versioned routes
```python
app_v1 = APIRouter(prefix="/api/v1/sla")
app_v2 = APIRouter(prefix="/api/v2/sla")

app.include_router(app_v1)
app.include_router(app_v2)
```

---

# Common Interview Questions & Answers

## Q: What happens if two requests try to create the same ticket simultaneously?

**A:**

**Scenario:**
```python
# Request 1: POST /sla/tickets with {"id": "TICKET-100", ...}
# Request 2: POST /sla/tickets with {"id": "TICKET-100", ...}
```

**Database handles it:**
```sql
-- src/sla/infrastructure/models.py:34
external_id VARCHAR(255) UNIQUE NOT NULL
                              ^^^^^^
```

**Without `ON CONFLICT`:**
```
Request 1: INSERT INTO tickets (external_id='TICKET-100') → Success
Request 2: INSERT INTO tickets (external_id='TICKET-100') → ERROR!
```

**Your code handles it gracefully:**
```python
# src/sla/interfaces/controllers.py:208
existing = await ticket_repo.get_by_external_id(ticket_dto.id)

if existing:
    # Update if newer
    if ticket_dto.updated_at > existing.updated_at:
        await ticket_repo.update(entity_dto)
        updated += 1
    else:
        pass  # Skip
else:
    await ticket_repo.create(entity_dto)
    created += 1
```

**Could improve with `ON CONFLICT`:**
```sql
INSERT INTO tickets (external_id, ...)
VALUES ('TICKET-100', ...)
ON CONFLICT (external_id)
DO UPDATE SET
  priority = EXCLUDED.priority,
  updated_at = EXCLUDED.updated_at
WHERE EXCLUDED.updated_at < tickets.updated_at;
```

---

## Q: How do you measure LLM token usage and costs?

**A:**

**Tracking in classification:**
```python
# src/triage/infrastructure/models.py:72
model_used: Mapped[str] = mapped_column(String(100))
prompt_tokens: Mapped[int] = mapped_column(Integer)
completion_tokens: Mapped[int] = mapped_column(Integer)
```

**Recording usage:**
```python
# src/triage/infrastructure/external.py:82
async def chat_completion(self, messages, **kwargs):
    start_time = time.time()

    response = await self._client.chat.completions.create(
        model=self._model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Track token usage
    usage = response.usage
    latency_ms = int((time.time() - start_time) * 1000)

    logger.info(
        "LLM call completed",
        extra={
            "model": self._model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_ms": latency_ms
        }
    )

    return ChatCompletionResult(
        content=response.choices[0].message.content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens
    )
```

---

## Q: Why did you use **Groq** instead of OpenAI?

**A:**

**Cost Comparison:**
| Provider | Model | Input (per 1M) | Output (per 1M) |
|----------|-------|----------------|-----------------|
| **Groq** | Llama 3.3 70B | **$0.00** | **$0.00** |
| OpenAI | GPT-4o | $2.50 | $10.00 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 |

**Decision factors:**
1. **Free tier** - Unlimited requests within rate limits
2. **Speed** - Groq's specialized inference is ultra-fast
3. **Quality** - Llama 3.3 70B performs competitively with GPT-4
4. **No quota issues** - OpenAI has stricter limits on free tier

**Trade-off:** Rate-limited (30 requests/minute)
- **For current scale:** Perfect fit
- **For high scale:** Would need to upgrade to paid tier

---

## Q: How does your RAG pipeline work?

**A:**

**Flow:**
```
User Question → Embedding → Vector Search → Context Building → LLM → Response
```

**Implementation:**
```python
# src/triage/application/services.py:87
async def generate_response(ticket, retrieved_docs):
    # 1. Build context from retrieved docs
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"""
[Source {i}]
Title: {doc.title}
Category: {doc.category}
Content: {doc.text}
URL: {doc.url}
""")

    context = "\n".join(context_parts)

    # 2. Build prompt with context
    prompt = f"""
You are a helpful support assistant. Use the following documentation
to answer the customer's question. Cite your sources.

CONTEXT:
{context}

CUSTOMER TICKET:
Subject: {ticket.subject}
Description: {ticket.content}

QUESTION: What should the engineer do to help this customer?

Provide a helpful response with source citations.
"""

    # 3. Generate response
    response = await llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.content
```

**Vector Search:**
```python
# src/infrastructure/vectorstore/__init__.py
async def search(query_vector: list[float], top_k: int = 5):
    results = self._client.search(
        collection_name=self.collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "title", "category", "url"]
    )
    return results
```

---

# FastAPI Decorators Deep Dive

## `@app.get` / `@router.post` - Route Decorators

```python
@router.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    pass
```

**Breaking it down:**

**1. `@router` - APIRouter grouping:**
```python
# src/sla/interfaces/controllers.py:36
router = APIRouter(prefix="/sla", tags=["SLA Monitoring"])

# All routes in this file start with /sla
# GET /sla/tickets/{id}
# POST /sla/tickets
```

**2. `@router.get` - HTTP method + path:**
```python
@router.get("/tickets/{ticket_id}")
#        ^^^^  ^^^^^^^^^^^^^^
#        |     Path parameter
#        HTTP method
```

**3. Path parameters:**
```python
@router.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    # ticket_id extracted from URL
    # GET /sla/tickets/TICKET-100 → ticket_id = "TICKET-100"
```

**Full decorators available:**
```python
@router.get("/items")       # GET /sla/items
@router.post("/items")      # POST /sla/items
@router.put("/items/{id}")   # PUT /sla/items/123
@router.patch("/items/{id}") # PATCH /sla/items/123
@router.delete("/items/{id}")# DELETE /sla/items/123
@router.options("/items")   # OPTIONS /sla/items
@router.head("/items")      # HEAD /sla/items
```

---

## `Query` - Query Parameters

```python
@router.get("/dashboard")
async def get_dashboard(
    customer_tier: Optional[str] = Query(
        None,
        description="Filter by customer tier"
    )
):
    # URL: /sla/dashboard?customer_tier=enterprise
    # customer_tier = "enterprise"
```

**Breakdown:**
```python
Query(
    None,           # Default value
    description="...",  # Swagger docs
    ge=1,           # Min value (if int)
    le=1000         # Max value (if int)
)
```

**Equivalent without `Query`:**
```python
# Old FastAPI style (< 0.100.0)
from fastapi import Query

customer_tier: Optional[str] = Query(None)
```

---

## `Body` - Request Body

```python
@router.post("/tickets")
async def ingest_tickets(
    request: TicketIngestRequest  # ← From request body
):
```

**Where does it come from?**
```python
# src/sla/application/dto.py
class TicketIngestRequest(BaseModel):
    tickets: List[TicketCreateDTO]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

**Request:**
```json
{
  "tickets": [
    {"id": "TICKET-100", "priority": "critical", ...}
  ]
}
```

---

## `status_code` - Custom Status Codes

```python
@router.post(
    "/tickets",
    status_code=201,  # Created
    response_model=IngestResponse
)
async def ingest_tickets(...):
    return IngestResponse(created=1, ...)
```

**Or dynamic:**
```python
from fastapi import status

@router.post("/tickets")
async def ingest_tickets(...):
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request"
        )

    return Response(
        content={"created": 1},
        status_code=status.HTTP_201_CREATED
    )
```

---

## `BackgroundTasks` - Run Tasks After Response

**Not used in your code**, but useful to know:

```python
from fastapi import BackgroundTasks

@router.post("/tickets")
async def ingest_tickets(
    request: TicketIngestRequest,
    background_tasks: BackgroundTasks
):
    # Save to DB
    created = await ticket_repo.create(request.tickets[0])

    # Send notification after response is sent
    background_tasks.add_task(
        send_slack_notification,
        created.id
    )

    # Returns immediately, notification happens in background
    return {"created": 1}
```

---

## `Middleware` - Custom Request/Response Processing

**Example - Request ID Middleware:**

```python
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Add unique ID to each request
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add header to response
        response.headers["X-Request-ID"] = request_id
        return response

# Add to app
app.add_middleware(RequestIDMiddleware)
```

---

# Reference: Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| [`src/sla/interfaces/controllers.py`](../src/sla/interfaces/controllers.py) | SLA API endpoints | `ingest_tickets`, `get_ticket_sla`, `get_dashboard` |
| [`src/sla/infrastructure/models.py`](../src/sla/infrastructure/models.py) | Database models | `TicketModel`, `AlertModel` |
| [`src/sla/application/services.py`](../src/sla/application/services.py) | Business logic | `SLAService`, `SLAEvaluationService` |
| [`src/triage/interfaces/controllers.py`](../src/triage/interfaces/controllers.py) | Triage API endpoints | `classify_ticket`, `generate_response` |
| [`src/infrastructure/llm/__init__.py`](../src/infrastructure/llm/__init__.py) | LLM clients | `GroqLLMClient`, `OpenAILLMClient` |
| [`src/config/__init__.py`](../src/config/__init__.py) | Configuration | `Settings`, constants |
| [`src/main.py`](../src/main.py) | Application entry | Lifespan, dependency setup |

---

# Summary: Key Talking Points

### Architecture
- **Modular Monolith** - Right choice for team size and complexity
- **Clean Architecture** - Separation of concerns, testable, maintainable
- **Async/await** - Critical for I/O performance in FastAPI

### SLA Monitoring
- **APScheduler** - In-process background evaluation
- **Watchdog** - Hot-reload configuration (optional in production)
- **Idempotent ingestion** - Safe for retries

### AI/Triage
- **Multi-provider LLM** - Groq (free, fast) → OpenAI → Z.AI
- **RAG pipeline** - Milvus vector search → context building → LLM
- **Structured logging** - Token tracking, latency metrics

### Database
- **SQLAlchemy 2.0 async** - Non-blocking database operations
- **UUID primary keys** - Distributed-friendly, secure
- **External ID** - User-friendly business identifiers

### FastAPI Techniques Used
- `@router.post` - Route definition
- `Depends()` - Dependency injection
- `response_model` - Output filtering & validation
- `Query()` - Query parameters
- `AsyncSession` - Database session management

---

**Document Version**: 1.0
**Last Updated**: 2025-12-25
**Author**: Kuldeep Pal
