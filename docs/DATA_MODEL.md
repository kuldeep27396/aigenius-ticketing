# Data Model Documentation

**AIGenius Ticketing System - Database Schema & Entity Relationships**

---

## ER Diagram (Entity-Relationship)

```mermaid
erDiagram
    %% SLA Module Tables
    TICKETS ||--o{ SLA_ALERTS : has
    TICKETS {
        uuid id PK
        string external_id UK "Business ID"
        string priority "critical|high|medium|low"
        string customer_tier "enterprise|business|standard|free"
        string status "open|in_progress|pending_customer|pending_vendor|resolved|closed"
        string subject
        text content
        datetime created_at
        datetime updated_at
        datetime first_response_at
        datetime resolved_at
        datetime closed_at
    }

    SLA_ALERTS {
        uuid id PK
        uuid ticket_id FK
        string ticket_external_id
        string sla_type "response|resolution"
        string alert_type "warning|breach"
        datetime triggered_at
        datetime deadline
        float remaining_seconds
        string state "on_track|at_risk|breached|met"
        int escalation_level
        bool notification_sent
        datetime notification_sent_at
    }

    %% Triage Module Tables
    TRIAGE_TICKETS ||--o{ CLASSIFICATIONS : has
    TRIAGE_TICKETS {
        uuid id PK
        string external_id "Optional reference"
        string subject
        text content
        datetime created_at
        datetime updated_at
    }

    CLASSIFICATIONS {
        uuid id PK
        uuid ticket_id FK
        string product "CASB|SWG|ZTNA|DLP|SSPM|CFW|GENERAL"
        string urgency "critical|high|medium|low"
        float confidence "0.0-1.0"
        text reasoning
        string model_used
        int latency_ms
        int prompt_tokens
        int completion_tokens
        datetime created_at
    }

    %% Relationships
    TICKETS ||--o{ TRIAGE_TICKETS : "can_correlate"
    SLA_ALERTS }o--|| TRIAGE_TICKETS : "references"
```

---

## Database Schema Diagram

```mermaid
graph TB
    subgraph "SLA Module Schema"
        TICKETS[tickets]
        ALERTS[sla_alerts]

        TICKETS -->|"1:N"| ALERTS
    end

    subgraph "Triage Module Schema"
        T_TRIAGE[triage_tickets]
        CLASS[classifications]

        T_TRIAGE -->|"1:N"| CLASS
    end

    subgraph "Cross-Module References"
        TICKETS -.->|"optional"| T_TRIAGE
    end

    style TICKETS fill:#e1f5ff
    style ALERTS fill:#fff4e1
    style T_TRIAGE fill:#f3e5f5
    style CLASS fill:#e8f5e9
```

---

## Table Definitions

### SLA Module

#### `tickets` Table
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT uuid4() | Internal primary key |
| `external_id` | VARCHAR(255) | UNIQUE, NOT NULL, INDEX | External ticket ID (e.g., TICKET-100) |
| `priority` | VARCHAR(50) | NOT NULL, DEFAULT 'medium' | Priority level |
| `customer_tier` | VARCHAR(50) | NOT NULL, DEFAULT 'standard' | Customer tier |
| `status` | VARCHAR(50) | NOT NULL, DEFAULT 'open' | Ticket status |
| `subject` | VARCHAR(500) | NOT NULL | Ticket subject |
| `content` | TEXT | NOT NULL | Ticket description |
| `created_at` | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Last update timestamp |
| `first_response_at` | TIMESTAMPTZ | NULLABLE | First response time |
| `resolved_at` | TIMESTAMPTZ | NULLABLE | Resolution time |
| `closed_at` | TIMESTAMPTZ | NULLABLE | Closure time |

**Indexes:**
- `external_id` (unique)
- Composite for status filtering

**Enums:**
- `priority`: `critical`, `high`, `medium`, `low`
- `customer_tier`: `enterprise`, `business`, `standard`, `free`
- `status`: `open`, `in_progress`, `pending_customer`, `pending_vendor`, `resolved`, `closed`

---

#### `sla_alerts` Table
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT uuid4() | Primary key |
| `ticket_id` | UUID | NOT NULL, INDEX (→ tickets.id) | Foreign key to tickets |
| `ticket_external_id` | VARCHAR(255) | NOT NULL | Denormalized external ID |
| `sla_type` | VARCHAR(50) | NOT NULL | SLA clock type |
| `alert_type` | VARCHAR(50) | NOT NULL | Alert severity |
| `triggered_at` | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Alert trigger time |
| `deadline` | TIMESTAMPTZ | NOT NULL | SLA deadline |
| `remaining_seconds` | FLOAT | NOT NULL | Time remaining (negative if breached) |
| `state` | VARCHAR(50) | NOT NULL | Current SLA state |
| `escalation_level` | INTEGER | NOT NULL, DEFAULT 1 | Escalation level |
| `notification_sent` | BOOLEAN | NOT NULL, DEFAULT FALSE | Slack notification status |
| `notification_sent_at` | TIMESTAMPTZ | NULLABLE | Notification timestamp |

**Enums:**
- `sla_type`: `response`, `resolution`
- `alert_type`: `warning`, `breach`
- `state`: `on_track`, `at_risk`, `breached`, `met`

---

### Triage Module

#### `triage_tickets` Table
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT uuid4() | Primary key |
| `external_id` | VARCHAR(255) | NULLABLE, INDEX | Optional external reference |
| `subject` | VARCHAR(500) | NOT NULL | Ticket subject |
| `content` | TEXT | NOT NULL | Ticket content |
| `created_at` | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | NULLABLE | Last update timestamp |

**Note:** This is a separate table from `tickets` to support RAG operations. Can be correlated via `external_id`.

---

#### `classifications` Table
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT uuid4() | Primary key |
| `ticket_id` | UUID | NOT NULL, INDEX, FK (→ triage_tickets.id ON DELETE CASCADE) | Foreign key |
| `product` | VARCHAR(50) | NOT NULL | Product category |
| `urgency` | VARCHAR(50) | NOT NULL | Urgency level |
| `confidence` | FLOAT | NOT NULL | Classification confidence (0.0-1.0) |
| `reasoning` | TEXT | NOT NULL | AI explanation |
| `model_used` | VARCHAR(100) | NOT NULL | LLM model identifier |
| `latency_ms` | INTEGER | NOT NULL | Processing time in milliseconds |
| `prompt_tokens` | INTEGER | NOT NULL, DEFAULT 0 | Input tokens used |
| `completion_tokens` | INTEGER | NOT NULL, DEFAULT 0 | Output tokens generated |
| `created_at` | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Classification timestamp |

**Enums:**
- `product`: `CASB`, `SWG`, `ZTNA`, `DLP`, `SSPM`, `CFW`, `GENERAL`
- `urgency`: `critical`, `high`, `medium`, `low`

**Relationship:** One ticket can have multiple classification records (history).

---

## Data Flow Diagram

```mermaid
flowchart TD
    %% Ticket Ingestion Flow
    INGEST["POST /sla/tickets"] -->|"TicketCreateDTO"| TICKET_REPO["SQLAlchemyTicketRepository"]
    TICKET_REPO -->|"create() or update()"| DB[(PostgreSQL)]
    TICKET_REPO -->|"return"| INGEST_RESPONSE["IngestResponse: created/updated/failed"]

    %% SLA Evaluation Flow
    SCHEDULER["APScheduler<br/>Every 60s"] -->|"Get all open tickets"| SLA_SERVICE["SLAEvaluationService"]
    SLA_SERVICE -->|"calculate_sla_metrics()"| CALCULATOR["SLACalculator"]
    CALCULATOR -->|"Time remaining ≤ 15%?"| CHECK{Threshold Check}
    CHECK -->|"Yes"| ALERT_SERVICE["Create Alert"]
    CHECK -->|"No"| MONITOR["Continue monitoring"]
    ALERT_SERVICE -->|"Send Slack"| SLACK_CLIENT["SlackClient"]
    ALERT_SERVICE -->|"save"| ALERTS_DB[(sla_alerts table)]
    SLACK_CLIENT -->|"success"| ALERTS_DB

    %% Classification Flow
    CLASSIFY_REQ["POST /triage/classify"] -->|"ClassificationRequest"| CLASSIFIER["ClassificationService"]
    CLASSIFIER -->|"generate prompt"| LLM["Groq Llama 3.3"]
    LLM -->|"chat_completion()"| CLASSIFIER
    CLASSIFIER -->|"save"| CLASS_DB[(classifications table)]
    CLASSIFIER -->|"return"| CLASS_RESPONSE["ClassificationResponse"]

    %% RAG Response Flow
    RESPOND_REQ["POST /triage/respond"] -->|"ResponseRequest"| RAG_SERVICE["RAGService"]
    RAG_SERVICE -->|"embed query"| EMBEDDING["OpenAI Embeddings"]
    EMBEDDING -->|"vector"| VECTOR_STORE[(Milvus/Zilliz)]
    VECTOR_STORE -->|"Top-K documents"| RAG_SERVICE
    RAG_SERVICE -->|"build context + query"| LLM
    LLM -->|"response"| RAG_SERVICE
    RAG_SERVICE -->|"return"| RAG_RESPONSE["ResponseResponse"]

    style DB fill:#87CEEB
    style ALERTS_DB fill:#87CEEB
    style CLASS_DB fill:#87CEEB
    style LLM fill:#90EE90
    style VECTOR_STORE fill:#DDA0DD
```

---

## State Transitions

### Ticket Status Lifecycle

```mermaid
stateDiagram-v2
    [*] --> OPEN: Ticket Created
    OPEN --> IN_PROGRESS: Agent Assigned
    OPEN --> PENDING_CUSTOMER: Waiting for Customer
    OPEN --> RESOLVED: Issue Fixed
    OPEN --> CLOSED: Resolved & Closed Immediately

    IN_PROGRESS --> PENDING_VENDOR: Escalated to Vendor
    IN_PROGRESS --> PENDING_CUSTOMER: Waiting for Customer
    IN_PROGRESS --> RESOLVED: Issue Fixed

    PENDING_CUSTOMER --> IN_PROGRESS: Customer Responded
    PENDING_CUSTOMER --> RESOLVED: Issue Fixed
    PENDING_CUSTOMER --> CLOSED: No Response

    PENDING_VENDOR --> IN_PROGRESS: Vendor Responded
    PENDING_VENDOR --> RESOLVED: Issue Fixed

    RESOLVED --> CLOSED: Confirmation Received
    RESOLVED --> OPEN: Reopened

    CLOSED --> [*]

    note right of OPEN
        Created at = now()
        Clocks started
    end note

    note right of RESOLVED
        resolution_sla clock stops
    end note
```

### SLA Alert State Machine

```mermaid
stateDiagram-v2
    [*] --> ON_TRACK: New Ticket
    ON_TRACK --> AT_RISK: ≤ 15% remaining
    ON_TRACK --> MET: Deadline met
    AT_RISK --> BREACHED: Deadline passed
    AT_RISK --> MET: Deadline met
    BREACHED --> [*]
    MET --> [*]

    note right of AT_RISK
        Warning alert sent
        Escalation level +1
    end note

    note right of BREACHED
        Breach alert sent
        Maximum escalation
        Slack notification
    end note
```

---

## Clean Architecture Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                              │
│  (Pure Python - No framework dependencies)                    │
├─────────────────────────────────────────────────────────────┤
│  src/sla/domain/                                            │
│  ├── entities.py          Ticket, SLAMetrics, SLAAlert      │
│  └── value_objects.py    SLAConfig, Priority, CustomerTier  │
│                                                             │
│  src/triage/domain/                                          │
│  └── entities.py          TriageTicket, ClassificationResult │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                           │
│  (Business logic - orchestrates domain + infrastructure)       │
├─────────────────────────────────────────────────────────────┤
│  src/sla/application/                                       │
│  ├── dto.py              TicketEntityDTO, IngestResponse      │
│  └── services.py         SLAService, SLAEvaluationService    │
│                                                             │
│  src/triage/application/                                    │
│  ├── dto.py              ClassificationRequest, ResponseDTO   │
│  └── services.py         ClassificationService, RAGService    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                INFRASTRUCTURE LAYER                           │
│  (Database, External APIs - concrete implementations)           │
├─────────────────────────────────────────────────────────────┤
│  src/sla/infrastructure/                                   │
│  └── models.py          TicketModel, AlertModel (SQLAlchemy)│
│                                                             │
│  src/triage/infrastructure/                                │
│  └── models.py          TriageTicketModel, ClassificationModel│
└─────────────────────────────────────────────────────────────┘
                              ↓
                      ┌─────────────┐
                      │  PostgreSQL  │
                      │   Database   │
                      └─────────────┘
```

---

## External Systems Integration

```mermaid
graph LR
    subgraph "AIGenius Ticketing System"
        APP[FastAPI Application]
        SLA[SLA Module]
        TRIAGE[Triage Module]
        DB[(PostgreSQL)]
    end

    subgraph "External Services"
        SLACK[Slack Webhook]
        MILVUS[Milvus/Zilliz<br/>Vector Store]
        GROQ[Groq Llama 3.3<br/>LLM API]
        OPENAI[OpenAI<br/>Embeddings]
    end

    APP --> SLA
    APP --> TRIAGE
    SLA --> DB
    TRIAGE --> DB

    SLA -.->|Alerts| SLACK
    TRIAGE -.->|Search| MILVUS
    TRIAGE -.->|Classification| GROQ
    TRIAGE -.->|Embeddings| OPENAI

    style APP fill:#e1f5ff
    style DB fill:#87CEEB
    style SLACK fill:#FFA500
    style MILVUS fill:#DDA0DD
    style GROQ fill:#90EE90
    style OPENAI fill:#90EE90
```

---

## Enum Values Reference

### Priority Levels (SLA)
| Value | Response Target | Resolution Target | Description |
|-------|-----------------|-------------------|-------------|
| `critical` | 15 min | 2 hours | Production down |
| `high` | 30 min | 4 hours | Major feature broken |
| `medium` | 2 hours | 8 hours | Single feature affected |
| `low` | 4 hours | 24 hours | Minor issues |

### Customer Tier Multipliers
| Tier | Multiplier | Description |
|------|------------|-------------|
| `enterprise` | 0.5x | Faster SLAs |
| `business` | 0.75x | Priority SLAs |
| `standard` | 1.0x | Standard SLAs |
| `free` | 1.5x | Extended SLAs |

### Product Categories (Triage)
| Product | Description |
|---------|-------------|
| `CASB` | Cloud Access Security Broker |
| `SWG` | Secure Web Gateway |
| `ZTNA` | Zero Trust Network Access |
| `DLP` | Data Loss Prevention |
| `SSPM` | SaaS Security Posture Management |
| `CFW` | Cloud Firewall |
| `GENERAL` | Unclassified / Other |

### SLA States
| State | Condition | Action |
|-------|-----------|--------|
| `on_track` | > 15% remaining | Monitor |
| `at_risk` | ≤ 15% remaining | Send warning |
| `breached` | Deadline passed | Escalate |
| `met` | SLA achieved | Clock stopped |

---

## Index Strategy

### Performance Indexes
```sql
-- tickets table
CREATE INDEX idx_tickets_external_id ON tickets(external_id);
CREATE INDEX idx_tickets_status ON tickets(status);
CREATE INDEX idx_tickets_priority ON tickets(priority);
CREATE INDEX idx_tickets_customer_tier ON tickets(customer_tier);
CREATE INDEX idx_tickets_created_at ON tickets(created_at);

-- sla_alerts table
CREATE INDEX idx_sla_alerts_ticket_id ON sla_alerts(ticket_id);
CREATE INDEX idx_sla_alerts_state ON sla_alerts(state);
CREATE INDEX idx_sla_alerts_triggered_at ON sla_alerts(triggered_at);

-- triage_tickets table
CREATE INDEX idx_triage_tickets_external_id ON triage_tickets(external_id);
CREATE INDEX idx_triage_tickets_created_at ON triage_tickets(created_at);

-- classifications table
CREATE INDEX idx_classifications_ticket_id ON classifications(ticket_id);
CREATE INDEX idx_classifications_product ON classifications(product);
CREATE INDEX idx_classifications_urgency ON classifications(urgency);
CREATE INDEX idx_classifications_created_at ON classifications(created_at);
```

---

## Migration Strategy

### Schema Evolution

```mermaid
timeline
    title Database Schema Evolution
    section v1.0 (Initial)
        Create tickets table : Core SLA tracking
        Create sla_alerts table : Alert management
        Create triage_tickets table : RAG support
        Create classifications table : Classification history
    section v1.1 (Future)
        Add sla_metrics table : Historical SLA data
        Add ticket_history table : Status change audit
        Add embeddings table : Cached vectors
```

### Rollback Plan
- All migrations use reversible SQLAlchemy operations
- Foreign keys have `ON DELETE CASCADE` for cleanup
- Indexes can be dropped without data loss
- Timestamps use `timezone=True` for UTC consistency

---

## Data Volume Estimates

| Table | Rows (Year 1) | Rows (Year 3) | Storage |
|-------|---------------|---------------|---------|
| `tickets` | 50,000 | 200,000 | ~100 MB |
| `sla_alerts` | 150,000 | 600,000 | ~150 MB |
| `triage_tickets` | 50,000 | 200,000 | ~50 MB |
| `classifications` | 100,000 | 500,000 | ~200 MB |

**Total (Year 1):** ~500 MB
**Total (Year 3):** ~1.2 GB

---

## Related Files

- **SLA Models**: [`src/sla/infrastructure/models.py`](../src/sla/infrastructure/models.py)
- **Triage Models**: [`src/triage/infrastructure/models.py`](../src/triage/infrastructure/models.py)
- **Configuration**: [`src/config/__init__.py`](../src/config/__init__.py)
- **Database**: [`src/infrastructure/database/__init__.py`](../src/infrastructure/database/__init__.py)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-25
**Author**: Kuldeep Pal
