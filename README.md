# AIGenius - AI-Powered Customer Support Ticketing System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deploy](https://img.shields.io/badge/Deploy-Render-3895EF)](https://aigenius-ticketing.onrender.com)
[![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange)](https://groq.com)

An intelligent customer support automation platform with SLA monitoring, AI-powered ticket classification, and RAG-based response generation.

## ✨ What's New

- **Multi-Provider LLM Support**: Groq (default), OpenAI, Z.AI with automatic fallback
- **Production Deployed**: Live on Render with zero-downtime deployments
- **Container-Safe Config**: Optional file watching for Docker/production environments
- **Hot Reload SLA Config**: YAML configuration with automatic reload (local)

## 🚀 Features

| Module | Capability | Description |
|--------|-------------|-------------|
| **SLA Monitoring** | Real-time Tracking | Monitor response and resolution SLAs with automated alerts |
| **AI Classification** | Groq Llama 3.3 | Auto-classify tickets by product and urgency |
| **Smart Responses** | RAG with Milvus | Generate contextual responses from knowledge base |
| **Alert System** | Slack Integration | Real-time notifications for SLA breaches |

## 🏗️ Architecture

**Modular Monolith** - Clean architecture with clear module boundaries.

### High-Level Design (HLD)

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web UI]
        API_CLIENT[API Clients]
    end

    subgraph "API Gateway - FastAPI"
        SWAGGER[Swagger UI<br>/docs]
        HEALTH[Health Check<br>/health]
        SLA_API[SLA Endpoints<br>/sla/*]
        TRIAGE_API[Triage Endpoints<br>/triage/*]
    end

    subgraph "SLA Module"
        SLA_CONTROLLER[SLA Controller]
        SLA_SERVICE[SLA Service]
        SLA_REPO[(SLA Repository)]
        SLA_SCHEDULER[APScheduler<br>Background Jobs]
        SLA_CONFIG[YAML Config<br>Hot Reload]
        ALERT_MGR[Alert Manager]
        SLACK[Slack Webhook<br>Notifications]
    end

    subgraph "Triage Module (AI)"
        TRIAGE_CONTROLLER[Triage Controller]
        CLASSIFIER[Ticket Classifier<br>Product & Urgency]
        RAG_ENGINE[RAG Engine<br>Response Generation]
        VEC_STORE[(Milvus<br>Vector Store)]
    end

    subgraph "LLM Layer - Multi-Provider"
        GROQ[Groq Llama 3.3<br>Default - Free]
        OPENAI[OpenAI GPT-4o<br>Fallback]
        ZAI[Z.AI GLM 4.7<br>Fallback]
    end

    subgraph "Data Layer"
        POSTGRES[(PostgreSQL<br>Async)]
        TICKETS[tickets table]
        ALERTS[sla_alerts table]
        CLASSIFICATIONS[classifications table]
    end

    subgraph "Infrastructure"
        LOGS[Structured Logging<br>JSON + Correlation IDs]
        METRICS[OTLP Metrics<br>Grafana]
        CIRCUIT[Circuit Breaker<br>Resilience]
    end

    %% Client Connections
    WEB --> SWAGGER
    WEB --> HEALTH
    API_CLIENT --> SLA_API
    API_CLIENT --> TRIAGE_API

    %% SLA Flow
    SLA_API --> SLA_CONTROLLER
    SLA_CONTROLLER --> SLA_SERVICE
    SLA_SERVICE --> SLA_REPO
    SLA_SERVICE --> SLA_CONFIG
    SLA_SCHEDULER -.-> SLA_SERVICE
    SLA_SERVICE --> ALERT_MGR
    ALERT_MGR --> SLACK

    %% Triage Flow
    TRIAGE_API --> TRIAGE_CONTROLLER
    TRIAGE_CONTROLLER --> CLASSIFIER
    TRIAGE_CONTROLLER --> RAG_ENGINE
    CLASSIFIER --> GROQ
    GROQ -.->|fallback| OPENAI
    OPENAI -.->|fallback| ZAI
    RAG_ENGINE --> VEC_STORE
    RAG_ENGINE --> GROQ

    %% Data Persistence
    SLA_REPO --> POSTGRES
    POSTGRES --> TICKETS
    POSTGRES --> ALERTS
    CLASSIFIER --> CLASSIFICATIONS
    CLASSIFICATIONS --> POSTGRES

    %% Cross-cutting
    SLA_SERVICE -.-> LOGS
    TRIAGE_CONTROLLER -.-> LOGS
    RAG_ENGINE -.-> METRICS
    ALERT_MGR -.-> CIRCUIT

    style GROQ fill:#90EE90,stroke:#333,stroke-width:2px
    style POSTGRES fill:#87CEEB,stroke:#333,stroke-width:2px
    style VEC_STORE fill:#DDA0DD,stroke:#333,stroke-width:2px
    style SLACK fill:#FFA500,stroke:#333,stroke-width:2px
```

### Component Details

| Layer | Component | Description |
|-------|-----------|-------------|
| **API** | FastAPI | Async web framework with auto-generated docs |
| **SLA** | APScheduler | Background evaluation every 60s |
| **SLA** | YAML Watchdog | Hot-reload config changes |
| **AI** | Groq Llama 3.3 | Ultra-fast inference (1-2s latency) |
| **RAG** | Milvus/Zilliz | Semantic vector search |
| **Data** | PostgreSQL + AsyncPG | Async database operations |
| **Alerts** | Slack Webhook | Real-time breach notifications |
| **Metrics** | OTLP | Grafana integration |

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Package Manager | **UV** (fast package manager) |
| Framework | FastAPI |
| Database | PostgreSQL (async) |
| LLM | **Groq (Llama 3.3-70b-versatile)** - Fast & Free |
| Vector Store | Milvus (Zilliz Cloud) |
| Scheduler | APScheduler |
| Config | YAML with hot-reload |
| Metrics | Grafana OTLP |

## 📦 Quick Start

**Using UV (Recommended):**

```bash
cd aigenius-ticketing

# Install UV (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the service
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Using Docker:**

```bash
cd aigenius-ticketing
docker-compose up --build
```

## 🌐 API Documentation

**Live Demo (Production):**
- **Swagger UI**: https://aigenius-ticketing.onrender.com/docs
- **Health Check**: https://aigenius-ticketing.onrender.com/health

**Local Development:**
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## ⚙️ Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

# LLM (Groq - Fast & Free)
GROQ_API_KEY=your-groq-api-key
LLM_MODEL=llama-3.3-70b-versatile

# Vector Store (Optional - for RAG)
ZILLIZ_URI=your-zilliz-uri
ZILLIZ_API_KEY=your-zilliz-key

# Slack Alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
SLACK_CHANNEL=#support-alerts

# Grafana Metrics (Optional)
GRAFANA_HOST=https://otlp-gateway-prod-ap-south-1.grafana.net/otlp/v1/metrics
GRAFANA_API_KEY=your-grafana-key
```

## 📊 Endpoints

### SLA Monitoring
- `POST /sla/tickets` - Ingest tickets for SLA tracking
- `GET /sla/dashboard` - View all tickets with SLA status
- `GET /sla/tickets/{id}` - Get detailed SLA information

### AI Triage
- `POST /triage/classify` - Classify ticket by product and urgency
- `POST /triage/respond` - Generate AI-powered response (RAG)
- `GET /triage/stats` - View classification statistics

## 🧪 API Testing Examples

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "checks": {
    "database": "connected",
    "llm_client": "available"
  }
}
```

### Classify Ticket
```bash
curl -X POST http://localhost:8000/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-001",
    "subject": "CASB Salesforce sync issue",
    "content": "Our CASB integration with Salesforce has stopped syncing data."
  }'
```

**Response:**
```json
{
  "ticket_id": "uuid",
  "classification": {
    "product": "CASB",
    "urgency": "high",
    "confidence": 0.9,
    "reasoning": "CASB integration issue affecting data sync."
  },
  "processing_time_ms": 1100
}
```

### SLA Dashboard
```bash
curl http://localhost:8000/sla/dashboard
```

## 📝 Configuration

### SLA Configuration (`sla_config.yaml`)

```yaml
sla_targets:
  critical:
    response: 15      # minutes
    resolution: 120   # minutes
  high:
    response: 30
    resolution: 240

customer_tier_multipliers:
  enterprise: 0.5
  business: 0.75
  standard: 1.0
  free: 1.5
```

## 🔄 Project Structure

```
aigenius-ticketing/
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── config/                 # Pydantic settings & constants
│   ├── shared/
│   │   ├── infrastructure/     # Logging, database, exceptions
│   │   └── domain/             # Base entities & value objects
│   ├── sla/                    # SLA Monitoring Module
│   │   ├── domain/             # Ticket, Alert, SLA value objects
│   │   ├── application/        # Services & use cases
│   │   ├── infrastructure/     # Repository, Slack, Scheduler
│   │   └── interfaces/         # REST controllers
│   ├── triage/                 # AI Classification & RAG Module
│   │   ├── domain/             # Ticket, Classification entities
│   │   ├── application/        # Services & use cases
│   │   ├── infrastructure/     # LLM client, Vector store
│   │   └── interfaces/         # REST controllers
│   └── infrastructure/
│       └── llm/                # Multi-provider LLM clients
├── docs/
│   ├── images/                 # Screenshots (Slack, Grafana)
│   └── milvus_import.json      # Sample documentation for RAG
├── scripts/
│   └── prepare_milvus_import.py # Generate embeddings for import
├── docker-compose.yaml         # Local development
├── Dockerfile                  # Production deployment
├── render.yaml                 # Render deployment config
├── pyproject.toml              # Dependencies (UV)
├── requirements.txt            # Dependencies (pip/Render)
├── .env                        # Environment variables
├── .env.example                # Environment template
├── sla_config.yaml             # SLA configuration (optional)
├── API_TESTING.md              # API Testing Guide
└── README.md                   # This file
```

### Clean Architecture Layers

```
src/
├── interfaces/     # API Controllers (FastAPI routes)
├── application/    # Business Logic (Services, Use Cases)
├── domain/         # Core Business Models (Entities, Value Objects)
└── infrastructure/ # External Services (DB, LLM, Slack, etc.)
```

## 🚀 Deployment

### Production (Render)

The application is deployed on Render: https://aigenius-ticketing.onrender.com

**Deployment Features:**
- Automatic deployments from `main` branch
- PostgreSQL database managed by Render
- GitHub Secrets for sensitive config
- Health check for monitoring
- Zero-downtime deployments

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env
```

**Required Variables:**
```bash
DATABASE_URL=postgresql+asyncpg://...
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
```

**Optional Variables:**
```bash
# Vector Store (for RAG)
ZILLIZ_URI=https://...
ZILLIZ_API_KEY=...

# Slack Alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
SLACK_CHANNEL=#alerts

# Grafana Metrics
GRAFANA_HOST=https://otlp-gateway-...
GRAFANA_API_KEY=...
```

## 📈 Metrics & Monitoring

- Structured JSON logging with correlation IDs
- Grafana OTLP metrics integration
- Health check endpoint with system status
- SLA breach alerts to Slack

### Slack SLA Breach Alerts
Real-time notifications sent to Slack when SLA thresholds are at risk or breached.

![Slack SLA Breach Alert](docs/images/slack-breach-alert.png)

### Grafana Monitoring Dashboard
Track LLM performance, API latency, and system metrics with OTLP integration.

![Grafana LLM Monitoring](docs/images/grafana-monitoring.png)



## 🤖 AI Models

### Classification (Groq Llama 3.3-70b)
- **Products**: CASB, SWG, ZTNA, DLP, SSPM, CFW, GENERAL
- **Urgency**: critical, high, medium, low
- **Confidence**: 0.0 - 1.0 score
- **Latency**: ~1-2 seconds (ultra-fast inference)

### Supported LLM Providers (Priority Order)
1. **Groq** (Default) - Llama 3.3, ultra-fast, free
2. OpenAI - GPT-4o (requires API key)
3. Z.AI - GLM 4.7 (requires API key)

### RAG (Retrieval Augmented Generation)
- **Vector Store**: Milvus with semantic search
- **Embeddings**: 768 dimensions (Groq-compatible)
- **Top-K**: 5 documents retrieved

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **GitHub**: https://github.com/kuldeep27396/aigenius-ticketing
- **Groq**: https://groq.com (Get free API key)
- **Zilliz Cloud**: https://zilliz.com (Vector database)
