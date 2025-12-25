# AIGenius - AI-Powered Customer Support Ticketing System

An intelligent customer support automation platform with SLA monitoring, AI-powered ticket classification, and RAG-based response generation.

## 🚀 Features

| Module | Capability | Description |
|--------|-------------|-------------|
| **SLA Monitoring** | Real-time Tracking | Monitor response and resolution SLAs with automated alerts |
| **AI Classification** | Groq Llama 3.3 | Auto-classify tickets by product and urgency |
| **Smart Responses** | RAG with Milvus | Generate contextual responses from knowledge base |
| **Alert System** | Slack Integration | Real-time notifications for SLA breaches |

## 🏗️ Architecture

**Modular Monolith** - Clean architecture with clear module boundaries.

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

Once running, visit:
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
│   ├── main.py                 # FastAPI application
│   ├── shared/                 # Shared infrastructure
│   ├── sla/                    # SLA Monitoring module
│   └── triage/                 # AI Classification & RAG module
├── docker-compose.yaml         # Local development
├── Dockerfile                  # Production deployment
├── pyproject.toml              # Dependencies (UV)
├── requirements.txt            # Dependencies (pip/Render)
├── .env                        # Environment variables
├── sla_config.yaml             # SLA configuration
├── API_TESTING.md              # API Testing Guide
└── README.md                   # This file
```

## 📈 Metrics & Monitoring

- Structured JSON logging with correlation IDs
- Grafana OTLP metrics integration
- Health check endpoint with system status
- SLA breach alerts to Slack

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
