# AIGenius - AI-Powered Customer Support Ticketing System

An intelligent customer support automation platform with SLA monitoring, AI-powered ticket classification, and RAG-based response generation.

## 🚀 Features

| Module | Capability | Description |
|--------|-------------|-------------|
| **SLA Monitoring** | Real-time Tracking | Monitor response and resolution SLAs with automated alerts |
| **AI Classification** | GLM 4.7 LLM | Auto-classify tickets by product and urgency |
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
| LLM | GLM 4.7 |
| Vector Store | Milvus (Zhipu Cloud) |
| Scheduler | APScheduler |
| Config | YAML with hot-reload |

## 📦 Quick Start

**Using UV (Recommended):**

```bash
cd aigenius-ticketing

# Install UV (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the service
uv run python -m src.main
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

# LLM (GLM 4.7)
ZAI_API_KEY=your-zai-api-key

# Slack Alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
SLACK_CHANNEL=#support-alerts
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
├── k8s/                        # Kubernetes manifests
├── config/                     # Configuration
├── docker-compose.yaml         # Local development
├── pyproject.toml              # Dependencies (UV)
├── .env                        # Environment variables
├── sla_config.yaml             # SLA configuration
└── README.md                   # This file
```

## 🧪 Testing

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest
```

## 📈 Metrics & Monitoring

- Structured JSON logging with correlation IDs
- Prometheus metrics integration
- Health check endpoint with system status
- SLA breach alerts to Slack

## 🤖 AI Models

### Classification (GLM 4.7)
- **Products**: CASB, SWG, ZTNA, DLP, SSPM, CFW, GENERAL
- **Urgency**: critical, high, medium, low
- **Confidence**: 0.0 - 1.0 score

### RAG (Retrieval Augmented Generation)
- **Vector Store**: Milvus with semantic search
- **Embeddings**: 1024 dimensions
- **Top-K**: 5 documents retrieved

## 📄 License

MIT License - see LICENSE file for details.
