# AIGenius Ticketing API - Testing Guide

Complete curl commands to test all endpoints of the AIGenius Ticketing API using **Groq Llama 3.3** (ultra-fast, free LLM).

**Base URL:** `http://localhost:8000`

## Table of Contents

- [Setup](#setup)
- [Health Check](#health-check)
- [SLA Monitoring Module](#sla-monitoring-module)
- [Triage Module](#triage-module)
- [Quick Test Script](#quick-test-script)
- [SLA Time Reference](#sla-time-reference)

---

## Setup

### Prerequisites

```bash
# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install jq for pretty JSON output (optional)
brew install jq  # macOS
# or
apt install jq   # Linux
```

### Environment Variables

Create or update `.env` file:

```bash
# Groq LLM Configuration (Fast & Free)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=llama-3.3-70b-versatile
MOCK_LLM=false

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
```

**Get a free Groq API key:** https://groq.com

### Start the Server

```bash
# Using UV (recommended)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or with explicit environment
GROQ_API_KEY=your-key LLM_MODEL=llama-3.3-70b-versatile uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## Health Check

### Check Service Health

```bash
curl -s http://localhost:8000/health | jq
```

**Expected Response:**
```json
{
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
```

---

## SLA Monitoring Module

### 1. Ingest Tickets

Create or update tickets for SLA tracking.

#### Single Ticket - Enterprise Critical

```bash
curl -s -X POST http://localhost:8000/sla/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      {
        "external_id": "TICKET-001",
        "priority": "critical",
        "customer_tier": "enterprise",
        "status": "open",
        "subject": "Production CASB outage - Salesforce sync down",
        "content": "Our production CASB integration with Salesforce has completely stopped syncing. This is affecting all our users and is a critical business issue."
      }
    ]
  }' | jq
```

#### Multiple Tickets - Various Priorities and Tiers

```bash
curl -s -X POST http://localhost:8000/sla/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      {
        "external_id": "TICKET-002",
        "priority": "high",
        "customer_tier": "enterprise",
        "status": "in_progress",
        "subject": "SWG policy not blocking malicious domains",
        "content": "Secure Web Gateway is allowing access to known malicious domains. Need urgent policy review."
      },
      {
        "external_id": "TICKET-003",
        "priority": "medium",
        "customer_tier": "business",
        "status": "open",
        "subject": "ZTNA configuration help needed",
        "content": "Need assistance configuring Zero Trust Network Access for new applications."
      },
      {
        "external_id": "TICKET-004",
        "priority": "high",
        "customer_tier": "standard",
        "status": "pending_customer",
        "subject": "DLP rule tuning required",
        "content": "Data Loss Prevention is generating too many false positives. Need to tune the rules."
      }
    ]
  }' | jq
```

**Available Values:**
- **Priority:** `critical`, `high`, `medium`, `low`
- **Customer Tier:** `enterprise`, `business`, `standard`, `free`
- **Status:** `open`, `in_progress`, `pending_customer`, `pending_vendor`, `resolved`, `closed`

---

### 2. Get SLA Dashboard

View all tickets with SLA status and summary statistics.

```bash
curl -s "http://localhost:8000/sla/dashboard" | jq
```

**Response Summary:**
```json
{
  "total_count": 4,
  "summary": {
    "total_tickets": 4,
    "breached_count": 2,
    "at_risk_count": 1,
    "on_track_count": 1,
    "met_count": 0
  }
}
```

---

## Triage Module

### 3. Classify Ticket (Groq Llama 3.3)

Classify a ticket by product area and urgency using **Groq Llama 3.3-70b-versatile**.

#### CASB Ticket Classification

```bash
curl -s -X POST http://localhost:8000/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-101",
    "subject": "CASB Salesforce sync not working",
    "content": "Our CASB integration with Salesforce has stopped syncing data since yesterday. All users are affected."
  }' | jq
```

**Response:**
```json
{
  "ticket_id": "uuid",
  "classification": {
    "product": "CASB",
    "urgency": "high",
    "confidence": 0.9,
    "reasoning": "CASB integration issue affecting business operations with sync failure."
  },
  "processing_time_ms": 1100
}
```

#### SWG Ticket Classification

```bash
curl -s -X POST http://localhost:8000/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-102",
    "subject": "SWG allowing access to phishing sites",
    "content": "Secure Web Gateway is not blocking known phishing domains. Multiple users reported accessing malicious sites."
  }' | jq
```

#### ZTNA Ticket Classification

```bash
curl -s -X POST http://localhost:8000/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-103",
    "subject": "Cannot access internal applications via ZTNA",
    "content": "Users are unable to connect to internal applications through Zero Trust Network Access. Getting connection timeout errors."
  }' | jq
```

#### DLP Ticket Classification

```bash
curl -s -X POST http://localhost:8000/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-104",
    "subject": "DLP blocking legitimate documents",
    "content": "Data Loss Prevention is flagging internal documents as sensitive. Users cannot share legitimate business documents."
  }' | jq
```

**Product Classifications:**
- `CASB` - Cloud Access Security Broker
- `SWG` - Secure Web Gateway
- `ZTNA` - Zero Trust Network Access
- `DLP` - Data Loss Prevention
- `SSPM` - SaaS Security Posture Management
- `CFW` - Cloud Firewall
- `GENERAL` - General inquiries

**Urgency Levels:**
- `critical` - Production down, critical business impact
- `high` - Major functionality affected
- `medium` - Partial functionality affected
- `low` - Minor issues or questions

---

### 4. Generate RAG Response

Generate AI-powered responses using Retrieval Augmented Generation.

```bash
curl -s -X POST http://localhost:8000/triage/respond \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-001",
    "query": "How do I configure CASB for Salesforce integration?"
  }' | jq
```

---

### 5. Get Classification Stats

View classification statistics and document index status.

```bash
curl -s http://localhost:8000/triage/stats | jq
```

**Response:**
```json
{
  "classifications_total": 4,
  "documents_indexed": 0,
  "product_distribution": {
    "CASB": 1,
    "SWG": 1,
    "ZTNA": 1,
    "DLP": 1
  }
}
```

---

## Quick Test Script

Save this as `test_api.sh` and run to test all endpoints:

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

echo "=== AIGenius Ticketing API Test (Groq LLM) ==="
echo ""

echo "1. Health Check"
curl -s $BASE_URL/health | jq '.status, .checks.llm_client'
echo -e "\n"

echo "2. Classify CASB Ticket"
curl -s -X POST $BASE_URL/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TEST-001",
    "subject": "CASB Salesforce sync issue",
    "content": "Our CASB integration with Salesforce has stopped syncing data."
  }' | jq '.classification'
echo -e "\n"

echo "3. Classify SWG Ticket"
curl -s -X POST $BASE_URL/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TEST-002",
    "subject": "SWG blocking legitimate sites",
    "content": "Secure Web Gateway is blocking business-critical websites."
  }' | jq '.classification'
echo -e "\n"

echo "4. Get Triage Stats"
curl -s $BASE_URL/triage/stats | jq
echo -e "\n"

echo "5. SLA Dashboard Summary"
curl -s $BASE_URL/sla/dashboard | jq '.summary'
echo -e "\n"

echo "=== Test Complete ==="
```

Run the script:
```bash
chmod +x test_api.sh
./test_api.sh
```

---

## SLA Time Reference

### Response SLA (Time to First Response)

| Priority | Enterprise | Business | Standard | Free |
|----------|-----------|----------|----------|------|
| Critical | 15 min    | 30 min   | 60 min   | 120 min |
| High     | 30 min    | 60 min   | 120 min  | 240 min |
| Medium   | 60 min    | 120 min  | 240 min  | 480 min |
| Low      | 120 min   | 240 min  | 480 min  | 960 min |

### Resolution SLA (Time to Complete Resolution)

| Priority | Enterprise | Business | Standard | Free |
|----------|-----------|----------|----------|------|
| Critical | 2 hours   | 4 hours  | 8 hours  | 16 hours |
| High     | 4 hours   | 8 hours  | 16 hours | 32 hours |
| Medium   | 8 hours   | 16 hours | 32 hours | 64 hours |
| Low      | 16 hours  | 32 hours | 64 hours | 128 hours |

---

## LLM Provider Configuration

The system supports multiple LLM providers (used in priority order):

1. **Groq** (Default) - `llama-3.3-70b-versatile`
   - Fastest inference (~1-2 seconds)
   - Free tier available
   - Get API key: https://groq.com

2. **OpenAI** - `gpt-4o`
   - Set `OPENAI_API_KEY` in .env
   - Requires paid account

3. **Z.AI** - `glm-4.7`
   - Set `ZAI_API_KEY` in .env
   - Requires account balance

### Switching LLM Providers

```bash
# Use Groq (default)
GROQ_API_KEY=your-key LLM_MODEL=llama-3.3-70b-versatile

# Use OpenAI
OPENAI_API_KEY=your-key LLM_MODEL=gpt-4o

# Use Z.AI
ZAI_API_KEY=your-key LLM_MODEL=glm-4.7
```

---

## Troubleshooting

### LLM Not Available (503 Error)

```json
{
  "detail": "Classification service not available - LLM not configured"
}
```

**Solution:** Set `GROQ_API_KEY` in your `.env` file or environment variables.

### Database Connection Error

```bash
# Check PostgreSQL is running
pg_isready

# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL
```

### View Server Logs

```bash
# If running in foreground
# Logs are displayed directly in terminal

# Check for errors
tail -f logs/app.log
```

---

## Additional Resources

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Groq Console**: https://console.groq.com
- **Health Check**: http://localhost:8000/health
