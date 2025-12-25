# AIGenius Ticketing API - Testing Guide

Complete curl commands to test all endpoints of the AIGenius Ticketing API.

**Base URL:** `http://localhost:8000`

## Table of Contents

- [Health Check](#health-check)
- [SLA Monitoring Module](#sla-monitoring-module)
  - [Ingest Tickets](#1-ingest-tickets)
  - [Get SLA Dashboard](#2-get-sla-dashboard)
  - [Get Ticket SLA Details](#3-get-ticket-sla-details)
- [Triage Module](#triage-module)
  - [Classify Ticket](#4-classify-ticket)
  - [Generate RAG Response](#5-generate-rag-response)
  - [Ingest Documentation](#6-ingest-documentation)
  - [Get Classification Stats](#7-get-classification-stats)

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
        "id": "TICKET-001",
        "priority": "critical",
        "customer_tier": "enterprise",
        "status": "open",
        "subject": "Production CASB outage - Salesforce sync down",
        "content": "Our production CASB integration with Salesforce has completely stopped syncing. This is affecting all our users and is a critical business issue.",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:00:00Z"
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
        "id": "TICKET-002",
        "priority": "high",
        "customer_tier": "enterprise",
        "status": "in_progress",
        "subject": "SWG policy not blocking malicious domains",
        "content": "Secure Web Gateway is allowing access to known malicious domains. Need urgent policy review.",
        "created_at": "2024-01-15T09:00:00Z",
        "updated_at": "2024-01-15T11:30:00Z",
        "first_response_at": "2024-01-15T09:15:00Z"
      },
      {
        "id": "TICKET-003",
        "priority": "medium",
        "customer_tier": "business",
        "status": "open",
        "subject": "ZTNA configuration help needed",
        "content": "Need assistance configuring Zero Trust Network Access for new applications.",
        "created_at": "2024-01-15T08:00:00Z",
        "updated_at": "2024-01-15T08:00:00Z"
      },
      {
        "id": "TICKET-004",
        "priority": "high",
        "customer_tier": "standard",
        "status": "pending_customer",
        "subject": "DLP rule tuning required",
        "content": "Data Loss Prevention is generating too many false positives. Need to tune the rules.",
        "created_at": "2024-01-14T14:00:00Z",
        "updated_at": "2024-01-15T07:00:00Z"
      },
      {
        "id": "TICKET-005",
        "priority": "low",
        "customer_tier": "free",
        "status": "open",
        "subject": "SSPM integration question",
        "content": "How do I integrate SaaS Security Posture Management with our existing SaaS applications?",
        "created_at": "2024-01-14T10:00:00Z",
        "updated_at": "2024-01-14T10:00:00Z"
      },
      {
        "id": "TICKET-006",
        "priority": "critical",
        "customer_tier": "business",
        "status": "open",
        "subject": "CFW blocking legitimate traffic",
        "content": "Cloud Firewall is incorrectly blocking legitimate business traffic. Urgent fix needed.",
        "created_at": "2024-01-15T07:00:00Z",
        "updated_at": "2024-01-15T07:00:00Z"
      },
      {
        "id": "TICKET-007",
        "priority": "medium",
        "customer_tier": "enterprise",
        "status": "resolved",
        "subject": "API rate limiting issue",
        "content": "Need to increase API rate limits for our integration.",
        "created_at": "2024-01-14T08:00:00Z",
        "updated_at": "2024-01-15T10:00:00Z",
        "resolved_at": "2024-01-15T10:00:00Z"
      }
    ]
  }' | jq
```

**Expected Response:**
```json
{
  "created": 6,
  "updated": 0,
  "failed": 0,
  "errors": []
}
```

**Available Values:**
- **Priority:** `critical`, `high`, `medium`, `low`
- **Customer Tier:** `enterprise`, `business`, `standard`, `free`
- **Status:** `open`, `in_progress`, `pending_customer`, `pending_vendor`, `resolved`, `closed`

---

### 2. Get SLA Dashboard

View all tickets with SLA status and summary statistics.

#### All Tickets (Default)

```bash
curl -s "http://localhost:8000/sla/dashboard" | jq
```

#### Filtered - Enterprise Critical Tickets

```bash
curl -s "http://localhost:8000/sla/dashboard?customer_tier=enterprise&priority=critical" | jq
```

#### Filtered - Only Breached SLAs

```bash
curl -s "http://localhost:8000/sla/dashboard?sla_state=breached" | jq
```

#### Filtered - High Priority, In Progress

```bash
curl -s "http://localhost:8000/sla/dashboard?priority=high&status=in_progress" | jq
```

#### With Pagination

```bash
curl -s "http://localhost:8000/sla/dashboard?limit=5&offset=0" | jq
```

**Expected Response Structure:**
```json
{
  "tickets": [
    {
      "ticket_id": "uuid",
      "external_id": "TICKET-001",
      "priority": "critical",
      "customer_tier": "enterprise",
      "status": "open",
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-15T10:00:00Z",
      "response_sla": {
        "deadline": "2024-01-15T10:15:00Z",
        "remaining_seconds": 900,
        "percentage_remaining": 25.0,
        "is_breached": false,
        "state": "on_track",
        "met_at": null
      },
      "resolution_sla": {
        "deadline": "2024-01-15T12:00:00Z",
        "remaining_seconds": 6300,
        "percentage_remaining": 87.5,
        "is_breached": false,
        "state": "on_track",
        "met_at": null
      },
      "overall_state": "on_track",
      "next_deadline": "2024-01-15T10:15:00Z",
      "active_alerts": []
    }
  ],
  "total_count": 7,
  "summary": {
    "total_tickets": 7,
    "breached_count": 2,
    "at_risk_count": 1,
    "on_track_count": 3,
    "met_count": 1,
    "breach_rate": 28.57
  }
}
```

**SLA States:**
- `breached` - SLA deadline has passed
- `at_risk` - Warning threshold exceeded (80% of SLA used)
- `on_track` - SLA is within safe limits
- `met` - SLA was successfully achieved

---

### 3. Get Ticket SLA Details

Get detailed SLA information for a specific ticket.

```bash
curl -s "http://localhost:8000/sla/tickets/TICKET-001" | jq
```

**Note:** Replace `TICKET-001` with the actual ticket ID from the database. Use the UUID from the dashboard response.

```bash
# Get the ticket ID from dashboard first
TICKET_ID=$(curl -s "http://localhost:8000/sla/dashboard?limit=1" | jq -r '.tickets[0].ticket_id')

# Then get detailed SLA information
curl -s "http://localhost:8000/sla/tickets/$TICKET_ID" | jq
```

**Expected Response:**
```json
{
  "ticket_id": "uuid",
  "external_id": "TICKET-001",
  "priority": "critical",
  "customer_tier": "enterprise",
  "status": "open",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "response_sla": {
    "deadline": "2024-01-15T10:15:00Z",
    "remaining_seconds": 900,
    "percentage_remaining": 25.0,
    "is_breached": false,
    "state": "on_track",
    "met_at": null
  },
  "resolution_sla": {
    "deadline": "2024-01-15T12:00:00Z",
    "remaining_seconds": 6300,
    "percentage_remaining": 87.5,
    "is_breached": false,
    "state": "on_track",
    "met_at": null
  },
  "overall_state": "on_track",
  "next_deadline": "2024-01-15T10:15:00Z",
  "active_alerts": [
    {
      "id": "alert-uuid",
      "sla_type": "response",
      "alert_type": "warning",
      "triggered_at": "2024-01-15T10:05:00Z",
      "deadline": "2024-01-15T10:15:00Z",
      "remaining_seconds": 600,
      "state": "at_risk",
      "escalation_level": 1,
      "notification_sent": false,
      "channels": ["slack", "email"]
    }
  ]
}
```

---

## Triage Module

### 4. Classify Ticket

Classify a ticket by product area and urgency using GLM 4.7 LLM.

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

#### SSPM Ticket Classification

```bash
curl -s -X POST http://localhost:8000/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-105",
    "subject": "SSPM not detecting misconfigurations",
    "content": "SaaS Security Posture Management is not reporting security misconfigurations in our Office 365 tenant."
  }' | jq
```

#### CFW Ticket Classification

```bash
curl -s -X POST http://localhost:8000/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-106",
    "subject": "Cloud Firewall rules not applying",
    "content": "Cloud Firewall rules are not being applied to new VPC deployments. Traffic is not being filtered correctly."
  }' | jq
```

**Expected Response:**
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

### 5. Generate RAG Response

Generate AI-powered responses using Retrieval Augmented Generation.

#### Basic Query

```bash
curl -s -X POST http://localhost:8000/triage/respond \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-001",
    "query": "How do I configure CASB for Salesforce integration?"
  }' | jq
```

#### Troubleshooting Query

```bash
curl -s -X POST http://localhost:8000/triage/respond \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-002",
    "query": "What are the steps to troubleshoot SWG policy not blocking traffic?"
  }' | jq
```

#### Configuration Query

```bash
curl -s -X POST http://localhost:8000/triage/respond \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-003",
    "query": "How do I set up ZTNA for custom applications?"
  }' | jq
```

**Expected Response:**
```json
{
  "ticket_id": "TICKET-001",
  "response": "To configure CASB for Salesforce integration, follow these steps:\n\n1. Navigate to Settings > CASB > Salesforce\n2. Enter your Salesforce credentials\n3. Configure sync preferences\n4. Test the connection\n\nFor detailed instructions, see the CASB Configuration Guide [1].",
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
```

**Note:** If no documents are indexed, the response will be:
```json
{
  "ticket_id": "TICKET-001",
  "response": "No documentation has been indexed yet. Please use the POST /ingest endpoint to add documents first.",
  "citations": [],
  "sources_used": 0,
  "processing_time_ms": 475
}
```

---

### 6. Ingest Documentation

Index documentation for RAG-based responses.

#### Ingest from URL (Web Scraping)

```bash
curl -s -X POST http://localhost:8000/triage/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.example.com/casb-configuration",
    "max_pages": 10
  }' | jq
```

**Expected Response (Web Scraping Not Yet Implemented):**
```json
{
  "status": "not_implemented",
  "message": "Web scraping not yet implemented for P0",
  "documents_processed": 0,
  "chunks_created": 0,
  "url": "https://docs.netskope.com/casb-configuration"
}
```

---

### 7. Get Classification Stats

View classification statistics and document index status.

```bash
curl -s http://localhost:8000/triage/stats | jq
```

**Expected Response:**
```json
{
  "classifications_total": 6,
  "documents_indexed": 0,
  "product_distribution": {
    "CASB": 2,
    "SWG": 1,
    "ZTNA": 1,
    "DLP": 1,
    "SSPM": 1,
    "CFW": 0,
    "GENERAL": 0
  }
}
```

---

## Quick Test Script

Save this as `test_api.sh` and run to test all endpoints:

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

echo "=== AIGenius Ticketing API Test ==="
echo ""

echo "1. Health Check"
curl -s $BASE_URL/health | jq
echo -e "\n"

echo "2. Ingest SLA Tickets"
curl -s -X POST $BASE_URL/sla/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      {
        "id": "TEST-001",
        "priority": "high",
        "customer_tier": "enterprise",
        "status": "open",
        "subject": "Test ticket",
        "content": "Test content",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:00:00Z"
      }
    ]
  }' | jq
echo -e "\n"

echo "3. Get SLA Dashboard"
curl -s "$BASE_URL/sla/dashboard?limit=3" | jq
echo -e "\n"

echo "4. Classify Ticket"
curl -s -X POST $BASE_URL/triage/classify \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TEST-002",
    "subject": "CASB issue",
    "content": "CASB not working"
  }' | jq
echo -e "\n"

echo "5. Get Triage Stats"
curl -s $BASE_URL/triage/stats | jq
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

## Tips for Testing

1. **Use `jq` for pretty output**: All commands include `| jq` for formatted JSON output. Install jq: `brew install jq` (macOS) or `apt install jq` (Linux).

2. **Check database for ticket IDs**: After ingesting tickets, use the dashboard to get actual ticket UUIDs for detailed queries.

3. **Monitor background jobs**: The SLA evaluation job runs every 60 seconds. Watch server logs for alert creation.

4. **Test SLA breaches**: Create tickets with old `created_at` timestamps to test breached SLA scenarios.

5. **Test different customer tiers**: Ingest tickets with different tiers to see varying SLA deadlines.

6. **Check the OpenAPI spec**: Visit http://localhost:8000/docs for interactive API documentation with "Try it out" buttons.

---

## Troubleshooting

### Server not responding
```bash
# Check if server is running
lsof -ti:8000

# Restart server (using UV - recommended)
cd modular-monolith
uv run python -m src.main

# Or using traditional venv
cd modular-monolith
source venv/bin/activate
PYTHONPATH=src python -m src.main
```

### Database errors
```bash
# Check PostgreSQL is running
pg_isready

# View database logs
tail -f /usr/local/var/log/postgres.log
```

### LLM API errors
The classification endpoint may return 429 errors if the LLM API quota is exceeded. This is an external service limitation.

---

## Additional Resources

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health
