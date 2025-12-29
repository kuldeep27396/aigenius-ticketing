# Part B - Theory Questions & Implementation Checklist

**Repository**: [aigenius-ticketing](https://github.com/kuldeep27396/aigenius-ticketing)
**Live Demo**: https://aigenius-ticketing.onrender.com/docs
**Author**: Kuldeep Pal

---

## Part A: Implementation Checklist

### Scenario I: Support SLA Monitoring Service

| ID | Requirement | Status | Code Reference |
|----|-------------|--------|----------------|
| FR-1 | POST /tickets endpoint (FastAPI) | ✅ | [`src/sla/interfaces/controllers.py:127`](../src/sla/interfaces/controllers.py#L127) |
| FR-2 | PostgreSQL persistence | ✅ | [`src/sla/infrastructure/models.py:17`](../src/sla/infrastructure/models.py#L17) |
| FR-3 | SLA Engine (scheduler) | ✅ | [`src/sla/infrastructure/external.py:355`](../src/sla/infrastructure/external.py#L355) |
| FR-4 | Escalation workflow (Slack) | ✅ | [`src/sla/infrastructure/external.py:204`](../src/sla/infrastructure/external.py#L204) |
| FR-5 | YAML Config + hot-reload | ✅ | [`src/sla/infrastructure/external.py:47`](../src/sla/infrastructure/external.py#L47) |
| FR-6 | Query endpoints | ✅ | [`src/sla/interfaces/controllers.py:250`](../src/sla/interfaces/controllers.py#L250) |
| FR-7 | WebSocket alerts | ❌ P2 | Not implemented |
| FR-8 | Structured logging | ✅ | [`src/shared/infrastructure/logging.py:13`](../src/shared/infrastructure/logging.py#L13) |
| FR-9 | Docker + Local Dev | ✅ | [`docker-compose.yaml`](../docker-compose.yaml) |
| FR-10 | Cloud IaC (AWS/GCP) | ❌ P2 | Deployed to Render instead |

### Scenario II: Ticket Triage Service

| ID | Requirement | Status | Code Reference |
|----|-------------|--------|----------------|
| FR-1 | /classify and /respond endpoints | ✅ | [`src/triage/interfaces/controllers.py:28`](../src/triage/interfaces/controllers.py#L28) |
| FR-2 | Vector DB ingestion | ✅ | [`src/infrastructure/vectorstore/__init__.py:1`](../src/infrastructure/vectorstore/__init__.py) |
| FR-3 | RAG pipeline | ✅ | [`src/triage/application/services.py:87`](../src/triage/application/services.py#L87) |
| FR-4 | PostgreSQL storage | ✅ | [`src/triage/infrastructure/models.py:17`](../src/triage/infrastructure/models.py#L17) |
| FR-5 | Docker | ✅ | [`docker-compose.yaml`](../docker-compose.yaml) |
| FR-6 | Cloud IaC (AWS/GCP) | ❌ | Deployed to Render |
| FR-7 | Prometheus metrics | ✅ | [`src/shared/infrastructure/grafana.py:1`](../src/shared/infrastructure/grafana.py) |

---

## Part B: Theory Questions

### Q1: Python Concurrency (asyncio vs threading vs multiprocessing)

**When to use what:**

| Approach | Best For | Why |
|----------|----------|-----|
| **asyncio** | I/O-bound (API calls, DB queries) | Single thread, handles 1000s of concurrent requests |
| **threading** | Blocking libraries, shared state | Works with sync code, but limited by GIL |
| **multiprocessing** | CPU-bound (ML, data processing) | True parallelism across CPU cores |

**In this project:**
- FastAPI endpoints use `async def` for non-blocking database operations
- Slack notifications use thread pool for blocking HTTP calls
- Could use multiprocessing for batch embedding generation

**Key insight**: Use asyncio by default in FastAPI. Drop to threading for blocking libraries, multiprocessing only for heavy CPU work.

---

### Q2: LLM Cost Modeling (Self-Hosted vs API)

**Cost comparison for 50K tokens/day:**

| Option | Monthly Cost | Notes |
|--------|-------------|-------|
| **Groq Llama 3.3** | $0 | Free within rate limits (30 req/min) |
| **OpenAI GPT-4o-mini** | ~$56 | Pay per token |
| **Self-hosted EC2 (p4d.24xlarge)** | ~$19,000 | GPU instance cost |

**Break-even calculation:**
- Self-hosting only makes sense at ~1.5M+ tokens/day
- For this project's scale, Groq is the clear winner
- API pricing is too cheap to beat for most workloads

**Formula:**
```
Total Cost = Fixed (instance, storage) + Variable (maintenance, engineering)

Self-host if: monthly_tokens > (fixed_cost / api_cost_per_token)
```

---

### Q3: RAG Pipeline Design

**Current architecture:**
```
Query → Embed → Vector Search (Milvus) → Build Context → LLM → Response
```

**Why this design:**
- **Milvus for vector DB**: Open-source, can self-host for privacy
- **Top-K=5 retrieval**: Enough context without overwhelming the LLM
- **Structured citations**: Each source numbered for traceability
- **Low temperature (0.3)**: More factual, less creative

**Recommended improvements:**

1. **Hybrid search**: Combine semantic + keyword search for exact terms (error codes, product names)

2. **Re-ranking**: Use cross-encoder to improve relevance by 15-30%

3. **Query expansion**: Generate related queries to handle vocabulary mismatch

4. **Citation grounding**: Verify all claims are backed by sources

5. **Evaluation metrics**: Track precision, recall, NDCG for retrieval quality

---

### Q4: Measuring RAG Hallucination (Without Human Labels)

**Automated metrics framework:**

| Metric | How It Works | What It Detects |
|--------|--------------|-----------------|
| **Self-Consistency** | Generate 5 responses, check similarity | Inconsistent outputs = high risk |
| **NLI Factual Check** | Use NLI model to verify claims against context | Claims not supported by sources |
| **Citation Quality** | Check if sources are actually cited | Missing or invalid citations |
| **LLM-as-a-Judge** | Another LLM scores faithfulness/relevance | Overall response quality |
| **Metamorphic Testing** | Paraphrase query, check if answer changes | Robustness to input variations |

**Unified hallucination score:**
```
Score = 0.25 × consistency + 0.25 × NLI + 0.20 × citations + 0.20 × faithfulness + 0.10 × metamorphic

Risk level: very_low (>0.85), low (0.75-0.85), medium (0.60-0.75), high (0.45-0.60), very_high (<0.45)
```

**Implementation**: Run these checks on every RAG response in production, alert if score drops.

---

### Q5: Prompt Injection Defense

**Layered defense strategy:**

**Layer 1: Application (Code)**
- Input validation: Block patterns like "ignore instructions", "act as", role manipulation
- Delimiter isolation: Wrap user input with `###USER_INPUT_START###` markers
- Output validation: Enforce strict JSON schema, check for injected instructions
- Prompt engineering: Tell LLM to ignore instructions within user content

**Layer 2: Infrastructure**
- Rate limiting: 10 requests/minute per IP
- Request tracking: UUID per request for audit logs
- LLM rotation: Multiple providers to reduce single point of failure

**Layer 3: Policy & Operations**
- Content moderation: Multi-stage checks (patterns, embeddings, LlamaGuard)
- Monitoring: Alert on repeated injection attempts
- Response policy: Block (high risk), sanitize (medium), proceed (low)

**Example input validation:**
```python
injection_patterns = [
    r'ignore\s+(all\s+)?previous\s+instructions',
    r'act as|pretend to be|roleplay',
    r'system:\s*you\s+are',
    r'\[INST\]|<\|.*?\|>'
]
```

---

**End of Part B: Theory Questions**
