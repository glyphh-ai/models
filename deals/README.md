# Glyphh Deals — Deal Intelligence

Encodes sales deals as searchable HDC glyphs with continuous pipeline metrics. Supports semantic search ("find stalled enterprise deals"), pattern matching, and real-time CRM sync via webhooks through the listener endpoint.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs →](https://glyphh.ai/docs)** · **[Glyphh Hub →](https://glyphh.ai/hub)**

---

## Getting Started

### 1. Install the Glyphh CLI

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install Glyphh (includes runtime)
pip install glyphh
```

### 2. Install the model

```bash
# Start the Glyphh shell (prompts login on first run)
glyphh

# Inside the shell:
# glyphh> hub install model-deals
```

### 3. Query the model

```bash
# Text queries (intent extraction → HDC encode → cosine search)
# glyphh> chat "find stalled enterprise deals"
# glyphh> chat "which deals are about to close"
# glyphh> chat "show me deals at risk with budget objections"
# glyphh> chat "deals that have gone dark"
```

## How It Works

Deals follows the same bifurcated architecture as all Glyphh models: **HDC handles deterministic parsing, LLM handles generation**.

```
QUERY INPUT (NL or deal record)
    ↓
Intent Extraction (intent.py)
    ↓ keywords, synonyms, phrases
Encoder (4-layer HDC binding)
    ↓ identity + profile + strategy + metrics
Bundle → 2,000-dim bipolar vector
    ↓
Searchable, comparable, updatable
```

**Two matching paths, one config:**
- **NL queries** (text, no metrics) → match on strategy layer
- **Deal records** (metrics + text) → match on all layers including metrics
- **Listener updates** (CRM webhook data) → metrics layer drives similarity

## Encoded Layers

| Layer | Weight | Roles | Encoding |
|-------|--------|-------|----------|
| **identity** | 0.10 | deal_id (key_part), account_name | symbolic |
| **profile** | 0.20 | industry, company_size, region, source | symbolic |
| **strategy** | 0.30 | description, keywords, stage, deal_type | bag_of_words, symbolic |
| **metrics** | 0.40 | deal_value, days_in_stage, activity_count, email_response_rate, meetings_held | numeric (thermometer) |

**Three encoding types:**
- **Symbolic** — Exact value match. Same value = identical vector.
- **Bag of words** — Split into words, encode each, bundle. Shared words = shared signal.
- **Numeric (thermometer)** — Binned values with adjacent similarity. Nearby values → similar vectors.

## Pipeline Metrics

| Role | Bin Width | Range | What It Captures |
|------|-----------|-------|------------------|
| deal_value | $25,000 | $0–$500K | Deal size (enterprise vs SMB) |
| days_in_stage | 10 days | 0–120 | Stall detection |
| activity_count | 5 | 0–50 | Engagement velocity |
| email_response_rate | 15% | 0–100% | Prospect responsiveness |
| meetings_held | 3 | 0–20 | Depth of engagement |

## Real-Time Updates via Webhooks

Pipeline metrics are updated in real-time through CRM webhooks → runtime listener endpoint:

```
CRM Event (stage change, email reply, meeting booked)
    ↓
POST /{org_id}/deals/listener
    Body: {
        "records": [{
            "name": "deal_acme_enterprise",
            "attributes": {
                "deal_id": "deal_acme_enterprise",
                "account_name": "Acme Corp",
                "industry": "technology",
                "company_size": "enterprise",
                "region": "us-west",
                "source": "inbound",
                "stage": "negotiation",
                "deal_type": "new_business",
                "deal_value": 250000,
                "days_in_stage": 8,
                "activity_count": 35,
                "email_response_rate": 85,
                "meetings_held": 12
            }
        }]
    }
    ↓
Runtime encodes → stores glyph with temporal timestamp
```

Each update creates a new glyph version (temporal tracking via `deal_id` key_part). Temporal deltas capture deal velocity — how fast metrics change between updates.

## Model Structure

```
deals/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # 4-layer encoder config, thresholds
├── encoder.py             # ENCODER_CONFIG + encode_query + entry_to_record
├── intent.py              # deal NL intent extraction (synonyms, phrases, stemming)
├── data/
│   └── exemplars.jsonl    # seed deal patterns (22 exemplars)
├── tests/
│   ├── conftest.py        # shared fixtures (encoder, deal glyphs)
│   ├── test_encoding.py   # config validation, encoding pipeline
│   ├── test_similarity.py # outcome matching, metrics similarity
│   ├── test_nl_queries.py # NL query → outcome matching
│   ├── test_queries.py    # encode_query unit tests
│   ├── test_intent.py     # keyword extraction, synonym expansion
│   ├── test_metrics.py    # numeric encoding edge cases
│   └── test_temporal.py   # temporal tracking, velocity
├── build.py               # build/package script
├── requirements.txt       # no special deps (pure HDC)
└── README.md
```

## Testing

```bash
# Run from the deals/ directory
PYTHONPATH="../../glyphh-runtime:.:$PYTHONPATH" python -m pytest tests/ -v

# Or specific test files
PYTHONPATH="../../glyphh-runtime:.:$PYTHONPATH" python -m pytest tests/test_encoding.py -v
PYTHONPATH="../../glyphh-runtime:.:$PYTHONPATH" python -m pytest tests/test_similarity.py -v
PYTHONPATH="../../glyphh-runtime:.:$PYTHONPATH" python -m pytest tests/test_nl_queries.py -v
PYTHONPATH="../../glyphh-runtime:.:$PYTHONPATH" python -m pytest tests/test_intent.py -v
```

The test suite runs entirely on synthetic data — no external dependencies needed.

## Data Format

Exemplars in `data/exemplars.jsonl` are pre-built deal patterns:

```json
{
    "question": "enterprise deal progressing quickly with strong champion engagement",
    "outcome": "won",
    "risk_level": "low",
    "deal_driver": "champion_engagement",
    "industry": "technology",
    "company_size": "enterprise",
    "source": "inbound",
    "stage": "negotiation",
    "deal_type": "new_business",
    "deal_value": 250000,
    "days_in_stage": 8,
    "activity_count": 35,
    "email_response_rate": 85,
    "meetings_held": 12,
    "keywords": ["enterprise", "champion", "engaged", "progressing", "strong"]
}
```

## Architecture — Agent + Model Closed Loop

The deals model is an **observation and retrieval layer** — it encodes what happened and lets you search/compare it. It never creates, sends, or modifies CRM records. Your LLM agent handles actions; the model handles memory and analytics.

```
┌──────────────────────────────────────────────────────────┐
│                    LLM AGENT (Claude)                    │
│                                                          │
│  Manages pipeline, writes follow-ups, updates stages,    │
│  creates tasks, adjusts forecasts                        │
│                                                          │
│  Queries the model via MCP to inform decisions:          │
│  "which deals are stalling?"                             │
│  "find deals similar to our best won deals"              │
│  "show at-risk enterprise deals"                         │
└────────────┬─────────────────────────┬───────────────────┘
             │ MCP nl_query            │ CRM actions
             ▼                         ▼
┌────────────────────────┐   ┌─────────────────────────────┐
│   GLYPHH RUNTIME       │   │   CRM SYSTEM                │
│                        │   │                             │
│  Encodes query → HDC   │   │  HubSpot / Salesforce       │
│  Cosine search pgvector│   │  Stage changes              │
│  Returns ranked results│   │  Email tracking             │
│  with similarity scores│   │  Meeting scheduling         │
└────────────────────────┘   └──────────┬──────────────────┘
             ▲                          │ Events fire
             │                          │ (stage change,
             │                          │  email reply,
             │                          │  meeting booked)
             │                          ▼
             │               ┌──────────────────────────────┐
             │               │   WEBHOOK WORKFLOWS          │
             │               │                              │
             │               │  CRM webhook → normalize     │
             │               │  Email tracking → normalize  │
             │               │  Calendar API → normalize    │
             └───────────────│                              │
              POST /listener │  → POST /{org}/deals/        │
                             │    listener                  │
                             └──────────────────────────────┘
```

**The closed loop:**
1. Agent queries the model → "which deals need attention?"
2. Model returns ranked deals with similarity scores
3. Agent acts on the insight → sends follow-up, escalates, adjusts forecast
4. CRM events fire (stage change, email reply, meeting booked)
5. Webhooks capture events → normalize → POST to listener
6. Model updates glyph pipeline metrics
7. Agent queries again → loop continues

## MCP Integration

LLM agents query the model via the runtime's MCP tools:

```python
# Available MCP tools:
# 1. nl_query — Natural language search
# 2. gql_query — Structured GQL search

# Example: Agent asks which deals are at risk
POST /{org_id}/deals/mcp
{
    "tool": "nl_query",
    "arguments": {
        "query": "find stalled enterprise deals with no activity"
    }
}

# Response:
{
    "content": [...],
    "result": {
        "matches": [
            {"deal_id": "deal_gamma_stalled", "score": 0.85},
            {"deal_id": "deal_theta_budget", "score": 0.71}
        ]
    },
    "confidence": 0.85,
    "query_time_ms": 3.8
}
```

## Data Pipeline — Event Sources

### CRM Webhooks (HubSpot / Salesforce)

| CRM Event | Maps To |
|-----------|---------|
| Deal stage change | stage, days_in_stage (reset) |
| Email sent/replied | activity_count++, email_response_rate |
| Meeting booked/held | meetings_held++, activity_count++ |
| Deal value change | deal_value |
| Deal won/lost | outcome (metadata only) |

### Webhook Workflow Structure

```
Workflow 1: Deal Stage Changes
  Trigger: CRM webhook (deal.propertyChange)
  → Extract deal_id, new stage, deal value
  → Calculate days_in_stage
  → POST /{org_id}/deals/listener

Workflow 2: Email Engagement
  Trigger: CRM webhook (email.replied, email.opened)
  → Aggregate by deal_id
  → Calculate response rate
  → POST /{org_id}/deals/listener

Workflow 3: Meeting Activity
  Trigger: Calendar webhook (meeting.completed)
  → Map to deal_id via CRM association
  → Increment meetings_held, activity_count
  → POST /{org_id}/deals/listener
```

## Use Cases

### Agent-Driven Pipeline Review
```
Agent: "Which deals need immediate attention?"
→ MCP nl_query: "stalled at-risk deals with declining engagement"
→ Model returns: deal_gamma_stalled (0.85), deal_theta_budget (0.71)
→ Agent drafts follow-up for each and alerts rep
```

### Find Similar Winning Deals
```bash
# glyphh> chat "find deals similar to our best closed-won deals"
```

### Identify At-Risk Deals
```bash
# glyphh> chat "show me deals that have gone dark"
```

### Pipeline Forecasting
```bash
# glyphh> chat "which enterprise deals are likely to close this quarter"
```

### Competitive Intelligence
```bash
# glyphh> chat "deals at risk due to competitor evaluation"
```

### Deal Velocity Analysis
```bash
# glyphh> chat "compare fast-closing vs slow deals"
```
