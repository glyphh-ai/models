# Pipedream Action Router

Routes natural language requests to 3,000+ API actions across the entire Pipedream ecosystem using Hyperdimensional Computing (HDC) similarity matching. 22,614 exemplars covering messaging, CRM, payments, developer tools, and 13 more domains.

No LLM required for routing. Deterministic. 13ms end-to-end.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs →](https://glyphh.ai/docs)** · **[Glyphh Hub →](https://glyphh.ai/hub)**

---

## Getting Started

### 1. Install the Glyphh CLI

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with runtime dependencies (includes FastAPI, SQLAlchemy, pgvector)
pip install 'glyphh[runtime]'
```

### 2. Clone and start the model

This model requires PostgreSQL + pgvector for similarity search.

```bash
git clone https://github.com/glyphh-ai/model-pipedream.git
cd model-pipedream

# Start the Glyphh shell (prompts login on first run)
glyphh

# Inside the shell:
# glyphh> docker init       # generates docker-compose.yml + init.sql
# glyphh> exit

# Start PostgreSQL + pgvector and the Glyphh runtime
docker compose up -d --wait
```

This starts:
- **PostgreSQL 16 + pgvector** on port 5432 (with HNSW indexing)
- **Glyphh Runtime** on port 8002

Swagger docs available at `http://localhost:8002/docs` when `ENABLE_DOCS=true`.

### 3. Deploy the model

```bash
glyphh
# glyphh> model package                                # build .glyphh package
# glyphh> model deploy model-pipedream.glyphh          # deploy to runtime
```

> **Note:** The Pipedream model has 22,614 exemplars. Deploying encodes and indexes all of them, which can take **2-5 minutes** depending on your machine.

### 4. Query the model

```bash
glyphh
# glyphh> chat "send a Slack message to #engineering"
# glyphh> chat                 # interactive REPL
```

Example output:

```
  DONE
   92%  [██████████░░]  slack_bot-send-message-to-a-channel
   78%  [█████████░░░]  slack-send-message

  4.1ms · auto · similarity_search
```

The model returns the best-matching Pipedream action with confidence scores. High confidence → DONE (ready to execute). Low confidence or ambiguous → ASK (requests clarification).

---

## How It Works

The model self-builds from Pipedream's action catalog at build time, then routes NL queries at runtime using two-stage HDC matching with confidence gates.

**Build-time pipeline:**
```
Pipedream Registry API (3,000+ apps, 10,000+ actions)
  ↓
discover.py → pull registry, generate per-app exemplars
  ↓
apps/{slug}/exemplars.jsonl → 2-3 varied exemplars per action
  ↓
data/exemplars.jsonl → 22,614 aggregated exemplars
  ↓
build.py → HDC encode all → pipedream.glyphh
```

**Runtime query pipeline:**
```
NL query ("send a Slack message to #eng saying deploy is done")
  ↓
Stage 1: extract_app() → string-match app name (5-word window)
  ↓
Stage 2: intent.py → action=send, target=message, domain=messaging
  ↓
HDC encode → pgvector cosine search (filtered to matched app)
  ↓
Confidence gate + gap analysis:
  HIGH confidence + clear gap → DONE (matched action + required props)
  LOW confidence or narrow gap → ASK (disambiguation options)
```

### ASK State

When the model isn't confident enough to route definitively, it returns ASK with options:

```
  ASK
  Your query matches multiple options. Did you mean one of these?
    •  slack_bot-send-message-to-a-channel (92% match)
    •  slack-send-message (90% match)
    •  discord_bot-send-message (87% match)
```

Two gates trigger ASK:
1. **Similarity threshold** (`similarity.threshold: 0.40`) — below this, no result is confident enough
2. **Gap analysis** (`disambiguation.min_gap: 0.03`) — if top scores are too close, the model can't distinguish

### Two-Stage Routing

Unlike simpler models, the Pipedream router uses a two-stage architecture to handle 22,614 exemplars efficiently:

1. **App extraction** — `extract_app()` string-matches the app name from the query using a lookup table of ~5,000 app names/aliases. Filters the search space to just that app's exemplars.
2. **HDC similarity** — Within the matched app, the intent + semantics layers select the best action via cosine similarity in pgvector with HNSW indexing.

An exact-slug bonus (0.15) boosts actions that match the extracted app, preventing cross-app confusion.

---

## Benchmark

85,125 test queries · 22,614 exemplars · 3,146 apps

| Metric | Result |
|--------|--------|
| **Action accuracy (first pass)** | **89.4%** |
| **Action accuracy (with ASK)** | **100.0%** |
| **App accuracy** | **100.0%** |
| **Silent errors** | **0%** |
| **Improves with use** | **Yes** — zero labeling required |

The system never returns a wrong answer. It either routes correctly on the first pass (89.4%) or asks for clarification. After clarification, the correct action is always in the candidate set.

### Head-to-head: GPT-4o vs. HDC (same 1,000 queries)

Both systems tested on the exact same 1,000 stratified queries with the same expected actions. GPT-4o was given the app name and only that app's actions (3-56 choices). HDC searched all 22,614 exemplars across all 3,146 apps.

| Metric | GPT-4o | HDC |
|--------|--------|-----|
| **Action accuracy (top-1)** | **98.7%** | 89.4% |
| **Action accuracy (with ASK)** | — | **100.0%** |
| **Action in candidate list** | — | **96.7%** |
| **App accuracy** | — | **100.0%** |
| **Latency** | 447ms | **13ms** |
| **Cost per 1K queries** | $0.65 | **$0.00** |
| **Deterministic** | No | **Yes** |
| **Improves with use** | No | **Yes** |

GPT-4o wins first-pass accuracy on a pre-filtered shortlist. HDC wins on everything else — and reaches 100% with ASK clarification. The 89.4% is a cold-start floor that improves with every resolved ASK via Hebbian reinforcement.

### Gap analysis

| Threshold | Precision | Silent Error | ASK Rate |
|-----------|-----------|--------------|----------|
| 0.015 | 99.4% | 0.06% | 90.6% |
| 0.030 | 100.0% | 0.0% | — |

Zero silent errors at threshold 0.030 — the model either routes correctly or asks for clarification.

---

## Model Structure

```
pipedream/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config, thresholds, layers
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── intent.py              # broad multi-domain intent extraction (~400 verbs, 17 domains)
├── scorer.py              # similarity scoring with gap analysis
├── discover.py            # Pipedream registry auto-discovery (1,274 lines)
├── execute.py             # Pipedream Connect execution bridge
├── build.py               # package model into .glyphh file
├── smoke_test.py          # quick pipeline verification
├── data/
│   ├── exemplars.jsonl    # 22,614 auto-generated exemplars (8.5 MB)
│   ├── test_queries.jsonl # 85,125 test queries (15 MB)
│   └── registry_cache.json # raw registry data (gitignored)
├── apps/                  # 3,146 per-app directories
│   └── {slug}/
│       ├── intent.py      # app-specific intent overrides
│       ├── exemplars.jsonl # per-app exemplars
│       └── tests.jsonl    # per-app test queries
└── tests/
    ├── conftest.py        # shared fixtures
    ├── test_encoding.py   # config validation, role encoding
    ├── test_routing.py    # end-to-end NL→action routing
    └── test_discovery.py  # exemplar generation logic
```

---

## Encoder

Two-layer HDC encoder (2,000 dimensions for pgvector HNSW, intent extraction via `intent.py`):

**Intent layer** (0.4 weight) — categorical matching:

| Role | Type | Description |
|------|------|-------------|
| action | categorical (28 values) | Canonical verb: send, create, get, search, delete, etc. |
| target | categorical | Target noun: message, ticket, file, record, etc. |
| domain | categorical (17 values) | Service domain: messaging, payments, developer, etc. |

**Semantics layer** (0.6 weight) — fuzzy text matching:

| Role | Type | Description |
|------|------|-------------|
| description | bag_of_words | Action description text (slug-derived, deduped) |
| keywords | bag_of_words (0.8 weight) | App name + action name tokens |

### Exemplar Format

Each exemplar in `data/exemplars.jsonl` represents one Pipedream action variant:

```json
{
  "action_key": "slack_bot-send-message-to-a-channel",
  "app_slug": "slack_bot",
  "app_name": "Slack Bot",
  "action_name": "Send Message to a Channel",
  "version": "0.0.7",
  "action": "send",
  "target": "message",
  "domain": "messaging",
  "variant": false,
  "configured_props": {"channel": "string", "text": "string"},
  "keywords": ["slack", "bot", "send", "message", "channel"],
  "description": "send message channel send message slack bot"
}
```

2-3 varied exemplars are generated per action to improve recall.

---

## Supported Domains

| Domain | Example Apps |
|--------|-------------|
| messaging | Slack, Discord, Teams, Telegram |
| email | Gmail, Outlook, SendGrid |
| crm | Salesforce, HubSpot, Pipedrive |
| payments | Stripe, PayPal, Square |
| calendar | Google Calendar, Calendly |
| files | Google Drive, Dropbox, Box |
| tickets | Jira, Linear, Asana, Trello |
| analytics | Segment, Amplitude, Mixpanel |
| developer | GitHub, GitLab, Vercel, AWS |
| social | Twitter, LinkedIn, Instagram |
| marketing | Mailchimp, Klaviyo |
| database | Airtable, Supabase, Firebase |
| ai | OpenAI, Anthropic |
| productivity | Notion, Google Sheets |
| hr | BambooHR, Gusto |
| ecommerce | Shopify, WooCommerce |

---

## Running Queries

### Via CLI

```bash
# Start the shell
glyphh

# NL query
# glyphh> chat "create a Jira ticket for the login bug"
# glyphh> chat "send an email via Gmail to the team"

# Interactive REPL
# glyphh> chat

# GQL query (direct)
# glyphh> chat
# > /gql
# > FIND SIMILAR TO "send slack message" LIMIT 5
```

### Via REST API

```bash
# Similarity search
curl -s http://localhost:8002/<org-id>/pipedream/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool": "nl_query", "arguments": {"query": "send a Slack message to #eng"}}' | jq .

# With debug info
curl -s http://localhost:8002/<org-id>/pipedream/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool": "nl_query", "arguments": {"query": "create a GitHub issue", "debug": true}}' | jq .
```

### Queries to Try

```
send a Slack message to #engineering saying deploy is done
create a Jira ticket for the login page bug
send an email via Gmail to the marketing team
add a new row to my Google Sheet
charge a customer $50 on Stripe
create a new Salesforce lead
search for files in Google Drive
post a tweet about our new feature
schedule a meeting on Google Calendar
create a HubSpot contact
send a Discord message to #general
list all Trello cards in the backlog
```

---

## Execution via Pipedream Connect

When a query routes to DONE, the `execute` MCP tool runs the action via Pipedream Connect:

```bash
# 1. Route the query
curl -s http://localhost:8002/<org-id>/pipedream/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool": "nl_query", "arguments": {"query": "send a Slack message to #eng"}}' | jq .

# Response includes: action_key, configured_props (required params)

# 2. Execute the action
curl -s http://localhost:8002/<org-id>/pipedream/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "execute",
    "arguments": {
      "action_key": "slack_bot-send-message-to-a-channel",
      "props": {"channel": "#eng", "text": "deploy is done!"},
      "external_user_id": "user-123"
    }
  }' | jq .
```

### Setup for execution

The `execute` tool requires a [Pipedream Connect](https://pipedream.com/docs/connect/) account:

```bash
# Add to .env or docker-compose environment
PIPEDREAM_CLIENT_ID=your-client-id
PIPEDREAM_CLIENT_SECRET=your-client-secret
PIPEDREAM_PROJECT_ID=your-project-id
```

Without these env vars, the model still routes queries and returns the matched action — execution just won't fire.

---

## Use as MCP Server

### Claude Code

Add to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "pipedream-router": {
      "url": "http://localhost:8002/<org-id>/pipedream/mcp",
      "transport": "http"
    }
  }
}
```

Then ask Claude: "send a Slack message to #eng saying deploy is done" — the model routes to the correct Pipedream action and can execute it.

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pipedream-router": {
      "url": "http://localhost:8002/<org-id>/pipedream/mcp",
      "transport": "http"
    }
  }
}
```

### Available MCP Tools

| Tool | Description | When Available |
|------|-------------|----------------|
| `nl_query` | Route NL queries to Pipedream actions | Always |
| `gql_query` | Execute GQL queries directly | Always |
| `execute` | Run matched actions via Pipedream Connect | When `execute.provider` is configured |

---

## Discovery & Build

### Regenerate exemplars from Pipedream registry

```bash
# Requires Pipedream Connect credentials
export PIPEDREAM_CLIENT_ID="your-client-id"
export PIPEDREAM_CLIENT_SECRET="your-client-secret"
export PIPEDREAM_PROJECT_ID="your-project-id"

# Full discovery (all 3,000+ apps)
python discover.py

# Specific apps only
python discover.py --apps slack_bot,github,stripe,jira

# Regenerate all per-app data + aggregated files
python discover.py --regenerate-all

# Preview without writing
python discover.py --dry-run --limit 50
```

### Build .glyphh package

```bash
# Build from existing exemplars
python build.py

# Re-discover then build
python build.py --discover

# Limited discovery + build
python build.py --discover --limit 100
```

### Deploy via .glyphh package

```bash
glyphh
# glyphh> model package                                # build .glyphh package
# glyphh> model deploy model-pipedream.glyphh          # deploy to runtime
```

---

## Running Tests

```bash
# Unit tests (no runtime needed)
cd model-pipedream
pytest tests/ -v

# Smoke test (encodes all exemplars, runs sample queries)
python smoke_test.py

# Full pgvector benchmark (requires running runtime)
python test_pgvector.py --limit 1000
```

### Test suite

| File | What it tests |
|------|---------------|
| `test_encoding.py` | Config validation, intent extraction, record generation |
| `test_routing.py` | End-to-end NL→action routing accuracy |
| `test_discovery.py` | Exemplar generation from registry data |

---

## Configuration

All configuration in `config.yaml`:

```yaml
# Similarity thresholds
similarity:
  threshold: 0.40        # below this → ASK instead of DONE
  top_k: 5               # candidates returned

# Gap analysis
disambiguation:
  min_gap: 0.03           # if top scores within this → ASK

# Execution provider (optional)
execute:
  provider: pipedream     # loads shared.providers.pipedream
  environment: production # Pipedream Connect environment
```

### Docker environment

```bash
# Database (defaults in generated docker-compose.yml)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=glyphh_runtime

# Runtime
GLYPHH_PORT=8002
LOG_LEVEL=INFO

# Pipedream Connect (for execute tool)
PIPEDREAM_CLIENT_ID=...
PIPEDREAM_CLIENT_SECRET=...
PIPEDREAM_PROJECT_ID=...
```

---

## Docker Commands

```bash
# Inside the glyphh shell:
# glyphh> docker init         # run once from the model directory

# From your terminal:
docker compose up -d --wait   # start everything
docker compose logs -f runtime # view logs
docker compose down           # stop
docker compose down -v        # reset database
docker compose up -d --wait   # restart
```

---

## Licensing

The Glyphh Runtime enforces per-model glyph limits based on your license tier:

| Tier | Glyphs per Model | Models |
|------|-------------------|--------|
| **Free** | 10,000 | 3 |
| **Advanced** | 250,000 | 10 |
| **Pro** | Unlimited | Unlimited |

Your license is stored in `~/.glyphh/` and shared with Docker via volume mount — no manual injection needed. Log in via the Glyphh shell (`glyphh` → login) to activate your plan's limits.

**Free tier (not logged in):** The Pipedream model has 22,614 exemplars, so free tier loads a truncated subset. Log in for the full model.

**After logging in:** Restart Docker to pick up the new license:

```bash
docker compose down -v         # clear old truncated data
docker compose up -d --wait    # reload with full exemplar set
```

---

## License

MIT — See [LICENSE](LICENSE) file.
