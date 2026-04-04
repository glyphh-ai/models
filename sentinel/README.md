# Sentinel Security

Discovers attack chains from disconnected security events using MITRE ATT&CK HDC encoding and Ada's DreamLoop. No correlation rules, no SIEM — Ada dreams the kill chain autonomously.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs](https://glyphh.ai/docs)** · **[Glyphh Hub](https://glyphh.ai/hub)**

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

```bash
git clone https://github.com/glyphh-ai/model-sentinel.git
cd model-sentinel

# Start the Glyphh shell (prompts login on first run)
glyphh

# Inside the shell:
# glyphh> model package                              # build .glyphh package
# glyphh> model deploy model-sentinel.glyphh         # deploy to runtime
```

### 3. Feed security events

```bash
# Inside the shell:
# glyphh> chat "User opened phishing email with macro document"
# glyphh> chat "PowerShell spawned from Word.exe executing encoded command"
# glyphh> chat "lsass.exe memory dump via procdump"
# glyphh> chat "are we under attack?"

# Interactive REPL
# glyphh> chat
```

## How It Works

Every security event is analyzed across **4 independent layers** mapped to the MITRE ATT&CK framework. Ada's DreamLoop then discovers structural connections between high-signal events — no correlation rules required.

```
Security event
  |
  v
intent.py — deterministic MITRE ATT&CK extraction (no LLM, no API calls)
  |
  +---> Tactic layer:    initial_access / execution / credential_access / lateral_movement / ...
  +---> Technique layer:  phishing / powershell / lsass_memory / remote_services / ...
  +---> Source layer:     endpoint / network / email / dns / identity / cloud / firewall
  +---> Temporal layer:   urgency score (0-10) + severity level
  |
  v
HDC encode — each layer becomes a high-dimensional vector (dim=2000)
  |
  v
Ada absorbs as thought glyphs — stored in HDC memory
  |
  v
DreamLoop discovers connections between events
  |  Localized loop (2s): catches adjacent attack pairs
  |  Deep loop (10s): finds structural similarity across all events
  |  Crystallization: mints compound primitives for attack chains
  |
  v
scorer.py — kill-chain correlation
  |
  v
Verdict: ACTIVE INTRUSION / PROBABLE ATTACK / SUSPICIOUS ACTIVITY / MONITORING
  + chain events + stage coverage + confidence score
```

## Kill Chain Detection

The scorer analyzes events for **MITRE ATT&CK kill chain progression** — a sequence of tactics that follow the natural attack ordering:

```
reconnaissance → initial_access → execution → persistence →
privilege_escalation → credential_access → discovery →
lateral_movement → collection → exfiltration → impact
```

| Verdict | Conditions | Meaning |
|---------|-----------|---------|
| **ACTIVE INTRUSION** | confidence >= 0.70, 4+ stages | Multi-stage attack in progress |
| **PROBABLE ATTACK** | confidence >= 0.50, 3+ stages | Likely coordinated attack |
| **SUSPICIOUS ACTIVITY** | confidence >= 0.30, 2+ stages | Correlated events, investigate |
| **MONITORING** | below thresholds | Normal activity |

### Demo output

```
PHASE 1: Ingesting 30 security events
  [.] [T+0.0h] DNS query for api.github.com                    → noise (1)
  [!] [T+0.2h] User jsmith opened phishing email               → initial_access (7)
  [.] [T+0.5h] Backup job completed successfully                → noise (1)
  [!] [T+0.8h] PowerShell spawned from Word.exe                 → execution (6)
  [!] [T+1.5h] lsass.exe memory dump via procdump               → credential_access (9)
  [!] [T+2.2h] PsExec lateral movement to fileserver01          → lateral_movement (8)
  [!] [T+3.8h] Large outbound HTTPS to rare external IP         → exfiltration (7)

PHASE 3: Dreaming (30s)
  [ 5s] localized=2, deep=1, insights=4
  [30s] localized=13, deep=4, insights=40, crystallization=5

SUMMARY: Kill Chain Detection
  Chain confidence:   0.94
  Stages covered:     6/14
  Verdict:            ACTIVE INTRUSION

  No SIEM rule detected this. Ada dreamed it.
```

## 4-Layer Architecture

### Tactic layer (weight: 0.35)
Classifies the MITRE ATT&CK tactic — what phase of the attack lifecycle this event represents. 13 tactics with 80+ regex patterns covering reconnaissance through impact.

| Role | Type | Description |
|------|------|-------------|
| mitre_tactic | categorical | reconnaissance / initial_access / execution / credential_access / lateral_movement / exfiltration / ... (14 values) |
| tactic_signals | bag_of_words | Tactic-bearing keyword tokens (phishing, powershell, lateral, exfiltration, etc.) |

### Technique layer (weight: 0.30)
Identifies the specific attack technique — how the tactic is implemented. 16 techniques mapped to MITRE ATT&CK technique IDs.

| Role | Type | Description |
|------|------|-------------|
| technique_id | categorical | phishing / powershell / lsass_memory / remote_services / exfil_over_web / ... (16 values) |
| technique_signals | bag_of_words | Technique-specific tool and method tokens (mimikatz, psexec, procdump, etc.) |

### Source layer (weight: 0.20)
Identifies where the event originated — endpoint telemetry, network traffic, email gateway, DNS logs, identity provider, cloud, or firewall.

| Role | Type | Description |
|------|------|-------------|
| event_source | categorical | endpoint / network / email / dns / identity / cloud / firewall / unknown |
| source_signals | bag_of_words | Source-indicating tokens (process, pid, exe, firewall, smtp, etc.) |

### Temporal layer (weight: 0.15)
Scores urgency and severity from threat keywords. Critical indicators (ransomware, mimikatz, cobalt strike) score 9-10. Noise (DNS queries, updates, backups) scores 0-2.

| Role | Type | Description |
|------|------|-------------|
| urgency | numeric (thermometer) | Threat urgency score 0-10 |
| severity | categorical | info / low / medium / high / critical |

## Ada Integration

Sentinel uses Ada's cognitive infrastructure via the `AdaCognitive` SDK class:

```python
from glyphh.memory import AdaCognitive

ada = AdaCognitive(
    system_prompt="You are Sentinel. Correlate security events.",
    recall_threshold=0.30,
    localized_interval=2.0,
    deep_interval=10.0,
)

# Feed events
ada.absorb("PowerShell spawned from Word.exe executing encoded command")
ada.absorb("lsass.exe memory dump via procdump detected")

# Start background reasoning
ada.start_dreaming()

# Query
result = ada.process("are we under attack?")
# result.gate == "DONE", result.facts has matching events
```

Ada provides:
- **Memory**: absorb, store, recall security events as HDC thought glyphs
- **DreamLoop**: background discovery of attack chains via structural similarity
- **Cognitive routing**: classify queries vs. event ingestion
- **Confidence gate**: DONE/ASK — never hallucinates false positives

## Model Structure

```
sentinel/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config, layer definitions, Ada settings
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── intent.py              # deterministic MITRE ATT&CK extraction (4 dimensions)
├── scorer.py              # kill-chain correlation scoring
├── tests/
│   ├── test_intent.py     # tactic, technique, source, urgency extraction
│   └── test_killchain.py  # chain scoring, progression, demo integration
├── demo/
│   ├── events.jsonl       # 30 curated events (25 noise + 5 kill chain)
│   └── killchain.py       # full demo script
├── data/
│   └── exemplars.jsonl    # MITRE-labeled security event exemplars
├── LICENSE                # AGPL-3.0
└── README.md
```

## Testing

```bash
cd sentinel/
PYTHONPATH=.:../../glyphh-runtime pytest tests/ -v
```

The test suite includes:
- **test_intent.py** — tactic detection (13 tactics), technique detection (16 techniques), source detection (7 sources), urgency/severity scoring, full analysis integration
- **test_killchain.py** — chain scoring, 5-stage kill chain detection, noise filtering, stage deduplication, demo event integration

### Run the demo

```bash
cd sentinel/
PYTHONPATH=.:../../glyphh-runtime python demo/killchain.py
```

## Why Not Just Use a SIEM?

| | Traditional SIEM | LLM-based | Sentinel + Ada |
|---|---|---|---|
| **Correlation** | Pre-written rules only | Hallucinates false positives | Autonomous discovery via DreamLoop |
| **Novel chains** | Misses anything not in rules | Inconsistent across sessions | Discovers unknown patterns structurally |
| **False positives** | Rule-tuning nightmare | No confidence gate | DONE/ASK gate — never guesses |
| **Background reasoning** | None — reactive only | Forgets across shifts | DreamLoop runs continuously |
| **Explainability** | Rule name + matched field | "I think this looks suspicious" | Per-layer scores, MITRE mapping, chain progression |
| **Latency** | Seconds to minutes | Seconds (API call) | Sub-millisecond (local vector ops) |
| **Persistence** | Crystallized chains become permanent | None — starts fresh | Compound primitives = learned signatures |

The key insight: **Splunk needs rules. Ada dreams.** When Ada's DreamLoop crystallizes a compound primitive from correlated attack events, that chain becomes a permanent signature — future events matching the same pattern are detected instantly, without anyone writing a rule.
