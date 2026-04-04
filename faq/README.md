# FAQ Helpdesk

Domain-agnostic FAQ model for helpdesk and knowledge base agents. Matches user questions to answers via HDC semantic similarity. Replace the sample data with your own Q&A pairs — no retraining, no embeddings API.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs →](https://glyphh.ai/docs)** · **[Glyphh Hub →](https://glyphh.ai/hub)**

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
git clone https://github.com/glyphh-ai/model-faq.git
cd model-faq

# Start the Glyphh shell (prompts login on first run)
glyphh

# Inside the shell:
# glyphh> dev start              # starts local dev server
```

### 3. Deploy the model

```bash
glyphh
# glyphh> model package                          # build .glyphh package
# glyphh> model deploy model-faq.glyphh          # deploy to runtime
```

### 4. Query the model

```bash
glyphh
# glyphh> chat "I forgot my password"
# glyphh> chat "where is my order"
# glyphh> chat "can I get a refund"

# Interactive REPL
# glyphh> chat
```

The server runs at `http://localhost:8002`. Use the **up-arrow** in `glyphh chat` to replay previous queries — history persists in `~/.glyphh/chat_history`.

## How It Works

User question → IntentExtractor (keywords + category) → HDC encode (10,000d) → cosine similarity against stored FAQ entries → best matching answer + category + confidence.

Primary matching signal: **bag-of-words on the question text**. Shared words between a user question ("forgot my password") and an FAQ entry ("how do I reset my password") drive similarity — no LLM, no embeddings API call.

### Response States

The runtime returns one of five states with every query:

| State | Meaning |
|-------|---------|
| `DONE` | Match found — confidence + answer returned |
| `ASK` | Query too vague or matches multiple options equally |
| `BLOCKED` | Action requires a prerequisite |
| `AUTH_REQUIRED` | Action requires authentication |
| `ERROR` | Query failed |

When `ASK` is returned, the CLI shows the clarifying question and any disambiguation options. Models can export an `assess_query()` function to define their own slot-completeness rules — see the [Glyphh docs](https://glyphh.ai/docs?page=models-development) for the full contract.

## Model Structure

```
faq/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config (similarity threshold, min_gap)
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── build.py               # package model into .glyphh file
├── data/
│   └── faq.jsonl          # FAQ entries (training data)
├── tests/
│   ├── test-concepts.json # 10 sample questions for testing (no answers)
│   ├── conftest.py        # shared fixtures
│   ├── test_encoding.py   # config validation, BoW encoding, entry_to_record
│   ├── test_similarity.py # end-to-end match accuracy tests
│   └── test_queries.py    # encode_query unit tests
└── benchmark/
    ├── run.py             # benchmark runner (accuracy, latency, category breakdown)
    └── queries.json       # 32 queries across 4 difficulty categories
```

## Encoder

Two-layer HDC with 10,000 dimensions:

| Layer | Weight | Segment | Role | Encoding | Weight |
|-------|--------|---------|------|----------|--------|
| semantic | 0.7 | identity | question_id | symbolic (key_part) | 0.1 |
| | | | category | lexicon (7 values) | 0.6 |
| | | content | question | bag-of-words | 1.0 |
| | | | answer | bag-of-words | 0.4 |
| context | 0.3 | tags | keywords | bag-of-words | 0.8 |

Category lexicons: `account`, `billing`, `product`, `shipping`, `returns`, `technical`, `general`

## FAQ Data Format

Replace `data/faq.jsonl` with your own Q&A pairs. Each line:

```json
{
  "question_id": "acct_reset_password",
  "category": "account",
  "question": "how do I reset my password",
  "answer": "Go to Settings > Security > Reset Password. Enter your email to receive a reset link.",
  "keywords": ["password", "reset", "forgot", "locked out"]
}
```

`question_id` and `category` are auto-generated if omitted — the encoder infers category from the question text via IntentExtractor + keyword signals.

**keywords** should include synonyms, abbreviations, and related terms that users might phrase the question with but that aren't in the question itself. Keywords should also cover phrasing variations that don't appear in the question — e.g. `"received damaged"` for an entry titled "what if my item arrived broken".

## Testing

```bash
# Inside the glyphh shell:
# glyphh> model test .
# glyphh> model test . -v
# glyphh> model test . -k similarity

# Directly with pytest
cd faq/
pytest tests/ -v
```

36 tests across encoding, similarity, and query inference. The suite encodes 10 raw user questions (no answers) and verifies the model returns the correct category and `question_id` for each.

## Benchmark

```bash
# Run accuracy benchmark (no API key needed)
python benchmark/run.py

# Adjust confidence threshold
python benchmark/run.py --threshold 0.45

# Save raw results
python benchmark/run.py --output benchmark/results/
```

32 queries across 4 difficulty categories:

| Category | n | Description |
|----------|---|-------------|
| `clear` | 10 | Unambiguous single-category questions |
| `near_collision` | 10 | Questions that overlap multiple categories |
| `adversarial` | 7 | Informal phrasing, abbreviations, unusual wording |
| `open_set` | 5 | Out-of-scope questions (should abstain) |
