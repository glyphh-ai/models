# Glyphh Models

Open source HDC models for the [Glyphh](https://glyphh.ai) runtime — compiled
cognition: deterministic routers, classifiers, and firewalls built from
exemplar data into binary `.glyphh` packages. Zero tokens, no LLM in the path.

## Building & installing (the `glyphh` CLI)

The runtime IS the toolchain — `pip install glyphh` gives you everything:

```bash
pip install glyphh

# build a model directory (manifest.yaml + config.yaml + data/*.jsonl) into a .glyphh
glyphh models build bfcl/

# install onto your runtime — a path, an https URL, or a bare library name
glyphh models install bfcl/bfcl-function-caller.glyphh
glyphh models install bfcl                 # resolves via this repo's library.json

# use it
glyphh models list
glyphh models info bfcl-function-caller
glyphh models route bfcl-function-caller "Move final_report.pdf to the temp directory"
glyphh models remove bfcl-function-caller

# remote runtimes: every verb honors -c/--context (routes through /mcp)
glyphh models list -c prod
```

Builds are **declarative**: the runtime encodes `data/*.jsonl` against the
layer schema in `config.yaml` (plus an optional `build:` field-mapping
section) with its own HDC encoder. Per-model `encoder.py` files in this repo
target a retired SDK and are not executed by v1 builds. The format and
lifecycle are specified in the runtime repo's `docs/models.md`.

**`library.json`** at the repo root is the discovery manifest
(`glyphh models install <name>` reads it) and records each model's honest
state: `buildable` · `source` (needs a `build:` mapping) · `incomplete` ·
`placeholder`. `bfcl` is the golden path — buildable end-to-end, with a
prebuilt package attached to a GitHub release.

---

## Repository Structure

Each model lives as a git submodule under this repo. The structure is the same
for every model:

```
model-name/
├── manifest.yaml        # identity, version, tags
├── config.yaml          # encoder config, layers, similarity thresholds,
│                        #   and the optional build:/query: mapping sections
├── encoder.py           # legacy (retired SDK) — not executed by v1 builds
├── data/
│   └── *.jsonl          # training exemplars
├── tests/
│   ├── conftest.py
│   ├── test_encoding.py
│   └── test_similarity.py
└── benchmark/           # evaluation queries and results
```

## Contributing a Model

1. Create a new model repo under `glyphh-ai/model-<name>`
2. Follow the structure above — make sure `glyphh models build` succeeds
3. Open a PR adding it as a submodule here and an entry to `library.json`

## License

MIT
