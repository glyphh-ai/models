# Glyphh Iris — Structured Visual Encoder

Decomposes images into searchable, manipulable HDC glyphs by binding specialized feature extractors (face, pose, depth, OCR, lighting, color, composition, objects) into a single 2,000-dimensional bipolar vector. Deterministic: same image always produces the same glyph.

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
# glyphh> hub install model-iris
```

### 3. Install CV dependencies (optional)

```bash
pip install -r requirements.txt
```

All CV dependencies are optional — extractors degrade gracefully if a dependency is missing. For a minimal install, only `pillow` and `scikit-learn` are needed. The lighting and composition extractors are pure numpy and always work.

| Dependency | Extractor | What it provides |
|-----------|-----------|-----------------|
| insightface | Face | 512-dim face embedding (ArcFace), expression, age |
| mediapipe | Pose | 33 body keypoints, head orientation |
| torch + MiDaS | Depth | Monocular depth map (pooled to 8x8) |
| paddleocr | OCR | Text regions with bounding boxes |
| ultralytics | Objects | YOLO v8 object detection |
| scikit-learn | Color | Dominant color palette via K-means |
| pillow | All | Image I/O |

### 4. Encode and query images

Image encoding is done **offline** — you process a directory of images into pre-extracted features, then deploy those to the runtime. No heavy CV models run at query time.

#### Ingest images (offline)

```bash
# Extract features from a directory of images → data/exemplars.jsonl
python ingest.py /path/to/your/images

# Lightweight mode — no heavy CV models needed (pure numpy)
python ingest.py /path/to/images --extractors color,lighting,composition

# Parallel processing for large datasets
python ingest.py /path/to/images --workers 4

# Test with a few images first
python ingest.py /path/to/images --limit 10 --verbose
```

**Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`

This writes `data/exemplars.jsonl` — one line per image with all extracted features. Extractors degrade gracefully: if a CV dependency is missing (e.g., InsightFace), that extractor returns default values and the rest still run.

#### Deploy to runtime

The model auto-deploys when the Docker runtime starts (via volume mount). To manually redeploy after updating exemplars:

```bash
# From the Glyphh shell:
# glyphh> model deploy .

# Or via API:
curl -X POST http://localhost:8002/<org-id>/iris/model/deploy \
  -F "model_path=."
```

The runtime loads `data/exemplars.jsonl`, HDC-encodes each entry, and stores the glyphs in pgvector.

#### Query via text

```bash
# Text queries (intent extraction → HDC encode → cosine search)
# glyphh> chat "find images with soft side lighting"
# glyphh> chat "show me outdoor portraits"
```

#### Query via file upload

The runtime exposes a generic `POST /query/file` endpoint that accepts any file via multipart form data. For Iris, this means you can upload an image and get similarity search results against your deployed exemplars.

```bash
# Upload an image → extract features → find similar images
curl -X POST http://localhost:8002/<org-id>/model-iris/query/file \
  -F "file=@photo.jpg" \
  -F "top_k=5"

# The runtime saves the file to a temp path, passes it to encode_query(),
# which detects the image extension and runs the full extraction pipeline.
# Returns a FactTree with similarity-ranked results.
```

From Python:

```python
import requests

with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8002/<org-id>/model-iris/query/file",
        files={"file": ("photo.jpg", f, "image/jpeg")},
        data={"top_k": 5},
    )
    results = resp.json()
```

This endpoint is **generic to the runtime** — not Iris-specific. Any model can accept file uploads; the model's `encode_query()` decides what to do with the file path. Iris detects image extensions and runs CV extractors. A different model could accept PDFs, audio, or CSVs.

#### Python API

```python
from iris.iris import Iris

iris = Iris()

# Encode an image
glyph = iris.encode("photo.jpg")

# Get structured spec
spec = glyph.to_json()
print(spec["lighting"])     # {"direction": "side_left", "quality": "soft", ...}
print(spec["identity"])     # {"expression": "happy", "age_group": "thirties", ...}

# Get prompt fragment for image generation
print(glyph.to_prompt())
# → "thirties person, happy expression, suit jacket tie, soft, side_left lighting, ..."

# Search by similarity
iris.add_to_index(glyph)
results = iris.search(query_glyph, top_k=10)

# Search by specific role (e.g., find similar lighting)
results = iris.search_by_role(query_glyph, layer_name="lighting")

# Swap a role
new_glyph = iris.swap(glyph, "direction", "back")

# Combine features from two images
combined = iris.combine(
    glyph_a, ["face_embedding", "expression", "clothing"],
    glyph_b, ["direction", "quality", "composition"],
)

# Compare two images per-role
diff = iris.diff(glyph_a, glyph_b)
# → {"expression": 0.0, "lighting": 1.0, "face_embedding": 0.95, ...}

# ControlNet signals
controls = iris.to_controlnet(glyph)
# → {"openpose": [[x,y,z], ...], "depth": [...]}
```

## How It Works

Iris follows the same bifurcated architecture as all Glyphh models: **HDC handles deterministic parsing, LLM handles generation**.

```
IMAGE INPUT
    ↓
Feature Extractors (8 specialized CV models)
    ↓ face_embedding(512), body_pose(99), depth_map(64), ...
ContinuousProjector (random projection + sign quantization)
    ↓ bipolar {-1, +1} vectors in 2K dims
HDC Binding (bind each feature with its role vector)
    ↓ bind(FACE_ROLE, face_vec) + bind(POSE_ROLE, pose_vec) + ...
Bundle → IrisGlyph (2,000-dim bipolar vector)
    ↓
Searchable, manipulable, decodable
```

**Three query modes:**
- **Image upload** (POST /query/file) → save to temp → full extraction pipeline → encode → similarity search
- **Image path** (local) → full extraction pipeline → encode → similarity search
- **Text query** ("find photos with soft lighting") → intent extraction → encode → search

**No runtime LLM.** Feature extraction uses pretrained CV models at build/encode time. Similarity search, role manipulation, and ControlNet export are all deterministic vector math.

## Model Structure

```
iris/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # 6-layer encoder config, thresholds
├── encoder.py             # ENCODER_CONFIG + encode_query + entry_to_record
├── ingest.py              # offline image ingestion CLI (images → exemplars.jsonl)
├── intent.py              # text query intent extraction (~100 visual keywords)
├── iris.py                # high-level Iris API (encode, search, swap, combine, diff)
├── exports.py             # to_controlnet(), to_prompt(), to_json(), diff()
├── extractors/
│   ├── base.py            # FeatureExtractor ABC + ExtractorRegistry
│   ├── face.py            # InsightFace/ArcFace → 512-dim embedding
│   ├── pose.py            # MediaPipe → 33 keypoints × 3
│   ├── depth.py           # MiDaS → depth map (pooled 8×8)
│   ├── ocr.py             # PaddleOCR → text + bboxes + role inference
│   ├── objects.py         # YOLO v8 → detections + categories
│   ├── color.py           # K-means → named color palette
│   ├── lighting.py        # Histogram analysis → direction/quality/contrast
│   └── composition.py     # Layout analysis → framing + scene category
├── codebooks/
│   ├── primitives.py      # ~150 universal primitives (spatial, lighting, mood, ...)
│   └── categories.py      # ~200 object categories, colors, clothing, relationships
├── data/
│   └── exemplars.jsonl    # seed images with pre-extracted features
├── tests/
│   ├── conftest.py        # shared fixtures (synthetic features, no CV deps needed)
│   ├── test_encoding.py   # config validation, encoding pipeline, similarity
│   ├── test_similarity.py # per-layer similarity, categorical/continuous/BoW
│   ├── test_extractors.py # lighting, composition, color (pure numpy)
│   ├── test_codebooks.py  # primitive vocabulary completeness
│   ├── test_intent.py     # text query parsing
│   ├── test_exports.py    # JSON spec, prompt generation, diff, ControlNet
│   └── test_iris_api.py   # high-level API (encode, search, swap, combine)
├── requirements.txt       # CV dependencies (all optional)
└── README.md
```

## Encoded Layers

| Layer | Weight | Roles | Encoding |
|-------|--------|-------|----------|
| **identity** | 0.25 | face_embedding, expression, age_group | continuous (512-dim), categorical, categorical |
| **pose** | 0.20 | body_pose, head_pose | continuous (99-dim), continuous (3-dim) |
| **appearance** | 0.20 | color_palette, clothing | bag_of_words, bag_of_words |
| **scene** | 0.15 | depth_map, composition, scene_category | continuous (64-dim), categorical, categorical |
| **lighting** | 0.10 | direction, quality, contrast | categorical, categorical, categorical |
| **text** | 0.10 | text_content, text_role, text_position | bag_of_words, categorical, categorical |

**Three encoding types:**
- **Categorical** — Exact symbol match via `generate_symbol()`. Same value = identical vector.
- **Bag of words** — Split into words, encode each, bundle. Shared words = shared signal.
- **Continuous** — `ContinuousProjector` (random projection + sign quantization). Similar float vectors → similar bipolar vectors.

## Testing

```bash
# Run from the iris/ directory
PYTHONPATH="../../glyphh-runtime:..:" python -m pytest tests/ -v

# Or specific test files
PYTHONPATH="../../glyphh-runtime:..:" python -m pytest tests/test_encoding.py -v
PYTHONPATH="../../glyphh-runtime:..:" python -m pytest tests/test_similarity.py -v
```

The test suite runs entirely on synthetic numpy arrays — no CV dependencies or real images needed. 91 tests covering:
- **test_encoding.py** — config validation, full pipeline, similarity ordering, determinism
- **test_similarity.py** — per-layer similarity, categorical/continuous/BoW properties
- **test_extractors.py** — lighting, composition, color extraction (pure numpy)
- **test_codebooks.py** — primitive vocabulary completeness
- **test_intent.py** — text query parsing (actions, targets, keywords)
- **test_exports.py** — JSON spec, prompt generation, diff, ControlNet signals
- **test_iris_api.py** — high-level API (encode, search, swap, combine, diff)

## Use Cases

### Image Search by Attribute
```python
# Find images with similar lighting but different subjects
results = iris.search_by_role(query_glyph, layer_name="lighting")
```

### Consistent Image Generation
```python
# Swap lighting while keeping everything else
for direction in ["front", "side_left", "back", "rim"]:
    variant = iris.swap(ref_glyph, "direction", direction)
    controls = iris.to_controlnet(variant)
    prompt = variant.to_prompt()
    # Send to Flux/SDXL/Midjourney
```

### LLM Context Injection
```python
# Give LLM structured context instead of raw pixels
spec = iris.to_json(glyph)
prompt = f"Analyze this image:\n{json.dumps(spec, indent=2)}\nQuestion: {user_q}"
# Cheaper, more accurate, deterministic
```

### Document/Slide Parsing
```python
glyph = iris.encode("slide.png")
spec = glyph.to_json()
print(spec["text"]["content"])   # All detected text
print(spec["scene"]["composition"])  # Layout classification
```

## Data Format

Exemplars in `data/exemplars.jsonl` are pre-extracted feature sets:

```json
{
  "image_id": "portrait_001",
  "features": {
    "face_embedding": [0.12, -0.34, ...],
    "expression": "happy",
    "age_group": "thirties",
    "body_pose": [0.5, 0.3, ...],
    "head_pose": [0.1, -0.05, 0.02],
    "color_palette": "blue white gray",
    "clothing": "suit jacket tie",
    "depth_map": [0.2, 0.4, ...],
    "composition": "rule_of_thirds",
    "scene_category": "indoor",
    "direction": "side_left",
    "quality": "soft",
    "contrast": "medium",
    "text_content": "",
    "text_role": "none",
    "text_position": "none"
  },
  "metadata": {
    "source": "studio_shoot",
    "camera": "Canon R5"
  }
}
```
