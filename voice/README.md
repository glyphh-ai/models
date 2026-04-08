# Glyphh Voice

Encodes human voice identity and cognitive/emotional state into HDC vectors from audio features. Speaker recognition, liveness detection, emotional state tracking, and baseline drift analysis — all via deterministic vector operations. No LLM, no cloud API, no neural network.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs](https://glyphh.ai/docs)** · **[Glyphh Hub](https://glyphh.ai/hub)**

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

### 2. Install audio dependencies

```bash
# openSMILE (recommended — full eGeMAPS feature set)
pip install opensmile

# OR librosa (fallback — no openSMILE dependency, covers all features via LPC/pyin)
pip install librosa soundfile
```

Both backends extract the same feature roles. openSMILE uses the standardized eGeMAPS v02 feature set. The librosa fallback computes equivalent features using LPC (formants), pyin (F0/jitter), and autocorrelation (HNR).

### 3. Install the model

```bash
# Start the Glyphh shell (prompts login on first run)
glyphh

# Inside the shell:
# glyphh> hub install model-voice
```

### 4. Enroll and identify speakers

```bash
# Inside the shell:
# glyphh> chat "enroll my voice"        # records audio and creates a voice glyph
# glyphh> chat "who is speaking?"       # identifies speaker from audio
```

## How It Works

Voice is encoded into 4 independent HDC layers. The **identity layer** captures vocal tract geometry (MFCCs, formants, HNR) — physical characteristics that are stable across sessions and nearly impossible to consciously fake. The **state layers** (emotional, cognitive, cadence) capture how someone is speaking right now — prosody, voice stability, rhythm.

```
Audio input (WAV/MP3)
  |
  v
openSMILE eGeMAPS  ──or──  librosa fallback
  |                           |
  v                           v
features.py — maps 37 acoustic features to encoder roles
  |
  v
encoder.py — 4-layer HDC encoding (dim=2000)
  |
  +---> Identity layer (0.55):      MFCCs 1-13, deltas, formants F1-F3, HNR
  +---> Emotional state layer (0.15): F0 mean/std/p20/p80, loudness, spectral slope/flux
  +---> Cognitive load layer (0.10):  jitter, shimmer, alpha ratio, hammarberg, F1 bandwidth
  +---> Cadence layer (0.20):        loudness peaks/sec, voiced/unvoiced segment lengths
  |
  v
pgvector cosine similarity — search enrolled voice glyphs
  |
  v
Identity match + state drift analysis + liveness score
```

### Speaker Recognition

Speaker matching uses **Pattern A** (role-level weighted cosine):

1. pgvector cosine on identity layer cortex gives top-5 candidates
2. Per-role vectors are compared individually (each MFCC, formant, HNR)
3. Weighted average produces final identity score

A voice clone that matches MFCCs 1-4 but misses the subtle detail in MFCCs 5-13 scores low on those roles, dragging the weighted average down. The 13 MFCCs capture increasingly fine resonance detail that is nearly impossible to consciously control.

### State Drift Analysis

Each enrollment creates a temporal snapshot keyed by speaker ID (G-number) with an auto-generated timestamp. Over time, you can track how someone's voice changes:

- **Emotional drift** — stress, excitement, fatigue (F0 contour, loudness dynamics)
- **Cognitive load** — concentration, deception, strain (jitter, shimmer, alpha ratio)
- **Cadence shift** — speech rhythm changes (pacing, pausing patterns)

### Liveness Detection

The `compute_liveness_score()` function detects replay attacks by analyzing spectral characteristics that change when audio is played through a speaker and re-recorded:

1. **High-frequency energy ratio** — speakers roll off above 4-8kHz, live voice retains energy
2. **Spectral flatness** — speaker resonance creates tonal peaks, lowering flatness

Live voice typically scores 0.6-1.0. Replay attacks score below 0.4.

## 4-Layer Architecture

### Identity layer (weight: 0.55)
Physical vocal tract geometry — your voice fingerprint. Stable across sessions, time of day, emotional state. These features are determined by the shape and size of your throat, mouth, and nasal passages.

| Role | Weight | Type | Range | Description |
|------|--------|------|-------|-------------|
| mfcc_1 through mfcc_13 | 1.0-0.5 | numeric (thermometer) | varies | Mel-frequency cepstral coefficients — vocal tract resonance shape |
| mfcc_delta_1 through mfcc_delta_4 | 0.7-0.6 | numeric (thermometer) | varies | MFCC rate of change — articulatory dynamics |
| formant_f1 | 0.7 | numeric (thermometer) | 200-1000 Hz | First formant frequency |
| formant_f2 | 0.6 | numeric (thermometer) | 500-3000 Hz | Second formant frequency |
| formant_f3 | 0.5 | numeric (thermometer) | 1500-4000 Hz | Third formant frequency |
| hnr | 0.8 | numeric (thermometer) | 0-40 dB | Harmonics-to-noise ratio — voice clarity |

### Emotional state layer (weight: 0.15)
Prosodic contour — how you're speaking right now. Changes per utterance. Encodes arousal, valence, and affect.

| Role | Weight | Type | Range | Description |
|------|--------|------|-------|-------------|
| f0_mean | 1.0 | numeric (thermometer) | 1-40 semitones | Mean fundamental frequency |
| f0_std | 0.9 | numeric (thermometer) | 0-1 | F0 variability (normalized) |
| f0_p20, f0_p80 | 0.8 | numeric (thermometer) | 1-40 semitones | Pitch range percentiles — compression/exaggeration detection |
| loudness_mean | 0.8 | numeric (thermometer) | 0-2 | Average loudness (RMS) |
| loudness_std | 0.7 | numeric (thermometer) | 0-1 | Loudness variability |
| spectral_slope | 0.6 | numeric (thermometer) | -0.05-0.05 | Spectral tilt |
| spectral_flux | 0.7 | numeric (thermometer) | 0-0.5 | Rate of spectral change — monotone speech = low flux |

### Cognitive load layer (weight: 0.10)
Voice stability markers — involuntary signals of stress, fatigue, or deception. These features are controlled by the autonomic nervous system and are extremely difficult to consciously suppress.

| Role | Weight | Type | Range | Description |
|------|--------|------|-------|-------------|
| jitter | 1.0 | numeric (thermometer) | 0-0.10 | Pitch perturbation (cycle-to-cycle F0 variation) |
| shimmer | 0.9 | numeric (thermometer) | 0-3.0 | Amplitude perturbation |
| voiced_std | 0.7 | numeric (thermometer) | 0-1.0 | Voiced segment length variability |
| pause_std | 0.8 | numeric (thermometer) | 0-1.0 | Unvoiced segment irregularity |
| alpha_ratio | 0.9 | numeric (thermometer) | -10-10 dB | Energy below/above 1kHz — laryngeal tension |
| hammarberg_index | 0.8 | numeric (thermometer) | 0-40 dB | Spectral peak ratio — throat tension marker |
| formant_f1_bw | 0.7 | numeric (thermometer) | 50-500 Hz | F1 bandwidth — widens under stress |
| mfcc_1_cv | 0.7 | numeric (thermometer) | 0-2.0 | MFCC1 coefficient of variation — within-utterance instability |

### Cadence layer (weight: 0.20)
Speech rhythm and phrasing — cognitive style and personality signature.

| Role | Weight | Type | Range | Description |
|------|--------|------|-------|-------------|
| loudness_peaks_per_sec | 1.0 | numeric (thermometer) | 0-8.0 | Syllabic rate proxy |
| voiced_segment_mean | 0.9 | numeric (thermometer) | 0-1.0 s | Average voiced segment length |
| unvoiced_segment_mean | 0.8 | numeric (thermometer) | 0-1.0 s | Average pause length |
| voiced_segments_per_sec | 0.7 | numeric (thermometer) | 0-10.0 | Speech rate |

## Model Structure

```
voice/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config, thresholds, layer definitions
├── encoder.py             # EncoderConfig + encode_features + scoring functions
├── features.py            # openSMILE + librosa feature extraction, liveness detection
├── build.py               # package model into .glyphh file
├── requirements.txt       # opensmile, librosa, soundfile
├── tests/
│   ├── conftest.py        # shared fixtures
│   ├── test_encoding.py   # config validation, layer structure, encoding
│   └── test_similarity.py # identity matching, state drift, scoring
├── data/                  # (empty — voice enrolls speakers at runtime)
├── LICENSE                # AGPL-3.0
└── README.md
```

**Runtime enrollment model.** Unlike other Glyphh models that ship with exemplar data, the voice model enrolls speakers at runtime. The `data/` directory is intentionally empty — voice glyphs are created when speakers enroll via the MCP API.

## Testing

Run the test suite before deploying:

```bash
# Inside the glyphh shell:
# glyphh> model test .
# glyphh> model test . -v
# glyphh> model test . -k identity

# Or directly
cd voice/
pytest tests/ -v
```

The test suite includes:
- **test_encoding.py** — config validation, 4-layer structure, role encoding, feature-to-concept mapping
- **test_similarity.py** — identity matching (same speaker vs different), state drift detection, liveness scoring

## MCP Integration

LLM agents interact with voice via 5 MCP tools:

### voice_enroll

Enroll a new speaker. Accepts base64-encoded WAV audio, extracts features, encodes as an HDC glyph, and assigns a G-number. Supports multi-phrase enrollment (3 phrases recommended for robust identity).

```python
POST /{org_id}/voice/mcp
{
    "tool": "voice_enroll",
    "arguments": {
        "audio_base64": "<base64 WAV data>",
        "speaker_name": "Alice",
        "recording_label": "baseline"
    }
}

# Response:
{
    "state": "DONE",
    "g_number": "G-00001",
    "liveness": {"is_live": true, "liveness_score": 0.87},
    "features_summary": {"mfcc_1": 23.4, "f0_mean": 18.2, ...}
}
```

### voice_identify

Identify a speaker from audio. Returns match confidence, G-number, and state drift vs baseline.

```python
POST /{org_id}/voice/mcp
{
    "tool": "voice_identify",
    "arguments": {
        "audio_base64": "<base64 WAV data>"
    }
}

# Response:
{
    "state": "DONE",
    "match": {"g_number": "G-00001", "score": 0.91, "speaker_name": "Alice"},
    "drift": {
        "identity": 0.95,
        "emotional_state": 0.72,
        "cognitive_load": 0.68,
        "cadence": 0.88
    }
}
```

### voice_compare

Compare two statement recordings against a baseline. Returns per-layer divergence scores — useful for detecting when someone's voice shifts (stress, deception, fatigue).

```python
POST /{org_id}/voice/mcp
{
    "tool": "voice_compare",
    "arguments": {
        "baseline_g": "G-00001",
        "statement_a_g": "G-00002",
        "statement_b_g": "G-00003"
    }
}
```

### voice_list

List enrolled voice identities (count and G-numbers).

### voice_cleanup

Delete all enrolled glyphs for a session ID (cleanup temporary/demo data).

## Feature Extraction

The model extracts 37 acoustic features from audio, mapped to 4 encoder layers:

| Feature Source | Features | Layer |
|---------------|----------|-------|
| MFCCs (openSMILE `mfcc1-13_sma3_amean`) | 13 coefficients + 4 deltas | Identity |
| Formants (LPC roots or openSMILE) | F1, F2, F3 frequencies + F1 bandwidth | Identity + Cognitive |
| HNR (autocorrelation) | Harmonics-to-noise ratio | Identity |
| F0 (pyin or openSMILE) | Mean, std, p20, p80 (semitones re 27.5Hz) | Emotional |
| Loudness (RMS) | Mean, std | Emotional |
| Spectral (STFT) | Slope, flux, alpha ratio, hammarberg index | Emotional + Cognitive |
| Perturbation (pyin) | Jitter (F0), shimmer (amplitude) | Cognitive |
| Segmentation (voiced/unvoiced) | Segment lengths, counts, variability | Cadence + Cognitive |

### openSMILE vs librosa

| | openSMILE | librosa fallback |
|---|---|---|
| **Feature set** | eGeMAPS v02 (standardized, 88 functionals) | Equivalent features via LPC, pyin, STFT |
| **Formants** | Direct from openSMILE | LPC roots (noisier, wider bins compensate) |
| **F0** | openSMILE F0 tracker | pyin (robust probabilistic tracker) |
| **Jitter/shimmer** | openSMILE standard | pyin-based (works on noisy browser audio) |
| **Install** | `pip install opensmile` (larger, needs C compiler) | `pip install librosa soundfile` (pure Python) |
| **Accuracy** | Reference implementation | Comparable for identity matching |

## Architecture — Agent + Model Closed Loop

The voice model is a **biometric encoding and comparison layer**. Your LLM agent handles decisions; the model handles voice identity and state analysis.

```
+---------------------------------------------------------+
|                    LLM AGENT (Claude)                   |
|                                                         |
|  "Enroll this speaker" | "Who is speaking?"             |
|  "Compare these two statements against baseline"        |
|  "Has the speaker's stress level changed?"              |
+---------+--------------------------+--------------------+
          | MCP voice_enroll/        | Agent actions
          | voice_identify/compare   |
          v                          v
+------------------------+   +----------------------------+
|   GLYPHH RUNTIME       |   |   APPLICATION              |
|                        |   |                            |
|  Extract features      |   |  Access control            |
|  HDC encode (4 layers) |   |  Customer verification     |
|  pgvector identity     |   |  Meeting analytics         |
|  search + drift score  |   |  Liveness gates            |
+------------------------+   +----------------------------+
```

## Use Cases

- **Voice authentication** — enroll users, verify identity from audio, detect replay attacks
- **Customer service** — identify returning callers, track emotional state across a call
- **Meeting analytics** — who spoke when, stress/engagement levels per speaker
- **Liveness detection** — distinguish live voice from speaker playback
- **Longitudinal tracking** — monitor voice baseline drift over time (health, fatigue, stress)
