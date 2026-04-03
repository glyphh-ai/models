"""
Encoder for the Sentinel Security model.

Exports:
  ENCODER_CONFIG — EncoderConfig with 4 layers: tactic, technique, source, temporal
  encode_query(text) — converts security event text to a Concept dict
  entry_to_record(entry) — converts a JSONL exemplar entry to a build record

Architecture:
  NL Extraction — intent.py (local, deterministic):
    - analyze_event(text) → MITRE ATT&CK features across 4 dimensions
    - Pattern-based detection for tactics, techniques, sources
    - Urgency scoring from threat keywords

  Main model (seed=42):
  - Tactic layer (0.35): mitre_tactic (lexicon) + tactic_signals (BoW)
  - Technique layer (0.30): technique_id (lexicon) + technique_signals (BoW)
  - Source layer (0.20): event_source (lexicon) + source_signals (BoW)
  - Temporal layer (0.15): urgency (numeric) + severity (lexicon)
"""

import hashlib

from glyphh.core.config import (
    EncoderConfig,
    EncodingStrategy,
    Layer,
    NumericConfig,
    Role,
    Segment,
)

from intent import analyze_event

# ---------------------------------------------------------------------------
# ENCODER_CONFIG — 4-layer MITRE ATT&CK encoding
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="tactic",
            similarity_weight=0.35,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="mitre_tactic",
                            similarity_weight=1.0,
                            lexicons=[
                                "none", "reconnaissance", "resource_development",
                                "initial_access", "execution", "persistence",
                                "privilege_escalation", "defense_evasion",
                                "credential_access", "discovery",
                                "lateral_movement", "collection",
                                "command_and_control", "exfiltration", "impact",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="tactic_signals",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="technique",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="technique_id",
                            similarity_weight=1.0,
                            lexicons=[
                                "none", "phishing", "spearphishing_attachment",
                                "command_scripting", "powershell", "scheduled_task",
                                "registry_run_key", "os_credential_dumping",
                                "lsass_memory", "remote_services", "smb_shares",
                                "data_staged", "data_encrypted",
                                "exfil_over_c2", "exfil_over_web",
                                "data_destruction",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="technique_signals",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="source",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="event_source",
                            similarity_weight=1.0,
                            lexicons=[
                                "unknown", "endpoint", "network", "email",
                                "dns", "identity", "cloud", "firewall",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="source_signals",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="temporal",
            similarity_weight=0.15,
            segments=[
                Segment(
                    name="scoring",
                    roles=[
                        Role(
                            name="urgency",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=1.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=10.0,
                            ),
                        ),
                        Role(
                            name="severity",
                            similarity_weight=1.0,
                            lexicons=[
                                "info", "low", "medium", "high", "critical",
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# encode_query — event text -> Concept dict
# ---------------------------------------------------------------------------

def encode_query(text: str) -> dict:
    """Convert security event text into a Concept-compatible dict."""
    features = analyze_event(text)
    stable_id = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return {
        "name": f"event_{stable_id:08d}",
        "attributes": features,
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL exemplar -> build record
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL exemplar entry into a record for building/encoding."""
    text = entry.get("text", "")

    if "mitre_tactic" in entry:
        # Pre-extracted features
        attributes = {
            "mitre_tactic": entry["mitre_tactic"],
            "tactic_signals": entry.get("tactic_signals", ""),
            "technique_id": entry.get("technique_id", "none"),
            "technique_signals": entry.get("technique_signals", ""),
            "event_source": entry.get("event_source", "unknown"),
            "source_signals": entry.get("source_signals", ""),
            "urgency": entry.get("urgency", 0),
            "severity": entry.get("severity", "info"),
        }
    else:
        attributes = analyze_event(text)

    return {
        "concept_text": entry.get("label", f"exemplar_{entry.get('id', 'unknown')}"),
        "attributes": attributes,
        "metadata": {
            "label": entry.get("label", "unknown"),
            "mitre_tactic": attributes.get("mitre_tactic", "none"),
            "technique_id": attributes.get("technique_id", "none"),
            "text": text[:200],
        },
    }
