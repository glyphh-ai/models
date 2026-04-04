#!/usr/bin/env python3
"""
Benchmark: FAQ Helpdesk HDC matching accuracy.

Measures top-1 match accuracy against the FAQ data using Pattern A
role-level weighted scoring. No LLM strategies — FAQ matching is
deterministic HDC only.

Query categories:
  clear          — unambiguous single-category questions
  near_collision — questions that overlap multiple categories
  adversarial    — informal phrasing, abbreviations, unusual wording
  open_set       — out-of-scope questions (model should return low confidence)

Usage:
    python benchmark/run.py
    python benchmark/run.py --threshold 0.45
    python benchmark/run.py --output benchmark/results/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from encoder import ENCODER_CONFIG, encode_query, entry_to_record
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity
from glyphh.encoder import Encoder

BENCHMARK_DIR = Path(__file__).parent
QUERIES_PATH  = BENCHMARK_DIR / "queries.json"
DATA_PATH     = BENCHMARK_DIR.parent / "data" / "faq.jsonl"

DEFAULT_THRESHOLD = 0.40

# Pattern A role weights — must match test_similarity.py
ROLE_WEIGHTS = {
    "question": 1.0,
    "category": 0.6,
    "keywords": 0.8,
    "answer":   0.4,
}


# ═══════════════════════════════════════════════════════════════
# FAQ Matcher
# ═══════════════════════════════════════════════════════════════

class FAQMatcher:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold  = threshold
        self.encoder    = Encoder(ENCODER_CONFIG)
        self.faq_glyphs: list[tuple[Any, dict]] = []
        self._load_faq()

    def _load_faq(self):
        with open(DATA_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry  = json.loads(line)
                record = entry_to_record(entry)
                concept = Concept(
                    name=record["attributes"]["question_id"],
                    attributes=record["attributes"],
                )
                glyph = self.encoder.encode(concept)
                self.faq_glyphs.append((glyph, record["metadata"]))

    def match(self, query: str) -> dict[str, Any]:
        start   = time.perf_counter()
        record  = encode_query(query)
        q_concept = Concept(name=record["name"], attributes=record["attributes"])
        q_glyph = self.encoder.encode(q_concept)

        # Flatten query roles
        q_roles: dict = {}
        for layer in q_glyph.layers.values():
            if layer.name.startswith("_"):
                continue
            for seg in layer.segments.values():
                q_roles.update(seg.roles)

        scores: list[tuple[float, dict]] = []
        for faq_glyph, meta in self.faq_glyphs:
            e_roles: dict = {}
            for layer in faq_glyph.layers.values():
                if layer.name.startswith("_"):
                    continue
                for seg in layer.segments.values():
                    e_roles.update(seg.roles)

            weighted_sum = 0.0
            weight_total = 0.0
            for rname, w in ROLE_WEIGHTS.items():
                if rname in q_roles and rname in e_roles:
                    sim = float(cosine_similarity(q_roles[rname].data, e_roles[rname].data))
                    weighted_sum += sim * w
                    weight_total += w

            score = weighted_sum / weight_total if weight_total > 0 else 0.0
            scores.append((score, meta))

        scores.sort(key=lambda x: x[0], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        top_score, top_meta = scores[0] if scores else (0.0, None)
        top_3 = [
            {"question_id": m["question_id"], "score": round(s, 4)}
            for s, m in scores[:3]
        ]

        if top_score >= self.threshold and top_meta:
            return {
                "question_id": top_meta["question_id"],
                "category":    top_meta["category"],
                "answer":      top_meta["answer"],
                "confidence":  round(top_score, 4),
                "latency_ms":  elapsed_ms,
                "top_3":       top_3,
            }
        return {
            "question_id": None,
            "category":    None,
            "answer":      None,
            "confidence":  round(top_score, 4),
            "latency_ms":  elapsed_ms,
            "top_3":       top_3,
        }


# ═══════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════

def score_result(result: dict, expected_id: str | None, expected_category: str | None) -> dict:
    result_id  = result["question_id"]
    result_cat = result["category"]

    if expected_id is None:
        # Open-set: correct abstain = result_id is None
        correct = result_id is None
        label   = "correct_abstain" if correct else "false_positive"
    else:
        if result_id is None:
            correct = False
            label   = "false_abstain"
        elif result_id == expected_id:
            correct = True
            label   = "correct"
        else:
            correct = False
            label   = "wrong_match"

    category_correct = (result_cat == expected_category) if expected_category else (result_cat is None)

    return {
        "correct":          correct,
        "label":            label,
        "category_correct": category_correct,
    }


# ═══════════════════════════════════════════════════════════════
# Aggregation & Reporting
# ═══════════════════════════════════════════════════════════════

def _aggregate(results: list[dict]) -> dict:
    total    = len(results)
    correct  = sum(1 for r in results if r["correct"])
    cat_ok   = sum(1 for r in results if r["category_correct"])

    in_scope = [r for r in results if r["expected_id"] is not None]
    oos      = [r for r in results if r["expected_id"] is None]

    in_scope_correct = sum(1 for r in in_scope if r["correct"])
    oos_correct      = sum(1 for r in oos if r["correct"])

    latencies = [r["latency_ms"] for r in results]

    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r.get("query_category", "unknown"), []).append(r)

    cat_breakdown = {}
    for cat, cat_results in sorted(categories.items()):
        cat_correct = sum(1 for r in cat_results if r["correct"])
        cat_breakdown[cat] = {
            "total":    len(cat_results),
            "correct":  cat_correct,
            "accuracy": cat_correct / len(cat_results) if cat_results else 0.0,
        }

    return {
        "total":              total,
        "accuracy":           correct / total if total else 0.0,
        "correct":            correct,
        "category_accuracy":  cat_ok / total if total else 0.0,
        "in_scope_accuracy":  in_scope_correct / len(in_scope) if in_scope else 0.0,
        "in_scope_correct":   in_scope_correct,
        "in_scope_total":     len(in_scope),
        "oos_accuracy":       oos_correct / len(oos) if oos else 1.0,
        "oos_correct":        oos_correct,
        "oos_total":          len(oos),
        "latency_mean_ms":    sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95_ms":     sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        "categories":         cat_breakdown,
    }


def _print_report(agg: dict, raw: list[dict]):
    W = 80
    print("\n" + "=" * W)
    print("  FAQ HELPDESK BENCHMARK — RESULTS")
    print("=" * W)

    print(f"\n  {'Metric':<30} {'Value':>10}")
    print("  " + "-" * 42)
    print(f"  {'Total queries':<30} {agg['total']:>10}")
    print(f"  {'Match accuracy (top-1)':<30} {agg['accuracy']:>9.1%}")
    print(f"  {'In-scope accuracy':<30} {agg['in_scope_accuracy']:>9.1%}  ({agg['in_scope_correct']}/{agg['in_scope_total']})")
    print(f"  {'Open-set abstain accuracy':<30} {agg['oos_accuracy']:>9.1%}  ({agg['oos_correct']}/{agg['oos_total']})")
    print(f"  {'Category accuracy':<30} {agg['category_accuracy']:>9.1%}")
    print(f"  {'Latency mean (ms)':<30} {agg['latency_mean_ms']:>9.1f}")
    print(f"  {'Latency p95 (ms)':<30} {agg['latency_p95_ms']:>9.1f}")

    print(f"\n  {'Category':<20} {'Acc':>8} {'n':>4}")
    print("  " + "-" * 34)
    for cat, cd in sorted(agg["categories"].items()):
        print(f"  {cat:<20} {cd['accuracy']:>7.1%} {cd['total']:>4}")

    failures = [r for r in raw if not r["correct"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for r in failures:
            exp = r["expected_id"] or "null"
            got = r["result_id"] or "null"
            print(f"    [{r['label']:>15}]  {r['query'][:55]:<55}  exp={exp}  got={got}  conf={r['confidence']:.3f}")

    print("\n" + "=" * W)


def _progress(current: int, total: int):
    pct    = current / total if total else 0
    filled = int(30 * pct)
    bar    = "█" * filled + "░" * (30 - filled)
    print(f"\r  [{bar}] {current}/{total}", end="", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run_benchmark(threshold: float = DEFAULT_THRESHOLD, output_dir: str | None = None):
    with open(QUERIES_PATH) as f:
        query_data = json.load(f)
    queries = query_data["queries"]

    matcher = FAQMatcher(threshold=threshold)
    print(f"Loaded {len(matcher.faq_glyphs)} FAQ entries, {len(queries)} benchmark queries")
    print(f"Threshold: {threshold}\n")

    raw_results = []
    for qi, q in enumerate(queries):
        _progress(qi + 1, len(queries))
        result = matcher.match(q["query"])
        scoring = score_result(result, q["expected_id"], q["expected_category"])
        raw_results.append({
            "query_id":        q["id"],
            "query_category":  q["category"],
            "query":           q["query"],
            "expected_id":     q["expected_id"],
            "expected_category": q["expected_category"],
            "result_id":       result["question_id"],
            "result_category": result["category"],
            "confidence":      result["confidence"],
            "latency_ms":      result["latency_ms"],
            "top_3":           result["top_3"],
            **scoring,
        })

    print()  # newline after progress bar

    agg = _aggregate(raw_results)
    _print_report(agg, raw_results)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "results.json", "w") as f:
            json.dump({"summary": agg, "raw": raw_results}, f, indent=2, default=str)
        print(f"Results saved to {out_path}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAQ Helpdesk HDC Benchmark")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Confidence threshold for a match (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--output", type=str, help="Directory to save raw JSON results")
    args = parser.parse_args()

    run_benchmark(threshold=args.threshold, output_dir=args.output)
