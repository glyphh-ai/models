#!/usr/bin/env python3
"""
LLM Baseline Benchmark for Pipedream Action Router.

Runs the same test queries through LLM routing and compares head-to-head
with HDC results. Tests two scenarios:

  Test A — App-scoped routing (fair comparison):
    LLM sees the query + all actions for the correct app.
    Same information HDC has after app extraction.

  Test B — Cold routing (real-world scale test):
    LLM sees the query + actions from 50 random apps (including the correct one).
    Tests whether the LLM can handle multi-app disambiguation.

Usage:
    # Claude Sonnet (default)
    python benchmark/llm_baseline.py --limit 1000

    # GPT-4o
    python benchmark/llm_baseline.py --limit 1000 --model gpt-4o

    # Full run (expensive — ~$15-25 for 1000 queries)
    python benchmark/llm_baseline.py --limit 1000 --test both

    # Resume from checkpoint
    python benchmark/llm_baseline.py --limit 1000 --resume

Requires:
    ANTHROPIC_API_KEY or OPENAI_API_KEY
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent
DATA_DIR = MODEL_DIR / "data"
RESULTS_DIR = MODEL_DIR / "benchmark"

# ---------------------------------------------------------------------------
# Load action catalog (ground truth)
# ---------------------------------------------------------------------------

def load_action_catalog() -> dict[str, list[dict]]:
    """Load unique actions grouped by app_slug from exemplars.jsonl."""
    actions_by_app: dict[str, dict[str, dict]] = defaultdict(dict)

    with open(DATA_DIR / "exemplars.jsonl") as f:
        for line in f:
            e = json.loads(line)
            key = e["action_key"]
            slug = e["app_slug"]
            if key not in actions_by_app[slug]:
                actions_by_app[slug][key] = {
                    "action_key": key,
                    "app_slug": slug,
                    "app_name": e.get("app_name", slug),
                    "action_name": e.get("action_name", key),
                    "action": e.get("action", ""),
                    "target": e.get("target", ""),
                    "domain": e.get("domain", ""),
                }

    # Convert to list per app
    return {
        slug: list(actions.values())
        for slug, actions in actions_by_app.items()
    }


def load_test_queries(limit: int = 1000, seed: int = 42) -> list[dict]:
    """Load stratified sample of test queries.

    Stratifies by test_type to get representative coverage.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)

    with open(DATA_DIR / "test_queries.jsonl") as f:
        for line in f:
            e = json.loads(line)
            by_type[e.get("test_type", "unknown")].append(e)

    rng = random.Random(seed)
    sampled = []
    total = sum(len(v) for v in by_type.values())

    for test_type, queries in by_type.items():
        # Proportional sampling
        n = max(1, round(limit * len(queries) / total))
        rng.shuffle(queries)
        sampled.extend(queries[:n])

    rng.shuffle(sampled)
    return sampled[:limit]


# ---------------------------------------------------------------------------
# LLM Clients
# ---------------------------------------------------------------------------

class ClaudeClient:
    """Claude API client for routing benchmark."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = model

    def route(self, query: str, actions: list[dict]) -> dict:
        """Ask Claude to route a query to an action. Returns {action_key, latency_ms, tokens}."""
        action_list = "\n".join(
            f"- {a['action_key']}: {a['action_name']} ({a['app_name']})"
            for a in actions
        )

        prompt = f"""You are a tool router. Given a user query, select the single best matching action from the list below.

ACTIONS:
{action_list}

USER QUERY: {query}

Respond with ONLY the action_key, nothing else. No explanation, no quotes, no punctuation."""

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return {
            "action_key": text,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


class OpenAIClient:
    """OpenAI API client for routing benchmark."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        self.client = openai.OpenAI()
        self.model = model
        self.name = model

    def route(self, query: str, actions: list[dict]) -> dict:
        """Ask GPT to route a query to an action."""
        action_list = "\n".join(
            f"- {a['action_key']}: {a['action_name']} ({a['app_name']})"
            for a in actions
        )

        prompt = f"""You are a tool router. Given a user query, select the single best matching action from the list below.

ACTIONS:
{action_list}

USER QUERY: {query}

Respond with ONLY the action_key, nothing else. No explanation, no quotes, no punctuation."""

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return {
            "action_key": text,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_test_a(
    client,
    queries: list[dict],
    catalog: dict[str, list[dict]],
    checkpoint_path: Path,
    existing: dict | None = None,
) -> dict:
    """Test A: App-scoped routing.

    LLM sees query + all actions for the correct app.
    Fair comparison — same info HDC has after app extraction.
    """
    results = existing or {
        "test": "A_app_scoped",
        "model": client.name,
        "total": len(queries),
        "completed": 0,
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "total_latency_ms": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "failures": [],
        "per_query": [],
    }

    start_idx = results["completed"]
    remaining = queries[start_idx:]

    for i, q in enumerate(remaining):
        idx = start_idx + i
        query = q["query"]
        expected = q["expected_action_key"]
        app_slug = q["expected_app_slug"]

        actions = catalog.get(app_slug, [])
        if not actions:
            results["errors"] += 1
            results["completed"] += 1
            continue

        try:
            r = client.route(query, actions)
            predicted = r["action_key"].strip().strip('"').strip("'")

            correct = predicted == expected
            results["correct" if correct else "incorrect"] += 1
            results["total_latency_ms"] += r["latency_ms"]
            results["total_input_tokens"] += r["input_tokens"]
            results["total_output_tokens"] += r["output_tokens"]

            entry = {
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "app_slug": app_slug,
                "latency_ms": round(r["latency_ms"], 1),
                "tokens": r["total_tokens"],
                "test_type": q.get("test_type", ""),
                "num_actions": len(actions),
            }
            results["per_query"].append(entry)

            if not correct:
                results["failures"].append(entry)

        except Exception as e:
            results["errors"] += 1
            results["per_query"].append({
                "query": query, "expected": expected,
                "predicted": None, "correct": False,
                "error": str(e), "app_slug": app_slug,
            })

        results["completed"] += 1

        # Progress + checkpoint every 25 queries
        if (idx + 1) % 25 == 0:
            acc = results["correct"] / max(results["completed"], 1) * 100
            avg_lat = results["total_latency_ms"] / max(results["completed"], 1)
            print(
                f"  [{idx+1}/{len(queries)}] "
                f"acc={acc:.1f}% "
                f"avg_lat={avg_lat:.0f}ms "
                f"tokens={results['total_input_tokens'] + results['total_output_tokens']}"
            )
            _save_checkpoint(results, checkpoint_path)

    _save_checkpoint(results, checkpoint_path)
    return results


def run_test_b(
    client,
    queries: list[dict],
    catalog: dict[str, list[dict]],
    checkpoint_path: Path,
    existing: dict | None = None,
    num_distractor_apps: int = 49,
    seed: int = 42,
) -> dict:
    """Test B: Cold routing (multi-app).

    LLM sees query + actions from 50 apps (correct app + 49 random distractors).
    Tests real-world scale where app isn't pre-identified.
    """
    rng = random.Random(seed)
    all_app_slugs = list(catalog.keys())

    results = existing or {
        "test": "B_cold_routing",
        "model": client.name,
        "num_apps_shown": num_distractor_apps + 1,
        "total": len(queries),
        "completed": 0,
        "correct_action": 0,
        "correct_app": 0,
        "incorrect": 0,
        "errors": 0,
        "total_latency_ms": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "failures": [],
        "per_query": [],
    }

    start_idx = results["completed"]
    remaining = queries[start_idx:]

    for i, q in enumerate(remaining):
        idx = start_idx + i
        query = q["query"]
        expected = q["expected_action_key"]
        app_slug = q["expected_app_slug"]

        # Build action list: correct app + random distractors
        distractor_slugs = [s for s in all_app_slugs if s != app_slug]
        rng.shuffle(distractor_slugs)
        selected_slugs = [app_slug] + distractor_slugs[:num_distractor_apps]
        rng.shuffle(selected_slugs)  # randomize order

        actions = []
        for slug in selected_slugs:
            actions.extend(catalog.get(slug, []))

        try:
            r = client.route(query, actions)
            predicted = r["action_key"].strip().strip('"').strip("'")

            action_correct = predicted == expected
            # Check if at least the app is correct
            predicted_app = None
            for a in actions:
                if a["action_key"] == predicted:
                    predicted_app = a["app_slug"]
                    break
            app_correct = predicted_app == app_slug

            if action_correct:
                results["correct_action"] += 1
            if app_correct:
                results["correct_app"] += 1
            if not action_correct:
                results["incorrect"] += 1

            results["total_latency_ms"] += r["latency_ms"]
            results["total_input_tokens"] += r["input_tokens"]
            results["total_output_tokens"] += r["output_tokens"]

            entry = {
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "action_correct": action_correct,
                "app_correct": app_correct,
                "app_slug": app_slug,
                "predicted_app": predicted_app,
                "latency_ms": round(r["latency_ms"], 1),
                "tokens": r["total_tokens"],
                "num_actions_shown": len(actions),
                "test_type": q.get("test_type", ""),
            }
            results["per_query"].append(entry)

            if not action_correct:
                results["failures"].append(entry)

        except Exception as e:
            results["errors"] += 1
            results["per_query"].append({
                "query": query, "expected": expected,
                "predicted": None, "action_correct": False,
                "app_correct": False, "error": str(e),
            })

        results["completed"] += 1

        if (idx + 1) % 25 == 0:
            act_acc = results["correct_action"] / max(results["completed"], 1) * 100
            app_acc = results["correct_app"] / max(results["completed"], 1) * 100
            avg_lat = results["total_latency_ms"] / max(results["completed"], 1)
            print(
                f"  [{idx+1}/{len(queries)}] "
                f"action={act_acc:.1f}% app={app_acc:.1f}% "
                f"avg_lat={avg_lat:.0f}ms "
                f"tokens={results['total_input_tokens'] + results['total_output_tokens']}"
            )
            _save_checkpoint(results, checkpoint_path)

    _save_checkpoint(results, checkpoint_path)
    return results


def _save_checkpoint(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_summary(results: dict):
    """Print formatted summary of benchmark results."""
    test = results["test"]
    model = results["model"]
    n = results["completed"]
    errors = results["errors"]

    print(f"\n{'='*60}")
    print(f"  {test} — {model}")
    print(f"{'='*60}")
    print(f"  Queries:     {n}")
    print(f"  Errors:      {errors}")

    if "correct" in results:
        # Test A
        correct = results["correct"]
        acc = correct / max(n - errors, 1) * 100
        print(f"  Correct:     {correct}/{n - errors} ({acc:.1f}%)")
    else:
        # Test B
        act = results["correct_action"]
        app = results["correct_app"]
        act_acc = act / max(n - errors, 1) * 100
        app_acc = app / max(n - errors, 1) * 100
        print(f"  Action correct: {act}/{n - errors} ({act_acc:.1f}%)")
        print(f"  App correct:    {app}/{n - errors} ({app_acc:.1f}%)")

    avg_lat = results["total_latency_ms"] / max(n, 1)
    total_tokens = results["total_input_tokens"] + results["total_output_tokens"]
    avg_tokens = total_tokens / max(n, 1)
    print(f"  Avg latency: {avg_lat:.0f}ms")
    print(f"  Avg tokens:  {avg_tokens:.0f}")
    print(f"  Total tokens:{total_tokens:,}")

    # Cost estimate (Claude Sonnet: $3/$15 per MTok, GPT-4o: $2.50/$10)
    if "claude" in model.lower() or "sonnet" in model.lower():
        cost = (results["total_input_tokens"] * 3 + results["total_output_tokens"] * 15) / 1_000_000
    else:
        cost = (results["total_input_tokens"] * 2.5 + results["total_output_tokens"] * 10) / 1_000_000
    print(f"  Est. cost:   ${cost:.2f}")

    # Failure breakdown by test_type
    if results.get("failures"):
        print(f"\n  Failures by test_type:")
        by_type = Counter(f.get("test_type", "?") for f in results["failures"])
        for t, c in by_type.most_common():
            print(f"    {t}: {c}")

    # Per-app accuracy for apps with >1 action (Test A)
    if "correct" in results and results.get("per_query"):
        by_app = defaultdict(lambda: {"total": 0, "correct": 0, "num_actions": 0})
        for pq in results["per_query"]:
            slug = pq.get("app_slug", "?")
            by_app[slug]["total"] += 1
            by_app[slug]["correct"] += 1 if pq.get("correct") else 0
            by_app[slug]["num_actions"] = max(by_app[slug]["num_actions"], pq.get("num_actions", 0))

        # Show worst apps (with >5 queries)
        worst = sorted(
            [(s, d) for s, d in by_app.items() if d["total"] >= 5],
            key=lambda x: x[1]["correct"] / x[1]["total"],
        )
        if worst:
            print(f"\n  Worst apps (>=5 queries):")
            for slug, d in worst[:10]:
                acc = d["correct"] / d["total"] * 100
                print(f"    {slug}: {d['correct']}/{d['total']} ({acc:.0f}%) — {d['num_actions']} actions")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM baseline benchmark")
    parser.add_argument("--limit", type=int, default=1000, help="Number of queries to test")
    parser.add_argument("--model", default="claude-sonnet", help="LLM model: claude-sonnet, claude-haiku, gpt-4o, gpt-4o-mini")
    parser.add_argument("--test", default="A", choices=["A", "B", "both"], help="Test scenario")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Map model shortcuts
    model_map = {
        "claude-sonnet": "claude-sonnet-4-20250514",
        "claude-haiku": "claude-haiku-4-5-20251001",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    }
    model_id = model_map.get(args.model, args.model)

    # Create client
    if "claude" in model_id or "sonnet" in model_id or "haiku" in model_id:
        client = ClaudeClient(model_id)
    else:
        client = OpenAIClient(model_id)

    print(f"Loading catalog...")
    catalog = load_action_catalog()
    print(f"  {len(catalog)} apps, {sum(len(v) for v in catalog.values())} unique actions")

    print(f"Loading {args.limit} test queries (stratified sample, seed={args.seed})...")
    queries = load_test_queries(args.limit, args.seed)
    print(f"  {len(queries)} queries loaded")

    type_dist = Counter(q.get("test_type", "?") for q in queries)
    for t, c in type_dist.most_common():
        print(f"    {t}: {c}")

    # Run Test A
    if args.test in ("A", "both"):
        print(f"\n--- Test A: App-Scoped Routing ({client.name}) ---")
        cp_path = RESULTS_DIR / f"results_A_{args.model}.json"
        existing = None
        if args.resume and cp_path.exists():
            existing = json.loads(cp_path.read_text())
            print(f"  Resuming from checkpoint: {existing['completed']}/{len(queries)} done")

        results_a = run_test_a(client, queries, catalog, cp_path, existing)
        print_summary(results_a)

    # Run Test B
    if args.test in ("B", "both"):
        print(f"\n--- Test B: Cold Routing / 50 apps ({client.name}) ---")
        cp_path = RESULTS_DIR / f"results_B_{args.model}.json"
        existing = None
        if args.resume and cp_path.exists():
            existing = json.loads(cp_path.read_text())
            print(f"  Resuming from checkpoint: {existing['completed']}/{len(queries)} done")

        results_b = run_test_b(client, queries, catalog, cp_path, existing)
        print_summary(results_b)


if __name__ == "__main__":
    main()
