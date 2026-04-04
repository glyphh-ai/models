#!/usr/bin/env python3
"""
Build the Pipedream Action Router model from auto-generated exemplars.

The build pipeline:
  1. Run discover.py to pull the Pipedream registry (or use cached exemplars)
  2. Convert JSONL exemplars → build records via encoder.entry_to_record()
  3. Package as .glyphh model file

Usage:
    python build.py                        # build from existing exemplars
    python build.py --discover             # re-discover then build
    python build.py --discover --limit 50  # discover limited set then build
    python build.py --output path/to.glyphh
"""

import argparse
import json
import sys
from pathlib import Path

from encoder import entry_to_record

MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data"
DEFAULT_OUTPUT = MODEL_DIR / "pipedream.glyphh"

JSONL_FILES = [
    "exemplars.jsonl",
]


def load_all_jsonl(data_dir: Path) -> list[dict]:
    entries = []
    for filename in JSONL_FILES:
        path = data_dir / filename
        if not path.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue
        count = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError:
                    print(f"  Warning: bad JSON at {filename}:{lineno}, skipping")
        print(f"  {filename}: {count} entries")
    return entries


def build(output_path: Path | None = None, run_discover: bool = False, limit: int = 0) -> None:
    output = output_path or DEFAULT_OUTPUT

    if run_discover:
        print("Running Pipedream registry discovery...\n")
        from discover import discover
        result = discover(limit=limit)
        if result["exemplars"] == 0:
            print("Error: Discovery returned 0 exemplars.")
            sys.exit(1)
        print()
    else:
        # Re-aggregate from per-app data if apps/ exists
        apps_dir = MODEL_DIR / "apps"
        if apps_dir.exists() and any(apps_dir.iterdir()):
            print("Re-aggregating from per-app data...")
            from discover import aggregate
            aggregate(apps_dir)
            print()

    print("Loading JSONL exemplar data...")
    entries = load_all_jsonl(DATA_DIR)
    if not entries:
        print("Error: No entries found. Run with --discover to pull from Pipedream registry.")
        sys.exit(1)

    print(f"\nConverting {len(entries)} entries to records...")
    records = [entry_to_record(e) for e in entries]

    # Show app distribution
    app_counts: dict[str, int] = {}
    for r in records:
        app = r["metadata"]["app_slug"]
        app_counts[app] = app_counts.get(app, 0) + 1
    print(f"\nApp distribution ({len(app_counts)} apps):")
    for app, count in sorted(app_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {app}: {count} actions")
    if len(app_counts) > 20:
        print(f"  ... and {len(app_counts) - 20} more")

    # Domain distribution
    domain_counts: dict[str, int] = {}
    for e in entries:
        d = e.get("domain", "none")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"\nDomain distribution:")
    for d, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {count}")

    print(f"\nTotal records: {len(records)}")
    print(f"\nReady to package as {output}")
    print("(Packaging requires the Glyphh runtime SDK)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Pipedream Action Router model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--discover", action="store_true",
                        help="Run registry discovery before building")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max apps to discover (0 = unlimited)")
    args = parser.parse_args()
    build(args.output, run_discover=args.discover, limit=args.limit)
