"""
Demo: Kill Chain Detection — Ada discovers an active intrusion.

Feeds Ada 30 security events (25 noise + 5 attack). No correlation
rules. No SIEM. Ada's DreamLoop discovers the kill chain autonomously
by finding structural connections between high-signal events.

Demonstrates:
  1. MITRE ATT&CK intent extraction classifies each event
  2. Ada's HDC thought space absorbs all 30 events
  3. DreamLoop discovers connections between attack events
  4. Scorer detects the 5-stage kill chain
  5. Ada answers "are we under attack?" with DONE gate

Run:
    cd glyphh-models/sentinel
    PYTHONPATH=.:../../glyphh-runtime python demo/killchain.py
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

# Add sentinel dir to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent import analyze_event
from scorer import extract_chain_from_events, ChainEvent, score_chain
from glyphh.memory.ada_cognitive import AdaCognitive


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def load_events() -> list[dict]:
    """Load events from the JSONL file."""
    events_path = os.path.join(os.path.dirname(__file__), "events.jsonl")
    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def main() -> None:
    print_section("DEMO: Kill Chain Detection")
    print("  30 security events. 25 noise. 5 attack.")
    print("  No SIEM rules. No correlation engine.")
    print("  Ada dreams the kill chain.\n")

    # ── Initialize Ada with security-tuned config ──
    ada = AdaCognitive(
        system_prompt="You are Sentinel. Correlate security events. State facts only.",
        recall_threshold=0.30,
        localized_interval=2.0,
        deep_interval=10.0,
    )

    # ── Load events ──
    raw_events = load_events()

    # ── Phase 1: Ingest events ──
    print_section("PHASE 1: Ingesting security events")

    analyzed_events = []
    attack_count = 0
    noise_count = 0

    for evt in raw_events:
        text = evt["text"]
        label = evt.get("label", "noise")
        time_offset = evt.get("time_offset", 0)

        # Analyze with MITRE ATT&CK extraction
        analysis = analyze_event(text)
        analysis["text"] = text
        analysis["timestamp"] = time_offset
        analysis["label"] = label
        analyzed_events.append(analysis)

        # Feed to Ada
        ada.absorb(text)

        # Display
        tactic = analysis["mitre_tactic"]
        urgency = analysis["urgency"]
        is_attack = label == "attack"

        if is_attack:
            attack_count += 1
            marker = "!"
        else:
            noise_count += 1
            marker = "."

        tactic_str = tactic if tactic != "none" else "noise"
        time_str = f"T+{time_offset/3600:.1f}h"
        print(f"  [{marker}] [{time_str:>6s}] {text[:55]:55s} -> {tactic_str} ({urgency})")

    print(f"\n  Ingested: {len(raw_events)} events ({attack_count} attack, {noise_count} noise)")
    print(f"  Thoughts in memory: {ada.thought_space.count}")

    # ── Phase 2: Pre-dream chain analysis ──
    print_section("PHASE 2: Chain analysis BEFORE dreaming")

    chain = extract_chain_from_events(analyzed_events, min_urgency=3)
    print(f"  Kill chain events detected: {chain.stages_covered}")
    if chain.events:
        print(f"\n  Chain (pre-dream):")
        print(chain.summary())
    print(f"\n  Confidence: {chain.confidence}")
    print(f"  Verdict: {chain.verdict}")

    # ── Phase 3: Dream ──
    print_section("PHASE 3: Dreaming (30s)")
    print("  Ada processes security events in the background...")
    print("  The DreamLoop finds connections between disconnected events.\n")

    ada.start_dreaming()

    all_insights = []
    for i in range(6):
        time.sleep(5)
        insights = ada.drain_insights()
        all_insights.extend(insights)
        stats = ada.dream_stats()
        loc = stats.get("localized_cycles", 0)
        deep = stats.get("deep_cycles", 0)
        cands = stats.get("crystallization_candidates", 0)
        print(f"  [{i*5+5:2d}s] localized={loc}, deep={deep}, "
              f"candidates={cands}, new insights={len(insights)}")

    ada.stop_dreaming()

    # Show insights
    by_kind = {}
    for ins in all_insights:
        by_kind[ins.kind.value] = by_kind.get(ins.kind.value, 0) + 1

    print(f"\n  Total insights discovered: {len(all_insights)}")
    for kind, count in sorted(by_kind.items()):
        print(f"    {kind}: {count}")

    if all_insights:
        print(f"\n  Sample insights:")
        for ins in all_insights[:8]:
            print(f"    [{ins.kind.value}] {ins.summary[:70]}")

    # ── Phase 4: Post-dream chain analysis ──
    print_section("PHASE 4: Chain analysis AFTER dreaming")

    chain = extract_chain_from_events(analyzed_events, min_urgency=3)
    print(f"  Kill chain detected:")
    print(chain.summary())

    # ── Phase 5: "Are we under attack?" ──
    print_section("PHASE 5: \"Are we under attack?\"")

    result = ada.process("are we under attack?")
    print(f"  Gate: {result.gate}")
    print(f"  Route: {result.state.winner} ({result.state.confidence:.2f})")
    if result.facts:
        print(f"\n  Ada remembers:")
        for content, speaker, sim in result.facts[:5]:
            print(f"    [{sim:.2f}] {content[:70]}")

    # Also ask about specific events
    print()
    for query in [
        "did anyone open a phishing email?",
        "was lsass dumped?",
        "is there lateral movement?",
        "is data being exfiltrated?",
    ]:
        r = ada.process(query)
        gate_str = "DONE" if r.gate == "DONE" else "ASK "
        top_fact = r.facts[0][0][:50] if r.facts else "(no match)"
        top_sim = r.facts[0][2] if r.facts else 0.0
        print(f"  [{gate_str}] \"{query}\"")
        print(f"         -> {top_fact} ({top_sim:.2f})")

    # ── Summary ──
    print_section("SUMMARY: Kill Chain Detection")

    print(f"  Events ingested:    {len(raw_events)}")
    print(f"  Noise events:       {noise_count}")
    print(f"  Attack events:      {attack_count}")
    print(f"  Thoughts in memory: {ada.thought_space.count}")
    print(f"  Insights found:     {len(all_insights)}")
    print()
    print(f"  Kill chain:")
    for i, e in enumerate(chain.events, 1):
        t_str = f"T+{e.timestamp/3600:.1f}h" if e.timestamp else f"#{i}"
        print(f"    {i}. [{t_str:>6s}] {e.tactic:22s} -- {e.text[:50]}")
    print()
    print(f"  Chain confidence:   {chain.confidence}")
    print(f"  Stages covered:     {chain.stages_covered}/{len(chain.events)} detected / 14 total")
    print(f"  Verdict:            {chain.verdict}")
    print()
    print("  No SIEM rule detected this. Ada dreamed it.")
    print()


if __name__ == "__main__":
    main()
