"""
Kill-chain correlation scorer for the Sentinel Security model.

Takes security events (analyzed by intent.py) and scores them for
kill-chain progression. The DreamLoop discovers connections; this
module scores and ranks the chains it finds.

The MITRE ATT&CK kill chain has a natural ordering:
  reconnaissance -> initial_access -> execution -> persistence ->
  privilege_escalation -> credential_access -> discovery ->
  lateral_movement -> collection -> exfiltration -> impact

A chain is a sequence of events whose tactics follow this ordering.
Higher scores for longer chains and tighter temporal clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Kill chain ordering (MITRE ATT&CK)
# ---------------------------------------------------------------------------

KILL_CHAIN_ORDER = [
    "reconnaissance",
    "resource_development",
    "initial_access",
    "execution",
    "persistence",
    "privilege_escalation",
    "defense_evasion",
    "credential_access",
    "discovery",
    "lateral_movement",
    "collection",
    "command_and_control",
    "exfiltration",
    "impact",
]

_TACTIC_INDEX = {t: i for i, t in enumerate(KILL_CHAIN_ORDER)}
TOTAL_STAGES = len(KILL_CHAIN_ORDER)


# ---------------------------------------------------------------------------
# Chain event + result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChainEvent:
    """A security event positioned in a kill chain."""
    text: str
    tactic: str
    technique: str
    source: str
    urgency: int
    severity: str
    timestamp: float = 0.0  # simulated or real
    stage_index: int = -1

    def __post_init__(self):
        if self.stage_index < 0:
            self.stage_index = _TACTIC_INDEX.get(self.tactic, -1)


@dataclass
class KillChain:
    """A detected kill chain — ordered sequence of correlated events."""
    events: list[ChainEvent] = field(default_factory=list)
    confidence: float = 0.0
    stages_covered: int = 0
    verdict: str = "MONITORING"

    @property
    def stage_names(self) -> list[str]:
        return [e.tactic for e in self.events]

    def summary(self) -> str:
        lines = []
        for i, e in enumerate(self.events, 1):
            t_str = f"T+{e.timestamp/3600:.1f}h" if e.timestamp else f"#{i}"
            lines.append(f"  {i}. [{t_str}] {e.tactic:20s} — {e.text[:60]}")
        lines.append(f"  Chain confidence: {self.confidence:.2f}")
        lines.append(f"  Stages covered: {self.stages_covered}/{TOTAL_STAGES}")
        lines.append(f"  Verdict: {self.verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chain detection
# ---------------------------------------------------------------------------

def score_chain(events: list[ChainEvent]) -> KillChain:
    """Score a sequence of events for kill-chain progression.

    Events are sorted by their MITRE ATT&CK stage order, then scored:
      - Stages covered: how many distinct stages are present
      - Progression: are stages in the correct order?
      - Urgency: average urgency across chain events
      - Length bonus: longer chains are more confident

    Returns a KillChain with confidence score and verdict.
    """
    # Filter to events that map to a known tactic
    chain_events = [e for e in events if e.stage_index >= 0]
    if not chain_events:
        return KillChain()

    # Sort by stage order (not by timestamp — the chain is structural)
    chain_events.sort(key=lambda e: e.stage_index)

    # Deduplicate by tactic (keep first per stage)
    seen_stages: set[str] = set()
    deduped: list[ChainEvent] = []
    for e in chain_events:
        if e.tactic not in seen_stages:
            seen_stages.add(e.tactic)
            deduped.append(e)

    stages_covered = len(deduped)

    # Progression score: how well do stages follow the kill chain order?
    if stages_covered >= 2:
        correct_order = 0
        for i in range(len(deduped) - 1):
            if deduped[i].stage_index < deduped[i + 1].stage_index:
                correct_order += 1
        progression = correct_order / (stages_covered - 1)
    else:
        progression = 0.0

    # Average urgency (normalized 0-1)
    avg_urgency = sum(e.urgency for e in deduped) / (len(deduped) * 10)

    # Confidence: weighted combination
    # - Stage coverage matters most (3+ stages is significant)
    # - Progression validates it's a real chain
    # - Urgency confirms severity
    stage_score = min(1.0, stages_covered / 5)  # 5 stages = 1.0
    confidence = (
        stage_score * 0.50
        + progression * 0.30
        + avg_urgency * 0.20
    )

    # Verdict
    if confidence >= 0.70 and stages_covered >= 4:
        verdict = "ACTIVE INTRUSION"
    elif confidence >= 0.50 and stages_covered >= 3:
        verdict = "PROBABLE ATTACK"
    elif confidence >= 0.30 and stages_covered >= 2:
        verdict = "SUSPICIOUS ACTIVITY"
    else:
        verdict = "MONITORING"

    return KillChain(
        events=deduped,
        confidence=round(confidence, 2),
        stages_covered=stages_covered,
        verdict=verdict,
    )


def extract_chain_from_events(
    analyzed_events: list[dict],
    min_urgency: int = 3,
) -> KillChain:
    """Extract a kill chain from a list of analyzed event dicts.

    Filters noise (urgency < min_urgency) and scores the remaining
    high-signal events for chain progression.

    Args:
        analyzed_events: List of dicts from analyze_event() + text + timestamp.
        min_urgency: Minimum urgency to include in chain analysis.
    """
    chain_events = []
    for evt in analyzed_events:
        urgency = evt.get("urgency", 0)
        tactic = evt.get("mitre_tactic", "none")

        if urgency < min_urgency or tactic == "none":
            continue

        chain_events.append(ChainEvent(
            text=evt.get("text", ""),
            tactic=tactic,
            technique=evt.get("technique_id", "none"),
            source=evt.get("event_source", "unknown"),
            urgency=urgency,
            severity=evt.get("severity", "info"),
            timestamp=evt.get("timestamp", 0.0),
        ))

    return score_chain(chain_events)
