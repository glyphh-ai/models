"""Tests for kill-chain correlation scoring."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorer import ChainEvent, score_chain, extract_chain_from_events, KILL_CHAIN_ORDER
from intent import analyze_event


# ── Chain scoring ─────────────────────────────────────────────────────────

class TestScoreChain:
    def test_empty_chain(self):
        chain = score_chain([])
        assert chain.stages_covered == 0
        assert chain.verdict == "MONITORING"

    def test_single_event(self):
        events = [ChainEvent(
            text="phishing email", tactic="initial_access",
            technique="phishing", source="email", urgency=7, severity="high",
        )]
        chain = score_chain(events)
        assert chain.stages_covered == 1
        assert chain.confidence < 0.30

    def test_full_killchain_5_stages(self):
        events = [
            ChainEvent(text="phishing", tactic="initial_access",
                       technique="phishing", source="email", urgency=7, severity="high", timestamp=900),
            ChainEvent(text="powershell", tactic="execution",
                       technique="powershell", source="endpoint", urgency=6, severity="medium", timestamp=2700),
            ChainEvent(text="lsass dump", tactic="credential_access",
                       technique="lsass_memory", source="endpoint", urgency=9, severity="critical", timestamp=5400),
            ChainEvent(text="psexec lateral", tactic="lateral_movement",
                       technique="remote_services", source="endpoint", urgency=8, severity="high", timestamp=8100),
            ChainEvent(text="exfil https", tactic="exfiltration",
                       technique="exfil_over_web", source="network", urgency=9, severity="critical", timestamp=13500),
        ]
        chain = score_chain(events)
        assert chain.stages_covered == 5
        assert chain.confidence >= 0.70
        assert chain.verdict == "ACTIVE INTRUSION"

    def test_three_stages_probable(self):
        events = [
            ChainEvent(text="phishing", tactic="initial_access",
                       technique="phishing", source="email", urgency=7, severity="high"),
            ChainEvent(text="powershell", tactic="execution",
                       technique="powershell", source="endpoint", urgency=6, severity="medium"),
            ChainEvent(text="lsass dump", tactic="credential_access",
                       technique="lsass_memory", source="endpoint", urgency=9, severity="critical"),
        ]
        chain = score_chain(events)
        assert chain.stages_covered == 3
        assert chain.confidence >= 0.30

    def test_noise_only_no_chain(self):
        events = [
            ChainEvent(text="dns query", tactic="none",
                       technique="none", source="dns", urgency=1, severity="info"),
            ChainEvent(text="backup job", tactic="none",
                       technique="none", source="unknown", urgency=1, severity="info"),
        ]
        chain = score_chain(events)
        assert chain.stages_covered == 0
        assert chain.verdict == "MONITORING"

    def test_stages_deduplicated(self):
        events = [
            ChainEvent(text="phishing 1", tactic="initial_access",
                       technique="phishing", source="email", urgency=7, severity="high"),
            ChainEvent(text="phishing 2", tactic="initial_access",
                       technique="phishing", source="email", urgency=7, severity="high"),
        ]
        chain = score_chain(events)
        assert chain.stages_covered == 1


# ── Integration with intent.py ────────────────────────────────────────────

class TestExtractChain:
    def test_demo_events(self):
        """Test the 5 attack events from the demo produce a valid chain."""
        attack_texts = [
            "User jsmith opened phishing email and downloaded macro-enabled document",
            "PowerShell spawned from Word.exe on workstation-088 executing encoded command",
            "lsass.exe memory dump via procdump.exe detected on workstation-088",
            "PsExec lateral movement from workstation-088 to fileserver01 using admin credentials",
            "Large outbound HTTPS transfer from fileserver01 to rare external IP 45.33.32.156",
        ]

        analyzed = []
        for i, text in enumerate(attack_texts):
            evt = analyze_event(text)
            evt["text"] = text
            evt["timestamp"] = (i + 1) * 1800
            analyzed.append(evt)

        chain = extract_chain_from_events(analyzed, min_urgency=3)
        assert chain.stages_covered >= 4
        assert chain.verdict in ("ACTIVE INTRUSION", "PROBABLE ATTACK")

    def test_noise_filtered_out(self):
        """Noise events with low urgency should not appear in chain."""
        events = [
            {"text": "DNS query for api.github.com", "mitre_tactic": "none", "urgency": 1, "severity": "info"},
            {"text": "Backup completed", "mitre_tactic": "none", "urgency": 1, "severity": "info"},
        ]
        chain = extract_chain_from_events(events, min_urgency=3)
        assert chain.stages_covered == 0
