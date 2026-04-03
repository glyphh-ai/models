"""Tests for Sentinel intent extraction (MITRE ATT&CK feature extraction)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent import (
    analyze_event,
    detect_tactic,
    detect_technique,
    detect_source,
    compute_urgency,
    compute_severity,
)


# ── Tactic detection ──────────────────────────────────────────────────────

class TestDetectTactic:
    def test_initial_access_phishing(self):
        assert detect_tactic("User opened phishing email with macro document") == "initial_access"

    def test_execution_powershell(self):
        assert detect_tactic("PowerShell spawned from Word.exe executing encoded command") == "execution"

    def test_credential_access_lsass(self):
        assert detect_tactic("lsass.exe memory dump via procdump detected") == "credential_access"

    def test_lateral_movement_psexec(self):
        assert detect_tactic("PsExec lateral movement to fileserver01") == "lateral_movement"

    def test_exfiltration_outbound(self):
        assert detect_tactic("Large outbound HTTPS transfer to rare external IP") == "exfiltration"

    def test_persistence_registry(self):
        assert detect_tactic("Registry run key added for persistence") == "persistence"

    def test_discovery_scan(self):
        assert detect_tactic("Network scan detected from internal host") == "discovery"

    def test_impact_ransomware(self):
        assert detect_tactic("Ransomware encryption detected on file shares") == "impact"

    def test_noise_dns(self):
        assert detect_tactic("DNS query for api.github.com") == "none"

    def test_noise_update(self):
        assert detect_tactic("Windows Update check completed") == "none"

    def test_noise_backup(self):
        assert detect_tactic("Backup job completed successfully") == "none"


# ── Technique detection ───────────────────────────────────────────────────

class TestDetectTechnique:
    def test_powershell(self):
        assert detect_technique("PowerShell spawned from Word.exe") == "powershell"

    def test_lsass_memory(self):
        assert detect_technique("lsass.exe memory dump via procdump") == "lsass_memory"

    def test_remote_services_psexec(self):
        assert detect_technique("PsExec connection to fileserver01") == "remote_services"

    def test_phishing(self):
        assert detect_technique("Suspicious email with phishing link") == "phishing"

    def test_spearphishing_attachment(self):
        assert detect_technique("Macro document attachment opened") == "spearphishing_attachment"

    def test_none_for_noise(self):
        assert detect_technique("DNS query for api.github.com") == "none"


# ── Source detection ──────────────────────────────────────────────────────

class TestDetectSource:
    def test_endpoint(self):
        assert detect_source("Process spawned on workstation-088") == "endpoint"

    def test_network(self):
        assert detect_source("Firewall blocked inbound traffic from 1.2.3.4") == "network"

    def test_email(self):
        assert detect_source("Phishing email received via Exchange") == "email"

    def test_dns(self):
        assert detect_source("DNS query for suspicious domain") == "dns"

    def test_identity(self):
        assert detect_source("Failed login attempt for admin account via SSO") == "identity"

    def test_unknown(self):
        assert detect_source("Something happened") == "unknown"


# ── Urgency & severity ───────────────────────────────────────────────────

class TestUrgencySeverity:
    def test_critical_ransomware(self):
        assert compute_urgency("Ransomware detected encrypting files") == 10
        assert compute_severity("Ransomware detected encrypting files") == "critical"

    def test_high_lsass(self):
        assert compute_urgency("lsass memory dump detected") >= 9
        assert compute_severity("lsass memory dump detected") == "critical"

    def test_medium_suspicious(self):
        u = compute_urgency("Suspicious process detected")
        assert 4 <= u <= 7

    def test_low_noise(self):
        assert compute_urgency("Routine DNS query for google.com") <= 2

    def test_info_update(self):
        assert compute_urgency("Windows Update check completed") <= 2
        assert compute_severity("Windows Update check completed") == "info"


# ── Full analysis ─────────────────────────────────────────────────────────

class TestAnalyzeEvent:
    def test_phishing_event(self):
        result = analyze_event("User opened phishing email with macro-enabled document")
        assert result["mitre_tactic"] == "initial_access"
        assert result["event_source"] == "email"
        assert result["urgency"] >= 7

    def test_powershell_from_word(self):
        result = analyze_event("PowerShell spawned from Word.exe executing encoded command")
        assert result["mitre_tactic"] == "execution"
        assert result["technique_id"] == "powershell"
        assert result["event_source"] == "endpoint"

    def test_lsass_dump(self):
        result = analyze_event("lsass.exe memory dump via procdump.exe")
        assert result["mitre_tactic"] == "credential_access"
        assert result["technique_id"] == "lsass_memory"
        assert result["urgency"] >= 9

    def test_lateral_psexec(self):
        result = analyze_event("PsExec lateral movement to fileserver01")
        assert result["mitre_tactic"] == "lateral_movement"
        assert result["technique_id"] == "remote_services"

    def test_exfil_https(self):
        result = analyze_event("Large outbound HTTPS transfer to rare external IP")
        assert result["mitre_tactic"] == "exfiltration"
        assert result["urgency"] >= 5

    def test_noise_event(self):
        result = analyze_event("DNS query for api.github.com from workstation-042")
        assert result["mitre_tactic"] == "none"
        assert result["urgency"] <= 2
