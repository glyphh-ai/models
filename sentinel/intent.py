"""
Intent extraction for the Sentinel Security model.

Pure deterministic MITRE ATT&CK feature extraction — no LLM, no API.
Classifies security event text across 4 dimensions:

  Tactic:    MITRE ATT&CK tactic (initial_access, execution, credential_access, ...)
  Technique: Specific attack technique (phishing, powershell, lsass_memory, ...)
  Source:    Event source type (endpoint, network, email, dns, ...)
  Temporal:  Urgency score (0-10) + severity level

Exports:
  analyze_event(text) -> dict   — full feature extraction for all 4 layers
  detect_tactic(text) -> str
  detect_technique(text) -> str
  detect_source(text) -> str
  compute_urgency(text) -> int
  compute_severity(text) -> str
  extract_tactic_signals(text) -> str
  extract_technique_signals(text) -> str
  extract_source_signals(text) -> str
"""

import re


# ---------------------------------------------------------------------------
# MITRE ATT&CK Tactic detection
# ---------------------------------------------------------------------------

_TACTIC_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("initial_access", [
        re.compile(r"phish(ing|ed)?", re.I),
        re.compile(r"spearphish", re.I),
        re.compile(r"macro\s+(document|enabled|attachment)", re.I),
        re.compile(r"open(ed|ing)?\s+(malicious|suspicious|phishing)\s+(email|attachment|link|document)", re.I),
        re.compile(r"(clicked|opened)\s+(a\s+)?(phishing|suspicious|malicious)", re.I),
        re.compile(r"drive.by\s+(download|compromise)", re.I),
        re.compile(r"watering\s+hole", re.I),
        re.compile(r"exploit\s+public.facing", re.I),
        re.compile(r"valid\s+accounts?\s+(compromised|stolen|leaked)", re.I),
        re.compile(r"trusted\s+relationship\s+abuse", re.I),
    ]),
    ("execution", [
        re.compile(r"powershell\s+(spawned|executed|launched|ran|command|script|\.exe)", re.I),
        re.compile(r"(spawned|launched|executed|ran)\s+(from|by)\s+\w+\.exe", re.I),
        re.compile(r"(cmd|command)\s*(\.exe|prompt|shell)\s+(spawned|executed|launched)", re.I),
        re.compile(r"wscript|cscript|mshta|regsvr32", re.I),
        re.compile(r"(scheduled|schtasks)\s+task\s+(created|modified|executed)", re.I),
        re.compile(r"wmi\s+(execution|command|process)", re.I),
        re.compile(r"(script|macro)\s+(execution|executed|ran|launched)", re.I),
        re.compile(r"(process|child)\s+spawn(ed)?", re.I),
        re.compile(r"word\.exe.*powershell|powershell.*word\.exe", re.I),
        re.compile(r"excel\.exe.*cmd|cmd.*excel\.exe", re.I),
    ]),
    ("persistence", [
        re.compile(r"registry\s+(run|autorun|startup)\s+key", re.I),
        re.compile(r"scheduled\s+task\s+(created|added|installed)", re.I),
        re.compile(r"(startup|autorun)\s+(entry|folder|item)\s+(created|added|modified)", re.I),
        re.compile(r"(service|daemon)\s+(installed|created|modified)", re.I),
        re.compile(r"boot\s+kit|bootkit", re.I),
        re.compile(r"(dll|binary)\s+(side.?load|hijack|planted)", re.I),
        re.compile(r"(implant|backdoor|webshell)\s+(installed|deployed|detected)", re.I),
    ]),
    ("privilege_escalation", [
        re.compile(r"privilege\s+escalat", re.I),
        re.compile(r"(uac|user\s+account\s+control)\s+bypass", re.I),
        re.compile(r"(token|access)\s+(manipulation|theft|impersonation)", re.I),
        re.compile(r"(sudo|su|runas)\s+(abuse|exploit)", re.I),
        re.compile(r"(setuid|setgid|suid)\s+(binary|bit)", re.I),
        re.compile(r"exploit.*(kernel|driver|local)", re.I),
    ]),
    ("defense_evasion", [
        re.compile(r"(disable|tamper|stop)\w*\s+(antivirus|av|edr|defender|security|logging|sysmon)", re.I),
        re.compile(r"(clear|delete|wipe)\w*\s+(event\s+)?logs?", re.I),
        re.compile(r"(obfuscat|encod|pack|crypt)\w+\s+(payload|script|binary|malware)", re.I),
        re.compile(r"(process|dll)\s+(inject|hollow)", re.I),
        re.compile(r"(masquerad|disguise)\w*\s+(as|process|file)", re.I),
        re.compile(r"timestomp", re.I),
        re.compile(r"(indicator|artifact)\s+remov", re.I),
    ]),
    ("credential_access", [
        re.compile(r"lsass(\.exe)?\s+(memory|dump|access|read)", re.I),
        re.compile(r"(credential|password|hash)\s+(dump|harvest|extract|steal|theft|spray)", re.I),
        re.compile(r"(procdump|mimikatz|sekurlsa|hashcat|lazagne)", re.I),
        re.compile(r"(brute.?force|password\s+spray|credential\s+stuff)", re.I),
        re.compile(r"(kerberoast|asreproast|golden\s+ticket|silver\s+ticket)", re.I),
        re.compile(r"(ntlm|sam|ntds)\s+(hash|dump|extract|relay)", re.I),
        re.compile(r"(keylog|input\s+capture)", re.I),
    ]),
    ("discovery", [
        re.compile(r"(network|port|host|service)\s+scan", re.I),
        re.compile(r"(active\s+directory|ad|ldap)\s+(enumerat|query|recon)", re.I),
        re.compile(r"(file|share|permission|group|user)\s+(enumerat|discover|scan)", re.I),
        re.compile(r"(whoami|ipconfig|ifconfig|netstat|tasklist|systeminfo)", re.I),
        re.compile(r"(nmap|masscan|ping\s+sweep|arp\s+scan)", re.I),
        re.compile(r"(bloodhound|sharphound|adexplorer)", re.I),
    ]),
    ("lateral_movement", [
        re.compile(r"(psexec|paexec|smbexec|wmiexec)", re.I),
        re.compile(r"lateral\s+mov", re.I),
        re.compile(r"(remote|rdp|ssh|smb|winrm)\s+(session|connection|login|access|execution)\s+(to|from|on|established)", re.I),
        re.compile(r"(moved|spread|pivot)\w*\s+(to|lateral|across)\s+\w+", re.I),
        re.compile(r"pass.the.(hash|ticket)", re.I),
        re.compile(r"(admin\$?|c\$|ipc\$)\s+(share|access|connect)", re.I),
    ]),
    ("collection", [
        re.compile(r"(data|file|document)\s+(collect|gather|stage|compress|archive|zip|rar)", re.I),
        re.compile(r"(screen\s*capture|screenshot|clipboard)\s+(taken|captured|collected)", re.I),
        re.compile(r"(email|mailbox)\s+(collect|harvest|export|download)", re.I),
        re.compile(r"(database|db)\s+(dump|export|extract)", re.I),
    ]),
    ("command_and_control", [
        re.compile(r"(c2|c&c|command.and.control|beacon)", re.I),
        re.compile(r"(cobalt\s*strike|metasploit|empire|covenant|sliver)", re.I),
        re.compile(r"(dns\s+tunnel|domain\s+front|domain\s+generation)", re.I),
        re.compile(r"(encrypted\s+channel|covert\s+channel)", re.I),
        re.compile(r"(callback|heartbeat|check.in)\s+(to|from|detected)", re.I),
    ]),
    ("exfiltration", [
        re.compile(r"exfiltrat", re.I),
        re.compile(r"(data|file)\s+(transfer|upload|exfil|sent|moved)\s+(to|out|external)", re.I),
        re.compile(r"(large|unusual|suspicious)\s+(outbound|external|egress)\s+(transfer|traffic|connection|https?)", re.I),
        re.compile(r"(outbound|egress)\s+(to|connection)\s+(rare|unknown|suspicious|external|foreign)", re.I),
        re.compile(r"(upload|transfer)\s+to\s+(cloud|external|rare|unknown|foreign)", re.I),
        re.compile(r"(dns|icmp|https?)\s+(exfil|tunnel|covert)", re.I),
    ]),
    ("impact", [
        re.compile(r"(ransomware|encrypt|wiper|destruct)", re.I),
        re.compile(r"(defac|vandal|destroy)\w*\s+(website|data|files?|system)", re.I),
        re.compile(r"(denial.of.service|dos|ddos)", re.I),
        re.compile(r"(data|disk)\s+(wipe|destroy|encrypt|corrupt)", re.I),
    ]),
    ("reconnaissance", [
        re.compile(r"(recon|reconnais|osint)", re.I),
        re.compile(r"(scan|probe|enumerate)\w*\s+(external|public|internet|perimeter)", re.I),
        re.compile(r"(shodan|censys|whois|dns\s+lookup)", re.I),
        re.compile(r"(social\s+engineer|spear\s*phish\w*\s+recon)", re.I),
    ]),
]


def detect_tactic(text: str) -> str:
    """Detect MITRE ATT&CK tactic from event text."""
    text_lower = text.lower()
    for tactic, patterns in _TACTIC_PATTERNS:
        for pat in patterns:
            if pat.search(text_lower):
                return tactic
    return "none"


# ---------------------------------------------------------------------------
# MITRE ATT&CK Technique detection
# ---------------------------------------------------------------------------

_TECHNIQUE_MAP: dict[str, list[re.Pattern]] = {
    "phishing": [
        re.compile(r"phish(ing|ed)?\s+(email|message|link|campaign)", re.I),
        re.compile(r"(suspicious|malicious)\s+email", re.I),
    ],
    "spearphishing_attachment": [
        re.compile(r"spearphish", re.I),
        re.compile(r"macro\s+(document|attachment|enabled)", re.I),
        re.compile(r"(malicious|weaponized)\s+(attachment|document|pdf|docx?|xlsx?)", re.I),
    ],
    "powershell": [
        re.compile(r"powershell", re.I),
    ],
    "command_scripting": [
        re.compile(r"(cmd|command)\s*(\.exe|prompt|shell)", re.I),
        re.compile(r"(wscript|cscript|mshta|regsvr32)", re.I),
        re.compile(r"(batch|vbs|jscript)\s+(script|file|execution)", re.I),
    ],
    "scheduled_task": [
        re.compile(r"(scheduled|schtasks)\s+task", re.I),
        re.compile(r"(cron|at)\s+(job|task|entry)", re.I),
    ],
    "registry_run_key": [
        re.compile(r"registry\s+(run|autorun|startup)\s+key", re.I),
        re.compile(r"HKLM.*Run|HKCU.*Run", re.I),
    ],
    "os_credential_dumping": [
        re.compile(r"(credential|password|hash)\s+(dump|harvest|extract)", re.I),
        re.compile(r"(mimikatz|sekurlsa|hashcat|lazagne)", re.I),
    ],
    "lsass_memory": [
        re.compile(r"lsass", re.I),
        re.compile(r"procdump.*lsass|lsass.*procdump", re.I),
        re.compile(r"(memory\s+dump|dump\s+memory)\s+.*lsass", re.I),
    ],
    "remote_services": [
        re.compile(r"(psexec|paexec|smbexec|wmiexec)", re.I),
        re.compile(r"(rdp|ssh|winrm|smb)\s+(connection|session|login)", re.I),
    ],
    "smb_shares": [
        re.compile(r"(smb|cifs)\s+(share|access|connect)", re.I),
        re.compile(r"(admin\$?|c\$|ipc\$)", re.I),
    ],
    "data_staged": [
        re.compile(r"(data|file)\s+(staged?|compress|archive|zip|rar|pack)", re.I),
    ],
    "data_encrypted": [
        re.compile(r"(data|file)\s+encrypt", re.I),
        re.compile(r"(ransomware|ransom\s+note)", re.I),
    ],
    "exfil_over_c2": [
        re.compile(r"exfil\w*\s+(over|via|through)\s+(c2|c&c|beacon|command)", re.I),
    ],
    "exfil_over_web": [
        re.compile(r"exfil\w*\s+(over|via|through)\s+(https?|web|cloud)", re.I),
        re.compile(r"(outbound|egress)\s+(https?|ssl|tls)\s+(to|transfer|connection)", re.I),
        re.compile(r"(large|unusual)\s+outbound\s+https?", re.I),
    ],
    "data_destruction": [
        re.compile(r"(data|disk|file)\s+(wipe|destroy|corrupt|delete)", re.I),
        re.compile(r"(wiper|destroy)\s+(malware|payload|detected)", re.I),
    ],
}


def detect_technique(text: str) -> str:
    """Detect MITRE ATT&CK technique from event text."""
    text_lower = text.lower()
    for technique, patterns in _TECHNIQUE_MAP.items():
        for pat in patterns:
            if pat.search(text_lower):
                return technique
    return "none"


# ---------------------------------------------------------------------------
# Event source detection
# ---------------------------------------------------------------------------

_SOURCE_INDICATORS: dict[str, list[re.Pattern]] = {
    "endpoint": [
        re.compile(r"(process|pid|exe|binary|dll|service|registry|sysmon|edr|endpoint|host|workstation|server\d)", re.I),
        re.compile(r"(lsass|cmd|powershell|word|excel|outlook)\.exe", re.I),
        re.compile(r"(memory|disk|file\s*system|volume)", re.I),
    ],
    "network": [
        re.compile(r"(network|packet|traffic|flow|netflow|pcap|connection|socket|tcp|udp|port\s+\d)", re.I),
        re.compile(r"(firewall|ids|ips|nids|proxy|gateway)\s+(log|alert|event|block)", re.I),
        re.compile(r"(inbound|outbound|ingress|egress)\s+(traffic|connection|packet)", re.I),
    ],
    "email": [
        re.compile(r"(email|mail|smtp|imap|exchange|o365|outlook)\s+(message|delivery|received|sent|blocked)", re.I),
        re.compile(r"(phishing|spam|malicious)\s+(email|message|mail)", re.I),
        re.compile(r"(attachment|link)\s+(in|from)\s+(email|message)", re.I),
    ],
    "dns": [
        re.compile(r"(dns|domain|resolve|lookup|query)\s+(query|request|response|lookup|resolution)", re.I),
        re.compile(r"(nxdomain|dns\s+tunnel|domain\s+generation|dga)", re.I),
    ],
    "identity": [
        re.compile(r"(login|logon|auth|sso|mfa|2fa|ldap|kerberos|active\s+directory)\s+(success|fail|event|attempt|log)", re.I),
        re.compile(r"(user|account|credential)\s+(created|disabled|locked|modified|deleted|login)", re.I),
    ],
    "cloud": [
        re.compile(r"(aws|azure|gcp|cloud|s3|ec2|lambda|blob|iam)\s+(event|log|api|activity)", re.I),
        re.compile(r"(cloudtrail|cloudwatch|azure\s+monitor|stackdriver)", re.I),
    ],
    "firewall": [
        re.compile(r"(firewall|waf|ngfw)\s+(allow|deny|block|drop|alert|log|rule)", re.I),
        re.compile(r"(acl|rule)\s+(triggered|matched|blocked|allowed)", re.I),
    ],
}


def detect_source(text: str) -> str:
    """Detect event source type from text."""
    text_lower = text.lower()
    for source, patterns in _SOURCE_INDICATORS.items():
        for pat in patterns:
            if pat.search(text_lower):
                return source
    return "unknown"


# ---------------------------------------------------------------------------
# Urgency & severity scoring
# ---------------------------------------------------------------------------

_URGENCY_KEYWORDS: dict[str, int] = {
    # Critical indicators (8-10)
    "ransomware": 10, "exfiltration": 9, "exfiltrat": 9,
    "data breach": 10, "active intrusion": 10,
    "credential dump": 9, "lsass": 9, "mimikatz": 10,
    "lateral movement": 8, "psexec": 8, "wmiexec": 8,
    "privilege escalation": 8, "backdoor": 9,
    "c2 beacon": 9, "cobalt strike": 10,
    # High indicators (5-7)
    "powershell": 6, "command shell": 5, "suspicious process": 6,
    "phishing": 7, "spearphishing": 8,
    "malicious": 7, "exploit": 7, "vulnerability": 5,
    "brute force": 6, "password spray": 6,
    "unusual": 5, "anomalous": 5, "suspicious": 5,
    "rare external": 7, "rare ip": 7, "unknown ip": 6,
    "outbound": 5, "large transfer": 7,
    # Medium indicators (3-4)
    "failed login": 3, "account lockout": 4,
    "port scan": 4, "network scan": 4,
    "policy violation": 3, "configuration change": 3,
    # Low indicators (1-2)
    "info": 1, "audit": 1, "routine": 1,
    "update": 1, "patch": 1, "backup": 1,
    "dhcp": 1, "dns query": 1, "health check": 1,
}

_SEVERITY_MAP = {
    (0, 2): "info",
    (3, 4): "low",
    (5, 6): "medium",
    (7, 8): "high",
    (9, 10): "critical",
}


def compute_urgency(text: str) -> int:
    """Compute urgency score (0-10) from event text."""
    text_lower = text.lower()
    max_urgency = 0
    for keyword, score in _URGENCY_KEYWORDS.items():
        if keyword in text_lower:
            max_urgency = max(max_urgency, score)
    return max_urgency


def compute_severity(text: str) -> str:
    """Compute severity level from event text."""
    urgency = compute_urgency(text)
    for (lo, hi), severity in _SEVERITY_MAP.items():
        if lo <= urgency <= hi:
            return severity
    return "info"


# ---------------------------------------------------------------------------
# Signal extraction (bag-of-words for each layer)
# ---------------------------------------------------------------------------

_TACTIC_SIGNAL_WORDS = {
    "phishing", "spearphishing", "macro", "download", "execution",
    "powershell", "cmd", "wscript", "persistence", "autorun",
    "registry", "scheduled", "privilege", "escalation", "uac",
    "evasion", "obfuscation", "disable", "credential", "dump",
    "harvest", "lsass", "mimikatz", "kerberoast", "discovery",
    "enumerate", "scan", "lateral", "movement", "psexec", "rdp",
    "smb", "winrm", "collection", "stage", "archive", "c2",
    "beacon", "exfiltration", "transfer", "outbound", "ransomware",
    "encrypt", "wiper", "destroy", "reconnaissance", "osint",
    "remote", "admin", "spread", "pivot",
}

_TECHNIQUE_SIGNAL_WORDS = {
    "powershell", "cmd", "wscript", "cscript", "mshta", "regsvr32",
    "procdump", "mimikatz", "sekurlsa", "hashcat", "lazagne",
    "psexec", "paexec", "smbexec", "wmiexec", "bloodhound",
    "cobalt", "strike", "metasploit", "empire", "sliver",
    "nmap", "masscan", "lsass", "ntlm", "kerberos", "sam",
    "ntds", "schtasks", "cron", "registry", "hklm", "hkcu",
    "rdp", "ssh", "smb", "winrm", "dns", "tunnel",
    "base64", "encoded", "obfuscated", "packed", "encrypted",
    "macro", "vba", "document", "attachment", "payload",
}

_SOURCE_SIGNAL_WORDS = {
    "process", "pid", "exe", "binary", "dll", "service", "sysmon",
    "edr", "endpoint", "host", "server", "workstation",
    "network", "packet", "traffic", "flow", "firewall", "proxy",
    "email", "smtp", "exchange", "outlook", "attachment",
    "dns", "domain", "resolve", "query", "nxdomain",
    "login", "logon", "auth", "sso", "mfa", "ldap", "kerberos",
    "aws", "azure", "gcp", "cloud", "iam",
}


def _extract_signal_words(text: str, signal_set: set[str]) -> str:
    """Extract matching signal words from text."""
    text_lower = text.lower()
    words = re.findall(r'[a-z_]+', text_lower)
    found = []
    for w in words:
        if w in signal_set:
            found.append(w)
    # Also check multi-word signals
    for signal in signal_set:
        if " " in signal and signal in text_lower and signal not in found:
            found.append(signal)
    return " ".join(sorted(set(found)))


def extract_tactic_signals(text: str) -> str:
    return _extract_signal_words(text, _TACTIC_SIGNAL_WORDS)


def extract_technique_signals(text: str) -> str:
    return _extract_signal_words(text, _TECHNIQUE_SIGNAL_WORDS)


def extract_source_signals(text: str) -> str:
    return _extract_signal_words(text, _SOURCE_SIGNAL_WORDS)


# ---------------------------------------------------------------------------
# Full analysis — one call, all layers
# ---------------------------------------------------------------------------

def analyze_event(text: str) -> dict:
    """Extract MITRE ATT&CK features from security event text.

    Returns a dict ready for the encoder's concept mapping:
      {
        "mitre_tactic": "lateral_movement",
        "tactic_signals": "psexec remote admin lateral movement",
        "technique_id": "remote_services",
        "technique_signals": "psexec smb",
        "event_source": "endpoint",
        "source_signals": "process edr host",
        "urgency": 8,
        "severity": "high",
      }
    """
    return {
        "mitre_tactic": detect_tactic(text),
        "tactic_signals": extract_tactic_signals(text),
        "technique_id": detect_technique(text),
        "technique_signals": extract_technique_signals(text),
        "event_source": detect_source(text),
        "source_signals": extract_source_signals(text),
        "urgency": compute_urgency(text),
        "severity": compute_severity(text),
    }
