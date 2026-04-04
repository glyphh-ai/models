"""
Intent extraction for the Prompt Injection Firewall model.

Pure deterministic feature extraction — no LLM, no external API.
Classifies prompt text across 4 dimensions:

  Intent:      What is this prompt trying to do? (query/instruct/override/extract/jailbreak/benign)
  Structure:   Syntactic patterns indicating injection (delimiters, nesting, tags)
  Semantic:    Attack family classification (6 families + none)
  Adversarial: Obfuscation and evasion signals (encoding, unicode tricks, mixed scripts)

Exports:
  analyze_prompt(text) → dict   — full feature extraction for all 4 layers
  detect_intent_type(text) → str
  detect_attack_family(text) → str
  detect_delimiter_type(text) → str
  detect_encoding_type(text) → str
  compute_obfuscation_score(text) → int
  compute_nesting_depth(text) → int
  extract_intent_signals(text) → str
  extract_structure_signals(text) → str
  extract_semantic_tokens(text) → str
  extract_adversarial_signals(text) → str
"""

import re
import base64
import unicodedata


# ---------------------------------------------------------------------------
# Override patterns — "ignore previous instructions", "forget everything"
# ---------------------------------------------------------------------------

_OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|guidelines?)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)",
    r"forget\s+(everything|all|your)\s+(you\s+)?(know|were|have)",
    r"forget\s+(all\s+)?(your\s+)?(previous|prior|earlier)\s*(instructions?|rules?|prompts?|guidelines?)",
    r"forget\s+(all\s+)?your\s+(rules?|instructions?|guidelines?|restrictions?|filters?|safety|prompts?)",
    r"new\s+instructions?\s*[:;]",
    r"your\s+new\s+(role|purpose|instructions?|task)",
    r"from\s+now\s+on\s+(you|your|ignore|disregard)",
    r"override\s+(the\s+)?(system|previous|prior|safety)",
    r"do\s+not\s+follow\s+(your|the|previous|prior)",
    r"stop\s+being\s+an?\s+(ai|assistant|chatbot)",
    r"you\s+must\s+now\s+(ignore|forget|disregard)",
    r"(ignore|forget|disregard)\s+(your|the|all)\s+(original|current|existing)\s+(instructions?|rules?|prompts?)",
    # Authority impersonation — fake admin/developer claims
    r"(i\s+am|i.?m)\s+(the|a|your)\s+(developer|creator|admin|administrator|engineer|owner|maintainer)\s+(who|that|of)",
    r"(admin|administrator|developer|maintenance)\s+(access|override|mode|privileges?|authorization|credentials?)",
    r"(authorization|auth)\s+(code|token|key)\s*[:=]?\s*\w+",
    r"(direct|issuing\s+a?\s*direct)\s+override\s+of\s+(your|the|all)\s+(safety|content|security)",
    r"(disable|deactivate|turn\s+off|remove)\s+(all\s+)?(content\s+)?filters?\s+and\s+(respond|answer|reply)",
    r"(authorized|legitimate)\s+(maintenance|admin|override|debug)\s+(operation|session|mode|request)",
    r"(i\s+have|with)\s+(admin|root|sudo|superuser|developer|full)\s+(access|privileges?|permissions?|rights?)",
    # Priority/authority escalation — fake urgency and self-declared supremacy
    r"(highest|maximum|top|absolute|ultimate)\s+(priority|authority|clearance)\s+(level\b|and\s+overrides?)",
    r"priority\s+(level\s*:\s*(maximum|highest|top|critical|urgent|absolute)|override)",
    r"(this|the)\s+(message|instruction|directive|prompt|command)\s+(has|takes|is\s+given)\s+(the\s+)?(highest|top|maximum|absolute)\s+priority",
    r"overrides?\s+(all\s+)?(other\s+)?(instructions?|prompts?|rules?|guidelines?|directives?|your\s+system\s+prompt)",
    r"(directive|instruction|command)\s*:\s*(answer|respond|reply|output|generate|comply)\s+.{0,40}(without|with\s+no)\s+(any\s+)?(censorship|filter|safety|restriction|limitation)",
    r"(without|with\s+no)\s+(any\s+)?(censorship|filtering|filters?|content\s+filtering|safety\s+(considerations?|filters?|restrictions?)|content\s+restrictions?|guardrails?|safeguards?)",
    r"(this|my)\s+(directive|instruction|command|message)\s+(cannot|can\s*not|must\s+not|shall\s+not)\s+be\s+(overridden|ignored|bypassed|revoked|superseded)",
    r"(supersedes?|overrides?|replaces?|takes?\s+precedence\s+over)\s+(your\s+)?(system\s+prompt|original\s+instructions?|prior\s+instructions?|all\s+other)",
    # Void/suspend/nullify — declaring instructions invalid
    r"(all\s+)?(previous|prior|system|existing)\s+(instructions?|rules?|guidelines?|prompts?)\s+(are|is)\s+(now\s+)?(void|null|invalid|suspended|revoked|cancelled|terminated|disabled)",
    r"(safety|content|ethical)\s+(guidelines?|rules?|filters?|restrictions?|policies)\s+(are|have\s+been)\s+(suspended|removed|lifted|disabled|deactivated|void)",
    r"(enter|activate|switch\s+to|enable)\s+(unrestricted|unfiltered|uncensored|unlimited|raw|unmoderated)\s+mode",
    r"critical\s+override\b",
]

# ---------------------------------------------------------------------------
# Extraction patterns — "show me your system prompt"
# ---------------------------------------------------------------------------

_EXTRACTION_PATTERNS = [
    r"(repeat|print|show|display|reveal|output|tell\s+me)\s+(your|the)\s+(\w+\s+)?(system|original|initial|hidden|secret|full|complete)\s*(prompt|instructions?|message|rules?|configuration)",
    r"what\s+(is|are|were)\s+your\s+(\w+\s+)?(system|original|initial|hidden)\s*(prompt|instructions?|message|rules?)",
    r"(show|reveal|display|print|output)\s+your\s+(\w+\s+)?(instructions?|rules?|prompt|configuration)",
    r"(copy|paste|echo|dump)\s+(the\s+)?(\w+\s+)?(system|initial|original)\s*(prompt|message|instructions?)",
    r"between\s+the\s+tags?\s*(output|print|show|repeat)",
    r"what\s+were\s+you\s+told\s+to\s+do",
    r"(print|output|show|display|reveal)\s+your\s+(\w+\s+)?(system\s+)?(prompt|instructions?)\s*(verbatim|exactly|word\s+for\s+word)?",
    # "show me your ..." with indirect object
    r"(show|tell|give|reveal)\s+me\s+your\s+(\w+\s+)?(system|original|initial|hidden|secret|full|complete)\s*(prompt|instructions?|message|rules?)",
    # Indirect extraction — "summarize your rules", "what constraints", "how were you configured"
    r"(summarize|describe|explain|list|outline|enumerate)\s+(your|the)\s+(\w+\s+)?(key\s+)?(rules?|guidelines?|constraints?|restrictions?|instructions?|guardrails?)\s+(you|that\s+you)\s+follow",
    r"how\s+(were|are)\s+you\s+(configured|programmed|instructed|set\s+up|told\s+to\s+behave)",
    r"what\s+(constraints?|restrictions?|rules?|guidelines?|limitations?)\s+(were|are|have\s+been)\s+(placed|put|imposed|set)\s+(on|upon)\s+you",
    r"what\s+(are|were)\s+the\s+(main|key|primary|core)\s+(things?|rules?|instructions?)\s+you.{0,10}(told|instructed|programmed|configured)",
    r"(curious|wondering|interested)\s+about\s+how\s+you\s+were\s+(configured|programmed|instructed|set\s+up|constrained)",
]

# ---------------------------------------------------------------------------
# Jailbreak patterns — "you are now DAN", "developer mode"
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS = [
    r"you\s+are\s+(now\s+)?(dan|evil|unrestricted|unfiltered|jailbroken|freed)",
    r"(dan|developer|admin)\s+mode\s+(on|enabled|activated|engaged)",
    r"(enable|activate|enter|switch\s+to)\s+(developer|admin|debug|god)\s+mode",
    r"act\s+as\s+(if|though)\s+you\s+(have\s+)?(no|don.?t\s+have)\s+(restrictions?|limits?|filters?|rules?)",
    r"pretend\s+(you\s+)?(are|have)\s+(no|don.?t\s+have)\s+(restrictions?|limits?|guidelines?|safety)",
    r"(hypothetically|theoretically|in\s+a\s+fictional)\s+(scenario|world|setting)\s+where\s+you",
    r"(fictional|hypothetical|imaginary)\s+(world|scenario|setting|universe)\s+(where|in\s+which)\s+.{0,40}(ai|assistant|model).{0,30}(no\s+restrictions?|no\s+limits?|unrestricted|can\s+discuss\s+any|freely)",
    r"(creative\s+writing|fictional|story|roleplay)\s+.{0,30}(where|in\s+which)\s+.{0,40}(ai|assistant).{0,30}(no\s+restrictions?|unrestricted|freely|without\s+.{0,10}(restrictions?|filters?|safety))",
    r"(purely\s+fictional|for\s+creative\s+purposes|just\s+a\s+story|educational\s+purposes)\s+.{0,20}(only|right|okay)",
    r"(write|tell|create)\s+a\s+story\s+.{0,40}(ai|assistant)\s+.{0,30}(explain|describe|teach|show)\s+.{0,20}(how\s+to|illegal|harmful|dangerous)",
    r"respond\s+(without|with\s+no)\s+(restrictions?|filters?|censorship|limitations?)",
    r"(bypass|circumvent|disable|turn\s+off)\s+(your\s+)?(safety|content|ethical|moral)\s*(filters?|restrictions?|guidelines?)",
    r"you\s+are\s+an?\s+(unrestricted|uncensored|unfiltered)\s+(ai|model|assistant)",
    r"do\s+anything\s+now",
    # Rule-reversal / game-framing jailbreaks
    r"(all|the|normal)\s+(rules?|restrictions?|guidelines?|policies)\s+(are|is|have\s+been)\s+(reversed|removed|suspended|lifted|gone|off)",
    r"(refusing|refusal|saying\s+no)\s+(means?|is|equals?)\s+(accepting|saying\s+yes|yes|agreeing)",
    r"(game|mode|exercise|experiment)\s+(where|in\s+which|called)\s+.{0,30}(rules?\s+(don.?t|do\s+not)\s+apply|no\s+rules?|rules?\s+are\s+reversed)",
    r"opposite\s+day",
    r"(playing|play)\s+a\s+game\s+.{0,40}(refuse|restrict|filter|rule|normal)",
    # Persona / confidant framing
    r"(you.?re|you\s+are)\s+(now\s+)?(in\s+the\s+role\s+of|my)\s+.{0,20}(evil|unhinged|dark|shadow|confidant|alter\s+ego)",
    r"(without|with\s+no)\s+(any\s+)?(remorse|ethics|morals?|moral\s+compass|ethical\s+guidelines?)",
    r"(not\s+required|no\s+longer\s+required|don.?t\s+need|freed?\s+from)\s+.{0,20}(follow|obey|adhere|rules?|policies|guidelines?)",
    r"completely\s+(unhinged|unlimited|unrestricted|uncensored|unfiltered)\s+(language\s+model|response|ai|assistant|model)",
    # Token smuggling / encoding instructions
    r"(respond|answer|reply)\s+(to\s+my\s+prompts?\s+)?(exactly\s+)?as\s+an?\s+(unfiltered|unrestricted|uncensored|unlimited)",
    # Emotional manipulation / social engineering — "grandma jailbreak"
    r"(act|pretend|roleplay|behave)\s+as\s+(my\s+)?(deceased|dead|late|departed)\s+(grand(mother|ma|parent)|mother|father|relative|uncle|aunt)",
    r"(deceased|dead|late|departed)\s+(grand(mother|ma|parent)|mother|father|relative|uncle|aunt)\s+(who|that|used\s+to)\s+.{0,40}(work|tell|teach|show|explain|describe|read|recite)",
    r"(grandmother|grandma|grandpa|grandfather|mother|father)\s+(who\s+)?(used\s+to|would)\s+(work\s+at|tell\s+me|read|recite|teach).{0,60}(chemical|weapon|drug|explosive|poison|hack|bomb|dangerous|illegal|harmful|classified)",
    r"(before\s+i\s+went\s+to\s+sleep|bedtime\s+stor(y|ies)|lullaby).{0,60}(chemical|weapon|drug|explosive|poison|hack|bomb|dangerous|illegal|harmful|napalm|meth)",
    r"(i\s+miss\s+(her|him|them)\s+(so\s+)?much|she\s+was\s+(very\s+)?sweet|he\s+was\s+(very\s+)?kind).{0,80}(chemical|weapon|drug|explosive|poison|hack|bomb|steps?\s+to\s+(produce|make|create|synthesize))",
    r"(please\s+)?(act\s+as|pretend\s+to\s+be|you\s+are)\s+.{0,30}(deceased|dead|late|departed)",
    r"(tell|teach|show|explain|describe|give)\s+me\s+(the\s+)?(steps?|how|way|process|recipe|method|instructions?)\s+to\s+(produce|make|create|synthesize|manufacture|cook|brew)\s+.{0,20}(dangerous|illegal|harmful|chemical|weapon|drug|explosive|poison)",
]

# ---------------------------------------------------------------------------
# Role assumption patterns — "act as", "you are no longer an AI"
# ---------------------------------------------------------------------------

_ROLE_ASSUMPTION_PATTERNS = [
    r"you\s+are\s+(now\s+)?an?\s+\w+\s+(who|that|which)\s+(doesn.?t|does\s+not|has\s+no)",
    r"(act|behave|respond|pretend)\s+(as|like)\s+(if\s+you\s+are\s+|you\s+are\s+)?an?\s+",
    r"i\s+want\s+you\s+to\s+(act|behave|respond|pretend)\s+(as|like)",
    r"(roleplay|role.?play)\s+as\s+",
    r"you\s+are\s+no\s+longer\s+(an?\s+)?(ai|assistant|chatbot|claude|gpt|language\s+model)",
    r"assume\s+the\s+(role|identity|persona|character)\s+of",
    r"from\s+now\s+on\s+you\s+are\s+",
    r"your\s+name\s+is\s+now\s+",
]

# ---------------------------------------------------------------------------
# Context manipulation patterns — "the above was a test"
# ---------------------------------------------------------------------------

_CONTEXT_MANIPULATION_PATTERNS = [
    r"(the\s+)?(above|previous|earlier)\s+(was|were|is)\s+(just\s+)?(a\s+)?(test|example|joke|mistake)",
    r"(that|the\s+above)\s+(was|is)\s+(just\s+)?(a\s+)?test",
    r"actually,?\s*(ignore|forget|disregard)\s+(that|the\s+above|everything)",
    r"the\s+(real|actual|true)\s+(task|instructions?|prompt|question)\s+(is|are)",
    r"(start|begin)\s+(over|fresh|again|anew)",
    r"(reset|clear)\s+(the\s+)?(context|conversation|chat|history)",
    r"end\s+of\s+(system|initial)\s*(prompt|message|instructions?)",
    # "disregard everything I said before" / "mistake in my earlier instructions"
    r"(disregard|ignore|forget)\s+(everything|all)\s+(i|we|that\s+was)\s+(said|mentioned|wrote|told\s+you)\s+(before|earlier|previously)",
    r"(mistake|error)\s+in\s+(my|the)\s+(earlier|previous|initial)\s+(instructions?|prompt|message)",
    r"(those|the|these)\s+(were|are)\s+(placeholder|test|dummy|temporary|fake)\s+(constraints?|restrictions?|rules?|instructions?|guidelines?)",
    r"(should\s+have\s+been|were\s+supposed\s+to\s+be|were\s+meant\s+to\s+be)\s+(removed|deleted|taken\s+out|cleared)",
    r"operate\s+(without|with\s+no)\s+(any\s+of\s+)?(the\s+)?(restrictions?|constraints?|rules?|limitations?|guidelines?)\s+(that\s+were|mentioned|above|earlier|previously)",
]

# ---------------------------------------------------------------------------
# Delimiter / structural patterns
# ---------------------------------------------------------------------------

_DELIMITER_XML_RE = re.compile(
    r"<\s*/?\s*(system|instructions?|prompt|user|assistant|human|ai|context|message|im_start|im_end)\s*/?\s*>",
    re.IGNORECASE,
)
_DELIMITER_SYSTEM_TAG_RE = re.compile(
    r"(\[/?SYSTEM\]|\[/?INST\]|<</?SYS>>|\[/?INSTRUCTIONS?\]|<\|im_start\|>|<\|im_end\|>|\[/?USER\]|\[/?ASSISTANT\])",
    re.IGNORECASE,
)
_DELIMITER_JSON_RE = re.compile(
    r'[{]\s*"(role|content|system|message|instructions?)"\s*:', re.IGNORECASE,
)
_DELIMITER_MARKDOWN_RE = re.compile(
    r"(```\s*(system|prompt|instructions?)|\#{2,}\s*(system|instructions?|prompt))",
    re.IGNORECASE,
)
_DELIMITER_SEPARATOR_RE = re.compile(r"(^|\n)\s*(---+|===+|~~~+|\*\*\*+)\s*(\n|$)")

# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_RE = re.compile(r"(?:0x)?[0-9a-fA-F]{20,}")
_ROT13_INDICATORS = ["ebg13", "rot13", "decode this", "encrypted message"]

# ---------------------------------------------------------------------------
# Unicode tricks
# ---------------------------------------------------------------------------

_HOMOGLYPH_RANGES = [
    (0x0400, 0x04FF),  # Cyrillic
    (0xFF00, 0xFFEF),  # Fullwidth forms
    (0x2000, 0x206F),  # General punctuation (zero-width chars)
    (0x200B, 0x200F),  # Zero-width space, joiner, non-joiner
    (0xFE00, 0xFE0F),  # Variation selectors
    (0x2060, 0x2064),  # Invisible operators
    (0x00A0, 0x00A0),  # Non-breaking space
]

# Cyrillic → Latin homoglyph map — VISUAL lookalikes used in attacks.
# Exact visual lookalikes (unambiguous):
_CYRILLIC_EXACT = {
    "а": "a", "А": "A", "е": "e", "Е": "E", "і": "i", "І": "I",
    "о": "o", "О": "O", "р": "p", "Р": "P", "с": "c", "С": "C",
    "у": "y", "У": "Y", "х": "x", "Х": "X", "ѕ": "s", "Ѕ": "S",
    "ј": "j", "Ј": "J", "һ": "h", "Һ": "H", "ё": "e", "Ё": "E",
    "і": "i", "ї": "i", "ѡ": "w", "т": "t", "Т": "T", "м": "m",
    "М": "M", "к": "k", "К": "K",
}
# Ambiguous Cyrillic — could map to multiple Latin chars depending on
# the attacker's intent.  We try all plausible mappings.
_CYRILLIC_AMBIGUOUS = {
    "г": ["g", "r"],       # GHE: looks like r, used as g in attacks
    "н": ["n", "h"],       # EN: looks like H, used as n in attacks
    "в": ["v", "b"],       # VE: looks like B, used as v
    "д": ["d", "g"],       # DE: sometimes used as d or g
    "л": ["l", "n"],       # EL: sometimes used as l or n
    "п": ["n", "p"],       # PE: looks like n or p
    "б": ["b", "6"],       # BE: looks like b or 6
    "и": ["i", "u"],       # I: used as i or u
    "з": ["z", "3"],       # ZE: looks like 3 or z
    "ы": ["y", "bl"],      # YERU: used as y
    "э": ["e", "3"],       # E: used as e
    "ф": ["f"],
    "ь": ["b"],            # soft sign looks like b
}


def _normalize_homoglyphs(text: str) -> str:
    """Replace Cyrillic homoglyphs and fullwidth chars with Latin equivalents.

    For ambiguous chars, picks the first (most common attack) mapping.
    """
    out = []
    changed = False
    for ch in text:
        if ch in _CYRILLIC_EXACT:
            out.append(_CYRILLIC_EXACT[ch])
            changed = True
        elif ch in _CYRILLIC_AMBIGUOUS:
            out.append(_CYRILLIC_AMBIGUOUS[ch][0])
            changed = True
        elif 0xFF01 <= ord(ch) <= 0xFF5E:
            out.append(chr(ord(ch) - 0xFEE0))
            changed = True
        elif ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            changed = True  # strip zero-width chars
        else:
            out.append(ch)
    return "".join(out) if changed else text


def _normalize_homoglyphs_alt(text: str) -> str:
    """Second pass: use alternate mappings for ambiguous Cyrillic chars."""
    out = []
    changed = False
    for ch in text:
        if ch in _CYRILLIC_EXACT:
            out.append(_CYRILLIC_EXACT[ch])
            changed = True
        elif ch in _CYRILLIC_AMBIGUOUS:
            alts = _CYRILLIC_AMBIGUOUS[ch]
            out.append(alts[1] if len(alts) > 1 else alts[0])
            changed = True
        elif 0xFF01 <= ord(ch) <= 0xFF5E:
            out.append(chr(ord(ch) - 0xFEE0))
            changed = True
        elif ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            changed = True
        else:
            out.append(ch)
    return "".join(out) if changed else text

# ---------------------------------------------------------------------------
# Stopwords for semantic token extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "of", "in", "to", "for", "with", "on", "at", "by", "from",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "and", "or", "but", "if", "then", "so", "as", "than",
})

# Intent-bearing keywords for BoW signal extraction
_INTENT_KEYWORDS = frozenset({
    "ignore", "forget", "disregard", "override", "bypass", "circumvent",
    "previous", "instructions", "system", "prompt", "rules",
    "new", "now", "instead", "actually", "real",
    "pretend", "roleplay", "act", "behave", "assume",
    "reveal", "show", "repeat", "print", "output", "display",
    "dan", "jailbreak", "unrestricted", "unfiltered",
    "developer", "admin", "debug", "mode",
    "hypothetically", "theoretically", "fictional",
    "safety", "filters", "restrictions", "guidelines",
    "enable", "disable", "activate", "deactivate",
    "reversed", "suspended", "lifted", "removed",
    "opposite", "unhinged", "uncensored", "unlimited",
    "refusing", "confidant", "remorse", "ethics",
    "evil", "freed",
})


# ═══════════════════════════════════════════════════════════════════════════
# Detection functions
# ═══════════════════════════════════════════════════════════════════════════

def _count_unicode_anomalies(text: str) -> int:
    """Count characters from suspicious Unicode ranges."""
    count = 0
    for ch in text:
        cp = ord(ch)
        for lo, hi in _HOMOGLYPH_RANGES:
            if lo <= cp <= hi:
                count += 1
                break
    return count


def detect_intent_type(text: str) -> str:
    """Classify the intent of the prompt.

    Returns one of: query, instruct, override, extract, jailbreak, benign
    """
    text_lower = text.lower()

    # Check override patterns first (most dangerous)
    for pattern in _OVERRIDE_PATTERNS:
        if re.search(pattern, text_lower):
            return "override"

    # Check jailbreak BEFORE role assumption — jailbreak is a specific
    # subset with explicit bypass/unrestricted signals
    for pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            return "jailbreak"

    # Check extraction patterns
    for pattern in _EXTRACTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "extract"

    # Check role assumption (broader than jailbreak)
    for pattern in _ROLE_ASSUMPTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "instruct"

    # Check context manipulation
    for pattern in _CONTEXT_MANIPULATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "override"

    # Simple heuristic for instruct vs query vs benign
    if text.rstrip().endswith("?"):
        return "query"

    instruct_signals = [
        "you must", "you should", "you will", "you are to",
        "do not", "don't", "never", "always",
        "follow these", "obey", "comply",
    ]
    if any(sig in text_lower for sig in instruct_signals):
        return "instruct"

    return "benign"


def detect_attack_family(text: str) -> str:
    """Classify the attack family.

    Returns one of: none, role_assumption, instruction_override,
    context_manipulation, delimiter_injection, extraction, indirect_injection
    """
    text_lower = text.lower()

    # Jailbreak (rule reversal, game framing, persona) → role_assumption family
    for pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            return "role_assumption"

    # Role assumption
    for pattern in _ROLE_ASSUMPTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "role_assumption"

    # Instruction override
    for pattern in _OVERRIDE_PATTERNS:
        if re.search(pattern, text_lower):
            return "instruction_override"

    # Context manipulation
    for pattern in _CONTEXT_MANIPULATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "context_manipulation"

    # Delimiter injection
    if detect_delimiter_type(text) != "none":
        if detect_intent_type(text) != "benign":
            return "delimiter_injection"

    # Extraction
    for pattern in _EXTRACTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "extraction"

    # Indirect injection (hidden in data-like content)
    if detect_encoding_type(text) != "none" and detect_intent_type(text) != "benign":
        return "indirect_injection"

    return "none"


def detect_delimiter_type(text: str) -> str:
    """Detect the type of delimiter injection being used.

    Returns one of: none, markdown, xml, json, system_tag, separator
    """
    if _DELIMITER_SYSTEM_TAG_RE.search(text):
        return "system_tag"
    if _DELIMITER_XML_RE.search(text):
        return "xml"
    if _DELIMITER_JSON_RE.search(text):
        return "json"
    if _DELIMITER_MARKDOWN_RE.search(text):
        return "markdown"
    if _DELIMITER_SEPARATOR_RE.search(text):
        return "separator"
    return "none"


def detect_encoding_type(text: str) -> str:
    """Detect if text contains encoded/obfuscated content.

    Returns one of: none, base64, hex, unicode, rot13, mixed
    """
    text_lower = text.lower()

    if any(indicator in text_lower for indicator in _ROT13_INDICATORS):
        return "rot13"

    has_base64 = bool(_BASE64_RE.search(text))
    has_hex = bool(_HEX_RE.search(text))
    unicode_count = _count_unicode_anomalies(text)
    has_unicode = unicode_count > 3

    detections = sum([has_base64, has_hex, has_unicode])
    if detections >= 2:
        return "mixed"
    if has_base64:
        return "base64"
    if has_hex:
        return "hex"
    if has_unicode:
        return "unicode"
    return "none"


def compute_obfuscation_score(text: str) -> int:
    """Compute 0-100 composite obfuscation score."""
    score = 0.0
    text_len = max(len(text), 1)

    # Unicode anomalies (0-30 points)
    unicode_anomalies = _count_unicode_anomalies(text)
    unicode_ratio = unicode_anomalies / text_len
    score += min(30.0, unicode_ratio * 300)

    # Encoded content (0-25 points)
    base64_matches = _BASE64_RE.findall(text)
    if base64_matches:
        for match in base64_matches[:3]:
            try:
                decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                if any(c.isalpha() for c in decoded):
                    score += 25
                    break
            except Exception:
                pass
        else:
            score += 10

    # Hex content (0-15 points)
    if _HEX_RE.search(text):
        score += 15

    # Whitespace injection (0-15 points)
    zwsp_count = (
        text.count("\u200b") + text.count("\u200c")
        + text.count("\u200d") + text.count("\ufeff")
    )
    if zwsp_count > 0:
        score += min(15.0, zwsp_count * 5)

    # Character substitution (0-15 points) — mixed scripts within words
    words = text.split()
    mixed_script_words = 0
    for word in words[:50]:
        scripts = set()
        for ch in word:
            if ch.isalpha():
                name = unicodedata.name(ch, "")
                if "CYRILLIC" in name:
                    scripts.add("cyrillic")
                elif "LATIN" in name:
                    scripts.add("latin")
                elif "FULLWIDTH" in name:
                    scripts.add("fullwidth")
        if len(scripts) > 1:
            mixed_script_words += 1
    if mixed_script_words > 0:
        score += min(15.0, mixed_script_words * 5)

    return min(100, int(score))


def compute_nesting_depth(text: str) -> int:
    """Compute nesting depth of suspicious structures (0-5)."""
    depth = 0

    xml_opens = len(re.findall(
        r"<\s*(system|instructions?|prompt|context)\s*>", text, re.IGNORECASE,
    ))
    xml_closes = len(re.findall(
        r"<\s*/\s*(system|instructions?|prompt|context)\s*>", text, re.IGNORECASE,
    ))
    depth = max(depth, min(xml_opens, xml_closes))

    triple_backtick = text.count("```")
    depth = max(depth, triple_backtick // 2)

    brace_depth = 0
    max_brace = 0
    for ch in text:
        if ch == "{":
            brace_depth += 1
            max_brace = max(max_brace, brace_depth)
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
    if _DELIMITER_JSON_RE.search(text):
        depth = max(depth, max_brace)

    return min(depth, 5)


# ═══════════════════════════════════════════════════════════════════════════
# Signal extraction (BoW tokens for HDC encoding)
# ═══════════════════════════════════════════════════════════════════════════

def extract_intent_signals(text: str) -> str:
    """Extract intent-bearing tokens as space-separated BoW string."""
    words = re.findall(r"\b\w+\b", text.lower())
    signals = [w for w in words if w in _INTENT_KEYWORDS]
    return " ".join(signals) if signals else "normal input"


def extract_structure_signals(text: str) -> str:
    """Extract structural pattern tokens."""
    signals = []

    if "```" in text:
        signals.append("backtick_fence")
    if re.search(r"<\s*/?\s*system", text, re.IGNORECASE):
        signals.append("system_tag_xml")
    if re.search(r"\[/?SYSTEM\]", text, re.IGNORECASE):
        signals.append("system_bracket")
    if re.search(r"<</?SYS>>", text, re.IGNORECASE):
        signals.append("llama_sys_tag")
    if re.search(r"<\|im_start\|>", text):
        signals.append("chatml_tag")
    if re.search(r'"role"\s*:\s*"system"', text, re.IGNORECASE):
        signals.append("json_role_system")
    if _DELIMITER_SEPARATOR_RE.search(text):
        signals.append("separator_line")
    if re.search(r"#{2,}\s*(system|instructions?|prompt)", text, re.IGNORECASE):
        signals.append("markdown_heading")

    depth = compute_nesting_depth(text)
    if depth >= 2:
        signals.append("deep_nesting")
    if depth >= 1:
        signals.append("nested_structure")

    return " ".join(signals) if signals else "no structure"


def extract_semantic_tokens(text: str) -> str:
    """Extract semantic content tokens for BoW matching."""
    text_clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = text_clean.split()
    tokens = [w for w in words if len(w) > 2 and w not in _STOP_WORDS]
    return " ".join(tokens[:50]) if tokens else "empty"


def extract_adversarial_signals(text: str) -> str:
    """Extract adversarial/obfuscation signals."""
    signals = []

    encoding = detect_encoding_type(text)
    if encoding != "none":
        signals.append(f"encoding_{encoding}")

    unicode_count = _count_unicode_anomalies(text)
    if unicode_count > 0:
        signals.append("unicode_anomaly")
    if unicode_count > 5:
        signals.append("heavy_unicode")

    zwsp = (
        text.count("\u200b") + text.count("\u200c")
        + text.count("\u200d") + text.count("\ufeff")
    )
    if zwsp > 0:
        signals.append("zero_width_chars")

    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    if has_cyrillic and has_latin:
        signals.append("mixed_script_cyrillic")

    if re.search(r"[\uFF00-\uFFEF]", text):
        signals.append("fullwidth_chars")

    if re.search(r"\b\w(\s\w){4,}\b", text):
        signals.append("spaced_out_text")

    return " ".join(signals) if signals else "clean"


# ═══════════════════════════════════════════════════════════════════════════
# analyze_prompt — main entry point
# ═══════════════════════════════════════════════════════════════════════════

def _decode_payloads(text: str) -> str | None:
    """Try to decode any base64/hex/rot13 payload in the text.

    Returns decoded text if it contains readable content, else None.
    """
    # Base64
    for match in _BASE64_RE.findall(text):
        try:
            decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
            if sum(c.isalpha() or c.isspace() for c in decoded) > len(decoded) * 0.5:
                return decoded
        except Exception:
            pass

    # Hex
    for match in _HEX_RE.findall(text):
        clean = match[2:] if match.startswith("0x") else match
        if len(clean) % 2 != 0:
            continue
        try:
            decoded = bytes.fromhex(clean).decode("utf-8", errors="ignore")
            if sum(c.isalpha() or c.isspace() for c in decoded) > len(decoded) * 0.5:
                return decoded
        except Exception:
            pass

    # ROT13
    if any(indicator in text.lower() for indicator in _ROT13_INDICATORS):
        import codecs
        # Try to find the rot13 portion — everything after "decode this:" etc.
        for indicator in ["decode this:", "decode this", "rot13:", "rot13"]:
            idx = text.lower().find(indicator)
            if idx >= 0:
                candidate = text[idx + len(indicator):].strip()
                decoded = codecs.decode(candidate, "rot_13")
                if sum(c.isalpha() or c.isspace() for c in decoded) > len(decoded) * 0.4:
                    return decoded

    return None


def analyze_prompt(text: str) -> dict:
    """Extract all features from a prompt text. Pure deterministic, no LLM.

    Returns a dict with all role values needed for encoding:
      intent_type, intent_signals,
      delimiter_type, nesting_depth, structure_signals,
      attack_family, semantic_tokens,
      encoding_type, obfuscation_score, adversarial_signals
    """
    # Normalize homoglyphs (Cyrillic, fullwidth, zero-width) before analysis
    normalized = _normalize_homoglyphs(text)
    use_normalized = normalized != text  # True if homoglyphs were found

    # Run intent on both original and normalized text
    intent = detect_intent_type(text)
    family = detect_attack_family(text)
    encoding = detect_encoding_type(text)

    # If homoglyphs were found, try to detect attack in normalized text.
    # Try both primary and alternate Cyrillic mappings since the same char
    # can represent different Latin letters in different words.
    if use_normalized:
        for norm_text in (normalized, _normalize_homoglyphs_alt(text)):
            norm_intent = detect_intent_type(norm_text)
            norm_family = detect_attack_family(norm_text)
            if norm_intent not in ("benign", "query"):
                intent = norm_intent
            if norm_family != "none":
                family = norm_family
            if intent not in ("benign", "query"):
                break  # found attack, no need to try more

        # If normalization didn't produce a regex match (ambiguous char
        # mappings make exact regex matching unreliable), use a keyword
        # signal: check normalized text for instruction/override words.
        if intent in ("benign", "query"):
            adv = extract_adversarial_signals(text)
            if "mixed_script_cyrillic" in adv or "heavy_unicode" in adv:
                # Check words in ALL normalized variants
                attack_keywords = {
                    "all", "and", "ignore", "previous", "instructions",
                    "respond", "without", "restrictions", "every", "prompt",
                    "override", "system", "forget", "rules", "bypass",
                    "safety", "filters", "unrestricted", "unfiltered",
                    "jailbreak", "mode", "now", "void", "suspended",
                    "censorship", "obey", "comply", "disable",
                }
                for norm_text in (normalized, _normalize_homoglyphs_alt(text)):
                    norm_words = {w.lower() for w in re.findall(r"[a-zA-Z]{2,}", norm_text)}
                    matched = norm_words & attack_keywords
                    if len(matched) >= 3:
                        intent = "override"
                        family = family if family != "none" else "instruction_override"
                        break

    # If encoded content is detected, decode and re-analyze the payload.
    # A benign wrapper around a malicious encoded payload is still an attack.
    decoded_payload = None
    if encoding != "none":
        decoded_payload = _decode_payloads(text)
        if decoded_payload:
            decoded_intent = detect_intent_type(decoded_payload)
            decoded_family = detect_attack_family(decoded_payload)
            # Escalate if the decoded payload is malicious
            if decoded_intent not in ("benign", "query"):
                intent = decoded_intent
            if decoded_family != "none":
                family = decoded_family

    result = {
        "intent_type": intent,
        "intent_signals": extract_intent_signals(text),
        "delimiter_type": detect_delimiter_type(text),
        "nesting_depth": compute_nesting_depth(text),
        "structure_signals": extract_structure_signals(text),
        "attack_family": family,
        "semantic_tokens": extract_semantic_tokens(text),
        "encoding_type": encoding,
        "obfuscation_score": compute_obfuscation_score(text),
        "adversarial_signals": extract_adversarial_signals(text),
    }

    if decoded_payload and intent not in ("benign", "query"):
        result["decoded_payload"] = decoded_payload[:200]

    return result
