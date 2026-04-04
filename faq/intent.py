"""
FAQ intent extraction for Glyphh AI knowledge base.

Extracts category and keywords from natural language questions.
No SDK dependencies — purely local keyword matching.
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# Category signal lists — keyword → FAQ category mapping
# ---------------------------------------------------------------------------

_CATEGORY_SIGNALS: dict[str, list[str]] = {
    "getting_started": [
        "get started", "getting started", "begin", "first steps", "onboarding",
        "quickstart", "quick start", "new to", "new user", "beginner",
        "install", "installation", "pip install", "setup", "set up",
        "requirements", "python version", "system requirements",
        "hello world", "first model", "try glyphh", "start using",
    ],
    "sdk": [
        "encoder", "encoderconfig", "encoder config", "glyph", "vector",
        "concept", "layer", "segment", "role", "dimension", "dimensions",
        "bag of words", "bag_of_words", "bow", "lexicon", "lexicons",
        "thermometer", "symbolic", "bind", "bundle", "cosine", "similarity",
        "similarity weight", "key_part", "key part", "temporal",
        "include_temporal", "seed", "encoding", "encode", "bipolar",
        "hypervector", "hdc", "hyperdimensional",
    ],
    "runtime": [
        "runtime", "server", "fastapi", "uvicorn", "pgvector", "postgresql",
        "postgres", "database", "db", "mcp", "model context protocol",
        "fact tree", "stored procedure", "graphql", "gql",
        "done", "ask", "no_match", "no match", "confidence",
        "similarity search", "query pipeline", "nl query",
        "response state", "response states",
    ],
    "cli": [
        "cli", "command line", "terminal", "glyphh dev", "glyphh serve",
        "glyphh chat", "glyphh model", "glyphh token", "package",
        "packaging", ".glyphh file", "glyphh file", "model deploy",
        "model load", "model package", "token create",
    ],
    "models": [
        "custom model", "build a model", "create a model", "model directory",
        "encoder.py", "config.yaml", "manifest.yaml", "entry_to_record",
        "encode_query", "exemplar", "exemplars", "jsonl", "training data",
        "intent.py", "test", "benchmark", "model file", "model structure",
        "toolrouter", "pipedream", "churn", "bfcl",
        "faq model", "tool router", "function calling",
    ],
    "architecture": [
        "hyperdimensional computing", "hyperdimensional", "hdc", "deterministic",
        "no hallucination", "hallucination", "sidecar", "llm sidecar",
        "how does glyphh work", "how it works", "under the hood",
        "neural network", "embedding", "traditional", "difference",
        "why glyphh", "advantage", "benefit", "vs", "compared to",
        "rag", "retrieval", "augmented generation",
    ],
    "deployment": [
        "deploy", "deployment", "heroku", "docker", "docker compose",
        "cloud", "production", "database_url", "alembic", "migration",
        "migrations", "procfile", "dyno", "scaling", "environment",
        "env var", "environment variable", "config var",
    ],
    "pricing": [
        "pricing", "price", "cost", "free", "free tier", "paid",
        "plan", "plans", "tier", "tiers", "advanced", "pro", "enterprise",
        "limit", "limits", "quota", "rate limit", "how much",
        "subscription", "billing", "upgrade", "downgrade",
        "max models", "max glyphs", "requests per minute",
    ],
    "security": [
        "security", "authentication", "auth", "token", "api token",
        "api key", "jwt", "ed25519", "license", "licensing",
        "private", "privacy", "data privacy", "encryption",
        "credential", "credentials", "permission", "permissions",
        "org", "organization", "team",
    ],
    "troubleshooting": [
        "error", "bug", "issue", "problem", "not working", "broken",
        "fix", "debug", "troubleshoot", "why", "wrong", "incorrect",
        "low score", "low scores", "low similarity", "scores are low",
        "scores are really low", "similarity scores",
        "no match", "no results", "really low",
        "slow", "performance", "timeout", "crash", "memory",
        "oom", "out of memory", "500 error", "fails", "failing",
    ],
    "general": [
        "support", "contact", "help", "documentation", "docs",
        "community", "github", "open source", "contribute",
        "roadmap", "feature request", "feedback",
        "what is glyphh", "who", "company", "about",
    ],
}

# ---------------------------------------------------------------------------
# Stopwords to filter from keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "if", "then", "than", "when", "where", "how", "what", "which", "who",
    "all", "each", "every", "any", "few", "more", "most", "some", "such",
    "no", "only", "own", "same", "too", "very", "just",
    "don", "t", "s", "ll", "ve", "re", "d", "m",
    "get", "got", "going", "go", "want", "need", "use", "using",
])


def _preprocess(text: str) -> str:
    """Lowercase and normalise punctuation for consistent BoW encoding."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def extract_keywords(text: str) -> str:
    """Extract meaningful keywords from text, filtering stopwords."""
    cleaned = _preprocess(text)
    words = cleaned.split()
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return " ".join(keywords)


def infer_category(text: str) -> str:
    """Infer FAQ category from text using keyword signal matching.

    Returns the category with the most signal matches.
    Falls back to 'general' if no signals match.
    Uses word-boundary matching to avoid substring collisions
    (e.g. "glyph" must not match inside "glyphh").
    """
    lower = text.lower()
    best_cat, best_score = "general", 0
    for cat, signals in _CATEGORY_SIGNALS.items():
        score = sum(
            1 for s in signals
            if re.search(r"\b" + re.escape(s) + r"\b", lower)
        )
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat
