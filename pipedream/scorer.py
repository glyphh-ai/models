"""
PipedreamScorer — ModelScorer implementation backed by pgvector.

Implements the ModelScorer protocol for CognitiveLoop integration.
Two-stage routing:
  1. extract_app() for app pre-filtering
  2. pgvector cosine similarity search within filtered app

Usage:
    from scorer import PipedreamScorer
    scorer = PipedreamScorer(encoder, db_conn)
    loop = CognitiveLoop(dimension=2000, model_scorer=scorer)
"""

import json
from typing import Any

import numpy as np
import psycopg2

from encoder import ENCODER_CONFIG, encode_query, extract_app
from glyphh.cognitive.model_scorer import ScorerResult
from glyphh.core.types import Concept
from glyphh.encoder import Encoder


class PipedreamScorer:
    """ModelScorer for Pipedream Action Router backed by pgvector.

    Performs two-stage routing:
      Stage 1: String-match app name from query → filter
      Stage 2: pgvector cosine similarity search (filtered or unfiltered)

    Returns ScorerResult with confidence derived from:
      - Top similarity score
      - Gap between top-1 and top-2 (disambiguation signal)
    """

    def __init__(
        self,
        encoder: Encoder,
        conn: Any,
        org_id: str = "custom",
        model_id: str = "pipedream",
        top_k: int = 5,
    ):
        self.encoder = encoder
        self.conn = conn
        self.org_id = org_id
        self.model_id = model_id
        self.top_k = top_k
        self._func_names: set[str] = set()

    # ── ModelScorer protocol ──

    def configure(self, func_defs: list[dict[str, Any]]) -> None:
        """Store known function names for validation."""
        self._func_names = {f["name"] for f in func_defs}

    def score(self, query: str) -> ScorerResult:
        """Score query via two-stage pgvector routing.

        Returns ScorerResult with:
          - functions: top matched action keys
          - confidence: composite of similarity + gap
          - all_scores: per-result scores for debugging
        """
        # Stage 1: Extract app from query
        matched_app = extract_app(query)

        # Stage 2: Encode query
        query_dict = encode_query(query)
        concept = Concept(name=query_dict["name"], attributes=query_dict["attributes"])
        glyph = self.encoder.encode(concept)
        embedding = glyph.global_cortex.data.astype(float).tolist()

        # Stage 3: pgvector search
        results = self._pgvector_search(embedding, matched_app)

        if not results:
            return ScorerResult(is_irrelevant=True, confidence=0.0)

        # Extract function names and scores
        functions = []
        all_scores = []
        for r in results:
            action_key = r["concept_text"]
            functions.append(action_key)
            all_scores.append({
                "function": action_key,
                "score": r["similarity"],
                "app_slug": r["app_slug"],
            })

        # Compute confidence from similarity + gap
        top_score = results[0]["similarity"]
        gap = (
            results[0]["similarity"] - results[1]["similarity"]
            if len(results) > 1
            else 0.1
        )

        # Confidence formula:
        #   - Base: top similarity score (0.0-1.0)
        #   - Gap bonus: larger gap → more confident (scaled 0-0.5)
        #   - Combined: weighted blend
        gap_factor = min(gap / 0.05, 1.0)  # Normalize gap (0.05 = full confidence)
        confidence = top_score * (0.6 + 0.4 * gap_factor)

        return ScorerResult(
            functions=functions[:1],  # Top match only
            arguments={},
            confidence=confidence,
            all_scores=all_scores,
            is_irrelevant=False,
        )

    def score_multi(self, query: str) -> ScorerResult:
        """Score for multiple matching functions (same as score)."""
        return self.score(query)

    def encode_query(self, query: str) -> Any:
        """Encode query into a Glyph for GlyphSpace caching.

        Returns None so the classifier falls through to score() path.
        We don't want GlyphSpace in-memory scoring — we use pgvector.
        """
        return None

    def get_func_glyphs(self) -> dict[str, Any]:
        """Return empty — we use pgvector, not in-memory glyphs."""
        return {}

    def scoring_strategy(self) -> Any:
        """Return None for default scoring strategy."""
        return None

    # ── Internal ──

    def _pgvector_search(
        self,
        embedding: list[float],
        matched_app: str | None,
    ) -> list[dict]:
        """Query pgvector for similar glyphs."""
        cur = self.conn.cursor()
        emb_str = "[" + ",".join(str(x) for x in embedding) + "]"

        if matched_app:
            cur.execute(
                """
                SELECT concept_text, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM glyphs
                WHERE org_id = %s AND model_id = %s
                  AND metadata->>'app_slug' = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (emb_str, self.org_id, self.model_id, matched_app, emb_str, self.top_k),
            )
        else:
            cur.execute(
                """
                SELECT concept_text, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM glyphs
                WHERE org_id = %s AND model_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (emb_str, self.org_id, self.model_id, emb_str, self.top_k),
            )

        rows = cur.fetchall()
        cur.close()

        results = []
        for row in rows:
            meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            results.append({
                "concept_text": row[0],
                "metadata": meta,
                "app_slug": meta.get("app_slug", ""),
                "similarity": row[2],
            })

        return results
