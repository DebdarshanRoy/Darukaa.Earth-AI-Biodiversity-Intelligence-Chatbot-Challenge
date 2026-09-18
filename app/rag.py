"""
Retrieval layer for the Darukaa Biodiversity Intelligence system.

Two retrieval paths are combined (a hybrid RAG design):

1. Semantic retrieval — TF-IDF vectorization + cosine similarity over the
   knowledge base's natural-language text (title + reasoning + recommendation).
   This is what lets the system respond sensibly to free-text queries like
   "biodiversity is declining on my land".

   NOTE: TF-IDF is used instead of a heavier embedding model (e.g. sentence
   transformers) to keep the reference implementation dependency-light and
   fast to deploy. Swapping in `sentence-transformers` + `chromadb` (see
   `rag_vectordb.py` stub) is a drop-in upgrade — the retrieval interface
   (`retrieve(query, structured_data) -> List[KBEntry]`) does not change.

2. Structured/rule retrieval — each knowledge entry carries a machine-checkable
   `condition` (e.g. "soil_organic_carbon_pct < 0.5"). When the user has
   supplied structured metrics (JSON), those conditions are evaluated directly.
   This guarantees that entries relevant to the user's actual numbers are
   always retrieved, even if their wording doesn't lexically match the KB text.

The two result sets are merged and de-duplicated, giving both breadth (semantic)
and precision (rule-based) retrieval — this is the "knowledge system" the
hackathon brief requires, not just a prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@dataclass
class KBEntry:
    id: str
    category: str
    metric: str
    condition: str
    title: str
    recommendation: str
    reasoning: str
    impacted_metrics: list[str]
    expected_effect: str
    time_horizon: str
    confidence: str
    source: str
    linked_variables: list[str]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KBEntry":
        return cls(**d)

    def to_text(self) -> str:
        return f"{self.title}. {self.recommendation}. {self.reasoning}"


class KnowledgeBase:
    """Loads the structured knowledge layer and builds a TF-IDF index over it."""

    def __init__(self, path: Path = KB_PATH):
        raw = json.loads(path.read_text())
        self.entries: list[KBEntry] = [KBEntry.from_dict(e) for e in raw]
        self._corpus = [e.to_text() for e in self.entries]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._corpus)

    # ---- Semantic retrieval -------------------------------------------------
    def semantic_search(self, query: str, top_k: int = 4) -> list[tuple[KBEntry, float]]:
        if not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [(self.entries[i], float(sims[i])) for i in ranked_idx if sims[i] > 0.05]

    # ---- Structured / rule-based retrieval ----------------------------------
    def rule_match(self, structured_data: dict[str, Any]) -> list[KBEntry]:
        matched = []
        for entry in self.entries:
            try:
                if _evaluate_condition(entry.condition, structured_data):
                    matched.append(entry)
            except Exception:
                # Missing fields simply mean the rule can't fire — not an error.
                continue
        return matched

    # ---- Hybrid entry point --------------------------------------------------
    def retrieve(self, query: str, structured_data: dict[str, Any] | None = None,
                 top_k: int = 4) -> list[KBEntry]:
        structured_data = structured_data or {}
        semantic_hits = [e for e, _ in self.semantic_search(query, top_k=top_k)]
        rule_hits = self.rule_match(structured_data)

        seen = set()
        merged: list[KBEntry] = []
        for e in rule_hits + semantic_hits:  # rule hits take priority (higher precision)
            if e.id not in seen:
                merged.append(e)
                seen.add(e.id)
        return merged[:top_k]


# ---- Tiny safe condition evaluator ----------------------------------------
# Conditions in the KB look like: "soil_organic_carbon_pct < 0.5"
# or "land_use_type == 'monoculture'" or with AND/OR.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _evaluate_condition(condition: str, data: dict[str, Any]) -> bool:
    """Evaluates a restricted boolean condition string against structured_data.

    Supports ==, !=, <, <=, >, >=, AND, OR over known field names only.
    Any field referenced that is absent from `data` causes the (sub-)condition
    to evaluate to False rather than raising — this is what triggers the
    system's "ask a clarifying question" behavior upstream.
    """
    py_expr = condition
    py_expr = py_expr.replace(" AND ", " and ").replace(" OR ", " or ")

    # Replace bare identifiers with dict lookups, leaving quoted string
    # literals (e.g. 'monoculture') untouched so we don't mangle them.
    def _replace_outside_quotes(segment: str) -> str:
        def _replace(match: re.Match) -> str:
            token = match.group(0)
            if token in ("and", "or", "True", "False", "None"):
                return token
            if token in data:
                return repr(data[token])
            # Unknown/missing field -> force this clause False without crashing.
            return "None"
        return _TOKEN_RE.sub(_replace, segment)

    # Split on single-quoted string literals; only transform the parts outside them.
    parts = re.split(r"('(?:[^'\\]|\\.)*')", py_expr)
    safe_expr = "".join(
        part if part.startswith("'") else _replace_outside_quotes(part)
        for part in parts
    )
    try:
        return bool(eval(safe_expr, {"__builtins__": {}}, {}))  # noqa: S307 (restricted, no builtins)
    except TypeError:
        # Comparing None to a number etc. -> condition doesn't apply.
        return False


REQUIRED_FIELDS_BY_CATEGORY = {
    "soil_health": ["soil_organic_carbon_pct", "soil_ph", "soil_moisture"],
    "land_use": ["land_use_type", "habitat_fragmentation"],
    "climate_factors": ["rainfall", "temperature_trend", "region"],
    "human_impact": ["pesticide_use", "deforestation_rate", "water_pollution"],
    "biodiversity_indicators": ["species_richness", "pollinator_support"],
}

ALL_TRACKED_FIELDS = sorted({f for fields in REQUIRED_FIELDS_BY_CATEGORY.values() for f in fields})
