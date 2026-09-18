"""
Extracts structured environmental metrics from free-text user input.

This is deliberately rule/regex-based (not an LLM call) so that structured
data extraction is fast, deterministic, and doesn't consume LLM tokens on
every turn. The LLM is reserved for reasoning over retrieved knowledge, not
for parsing numbers out of sentences.
"""

from __future__ import annotations

import re
from typing import Any

_NUM = r"(\d+(?:\.\d+)?)"

PATTERNS: dict[str, re.Pattern] = {
    "soil_organic_carbon_pct": re.compile(
        rf"(?:soil organic carbon|soc)[^\d%]{{0,15}}{_NUM}\s*%?", re.I
    ),
    "soil_ph": re.compile(rf"\bph\b[^\d]{{0,10}}{_NUM}", re.I),
}

LEVEL_WORDS = {"low", "moderate", "medium", "high"}

KEYWORD_FIELDS = {
    "soil_moisture": ["soil moisture"],
    "rainfall": ["rainfall", "precipitation"],
    "habitat_fragmentation": ["habitat fragmentation", "fragmentation"],
    "pesticide_use": ["pesticide", "agrochemical"],
    "deforestation_rate": ["deforestation", "forest clearing", "clearing"],
    "water_pollution": ["water pollution", "runoff", "nutrient pollution"],
    "species_richness": ["species richness", "species diversity"],
    "pollinator_support": ["pollinator", "bee activity", "insect activity"],
}

LAND_USE_KEYWORDS = [
    "monoculture", "intercropping", "agroforestry", "mixed cropping",
    "forest", "grassland", "wetland", "pasture", "orchard",
]

REGION_KEYWORDS = ["semi-arid", "arid", "tropical", "temperate", "subtropical", "coastal", "alpine"]

TEMPERATURE_TREND_KEYWORDS = {
    "rising": ["warming", "rising temperature", "getting hotter", "increasing temperature"],
    "falling": ["cooling", "falling temperature", "getting colder"],
    "stable": ["stable temperature", "no change in temperature"],
}


def extract_structured_data(text: str) -> dict[str, Any]:
    text_l = text.lower()
    out: dict[str, Any] = {}

    for field_name, pattern in PATTERNS.items():
        m = pattern.search(text)
        if m:
            out[field_name] = float(m.group(1))

    for field_name, keywords in KEYWORD_FIELDS.items():
        for kw in keywords:
            if kw in text_l:
                level = _nearby_level_word(text_l, kw)
                out[field_name] = level or "unspecified-mentioned"
                break

    for land_use in LAND_USE_KEYWORDS:
        if land_use in text_l:
            out["land_use_type"] = land_use
            break

    for region in REGION_KEYWORDS:
        if region in text_l:
            out["region"] = region
            break

    for trend, keywords in TEMPERATURE_TREND_KEYWORDS.items():
        if any(kw in text_l for kw in keywords):
            out["temperature_trend"] = trend
            break

    return out


def _nearby_level_word(text_l: str, keyword: str) -> str | None:
    """Looks for low/moderate/high within a small window around the keyword."""
    idx = text_l.find(keyword)
    window = text_l[max(0, idx - 25): idx + len(keyword) + 25]
    for w in LEVEL_WORDS:
        if w in window:
            return "moderate" if w == "medium" else w
    return None


def merge_structured_input(text_data: dict[str, Any], json_data: dict[str, Any] | None) -> dict[str, Any]:
    """JSON/structured input (when provided) takes priority over text extraction."""
    merged = dict(text_data)
    if json_data:
        merged.update({k: v for k, v in json_data.items() if v not in (None, "")})
    return merged
