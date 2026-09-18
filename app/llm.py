"""
Reasoning layer. Calls an LLM (Anthropic Claude by default) with the
retrieved knowledge base entries as grounding context, and instructs it to:

- never invent a recommendation that isn't traceable to a retrieved entry
- explicitly connect >=2 environmental variables (multi-metric reasoning)
- structure output as: recommendation / why it works / metrics impacted /
  time horizon / confidence / source

If no API key is configured, a deterministic offline fallback composes the
same structured output directly from the retrieved KB entries (so the system
is fully runnable/demoable without external credentials).
"""

from __future__ import annotations

import os
from typing import Any

from app.rag import KBEntry

SYSTEM_PROMPT = """You are an AI environmental scientist for Darukaa.Earth. You reason about \
biodiversity, soil, land use, climate, and human-impact metrics for a specific parcel of land.

Rules:
1. You may ONLY make claims that are grounded in the RETRIEVED KNOWLEDGE provided to you below. \
Do not invent statistics or sources.
2. Every recommendation must explicitly connect at least two environmental variables \
(e.g. soil health <-> biodiversity, water availability <-> species survival, land use <-> \
habitat fragmentation). Single-variable, generic advice like "use sustainable practices" is \
NOT acceptable and will be rejected.
3. If the user's question is missing key data needed for a confident recommendation, say so \
plainly and ask for the specific missing metric — don't guess.
4. Structure every substantive answer using this format for each recommendation:
   - Recommendation: <what to do>
   - Why it works: <scientific reasoning, referencing the mechanism>
   - Metrics impacted: <list>
   - Time horizon: short-term / medium-term / long-term
   - Confidence: high / medium / low
   - Source: <citation from the retrieved knowledge>
5. Be conversational in framing but rigorous in content. Keep total answers focused — \
2-3 recommendations maximum, prioritized by expected impact.
"""


def build_user_prompt(query: str, structured_data: dict[str, Any],
                       retrieved: list[KBEntry], history: str) -> str:
    kb_block = "\n\n".join(
        f"[{e.id}] {e.title}\n"
        f"Recommendation: {e.recommendation}\n"
        f"Reasoning: {e.reasoning}\n"
        f"Impacted metrics: {', '.join(e.impacted_metrics)}\n"
        f"Expected effect: {e.expected_effect}\n"
        f"Time horizon: {e.time_horizon} | Confidence: {e.confidence}\n"
        f"Source: {e.source}"
        for e in retrieved
    ) or "(no strong knowledge-base matches — say so and ask a clarifying question instead)"

    data_block = "\n".join(f"- {k}: {v}" for k, v in structured_data.items()) or "(none supplied yet)"

    return f"""CONVERSATION HISTORY:
{history or '(new conversation)'}

USER'S KNOWN METRICS SO FAR:
{data_block}

RETRIEVED KNOWLEDGE (grounding — cite these, do not go beyond them):
{kb_block}

CURRENT USER MESSAGE:
{query}

Respond as the AI environmental scientist, following the structured format rules."""


def generate_response(query: str, structured_data: dict[str, Any],
                       retrieved: list[KBEntry], history: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _offline_fallback(query, structured_data, retrieved)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        user_prompt = build_user_prompt(query, structured_data, retrieved, history)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
    except Exception as exc:  # noqa: BLE001 — degrade gracefully in a demo/offline setting
        return _offline_fallback(query, structured_data, retrieved) + \
            f"\n\n_(Note: LLM call failed [{exc}], showing knowledge-grounded fallback.)_"


def _offline_fallback(query: str, structured_data: dict[str, Any], retrieved: list[KBEntry]) -> str:
    """Deterministic, template-based composition directly from KB entries.

    Used when no LLM API key is configured, so the system is still fully
    functional and evidence-grounded for local demos / grading without
    requiring credentials.
    """
    if not retrieved:
        return ("I don't have enough grounded knowledge to make a confident recommendation yet. "
                "Could you share more detail — soil organic carbon %, rainfall pattern, and land "
                "use type are the most useful starting points?")

    lines = ["Based on what you've shared, here's my analysis:\n"]
    for e in retrieved[:3]:
        lines.append(
            f"**Recommendation:** {e.recommendation}\n"
            f"**Why it works:** {e.reasoning}\n"
            f"**Metrics impacted:** {', '.join(e.impacted_metrics)}\n"
            f"**Expected effect:** {e.expected_effect}\n"
            f"**Time horizon:** {e.time_horizon.replace('_', ' ')} | **Confidence:** {e.confidence}\n"
            f"**Source:** {e.source}\n"
        )
    return "\n".join(lines)
