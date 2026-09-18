"""
Conversation state management: multi-turn memory + clarifying-question logic.

Sessions are kept in-memory (dict keyed by session_id) for this reference
implementation. For production, swap `SESSION_STORE` for Redis or a DB table —
the `Session` interface below does not need to change.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.rag import ALL_TRACKED_FIELDS

CLARIFYING_QUESTIONS = {
    "soil_organic_carbon_pct": "What is the soil organic carbon percentage (SOC %)?",
    "soil_ph": "What is the soil pH?",
    "soil_moisture": "How would you describe soil moisture — low, moderate, or high?",
    "land_use_type": "What is the current land use / crop type (e.g., monoculture wheat, mixed cropping, forest)?",
    "habitat_fragmentation": "How fragmented is the surrounding habitat — low, moderate, or high?",
    "rainfall": "How would you describe rainfall in the area — low, moderate, or high?",
    "temperature_trend": "Has local temperature been rising, stable, or falling in recent years?",
    "region": "What type of region is this — e.g., semi-arid, tropical, temperate?",
    "pesticide_use": "How would you describe pesticide/agrochemical use — low, moderate, or high?",
    "deforestation_rate": "Is there active deforestation or land clearing nearby — low, moderate, or high?",
    "water_pollution": "Is there visible nutrient/chemical runoff into nearby water bodies — low, moderate, or high?",
    "species_richness": "How would you rate observed species richness on the land — low, moderate, or high?",
    "pollinator_support": "How would you rate pollinator (bee/insect) activity — low, moderate, or high?",
}


@dataclass
class Session:
    session_id: str
    turns: list[dict[str, str]] = field(default_factory=list)  # [{"role": .., "content": ..}]
    structured_data: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def update_structured_data(self, new_data: dict[str, Any]) -> None:
        self.structured_data.update({k: v for k, v in new_data.items() if v not in (None, "")})

    def missing_fields(self, min_required: int = 3) -> list[str]:
        """Returns which *known* fields have not yet been supplied.

        The system requires at least `min_required` environmental variables
        before generating a full recommendation (hackathon constraint: "must
        handle at least 3 environmental variables together").
        """
        return [f for f in ALL_TRACKED_FIELDS if f not in self.structured_data]

    def has_enough_context(self, min_required: int = 3) -> bool:
        return len(self.structured_data) >= min_required

    def history_as_text(self, max_turns: int = 8) -> str:
        recent = self.turns[-max_turns:]
        return "\n".join(f"{t['role']}: {t['content']}" for t in recent)


SESSION_STORE: dict[str, Session] = {}


def get_or_create_session(session_id: str | None) -> Session:
    if session_id and session_id in SESSION_STORE:
        return SESSION_STORE[session_id]
    sid = session_id or str(uuid.uuid4())
    session = Session(session_id=sid)
    SESSION_STORE[sid] = session
    return session


def next_clarifying_questions(session: Session, max_questions: int = 3) -> list[str]:
    missing = session.missing_fields()
    # Prioritize the fields most likely to be immediately knowable/observable.
    priority_order = [
        "land_use_type", "soil_organic_carbon_pct", "rainfall",
        "soil_moisture", "region", "species_richness", "pollinator_support",
        "soil_ph", "habitat_fragmentation", "pesticide_use",
        "deforestation_rate", "water_pollution", "temperature_trend",
    ]
    ordered_missing = [f for f in priority_order if f in missing]
    return [CLARIFYING_QUESTIONS[f] for f in ordered_missing[:max_questions]]
