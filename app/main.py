from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.conversation import get_or_create_session, next_clarifying_questions
from app.extraction import extract_structured_data, merge_structured_input
from app.llm import generate_response
from app.rag import KnowledgeBase

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description="RAG-grounded conversational system for biodiversity/environmental reasoning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

kb = KnowledgeBase()


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    structured_data: Optional[dict[str, Any]] = None  # optional JSON input (bonus requirement)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    known_metrics: dict[str, Any]
    clarifying_questions: list[str]
    retrieved_knowledge_ids: list[str]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session = get_or_create_session(req.session_id)
    session.add_turn("user", req.message)

    # 1. Extract structured metrics from free text + merge any explicit JSON input.
    text_extracted = extract_structured_data(req.message)
    merged = merge_structured_input(text_extracted, req.structured_data)
    session.update_structured_data(merged)

    # 2. Retrieve grounding knowledge (hybrid: rule-based + semantic).
    retrieved = kb.retrieve(req.message, session.structured_data, top_k=4)

    # 3. Decide whether we have enough multi-metric context, or need to ask.
    questions = next_clarifying_questions(session)
    enough_context = session.has_enough_context(min_required=3)

    # 4. Generate the grounded, structured reply.
    history = session.history_as_text()
    reply = generate_response(req.message, session.structured_data, retrieved, history)

    if not enough_context and questions:
        reply += ("\n\nTo sharpen this further, could you also tell me: "
                  + " ".join(questions))

    session.add_turn("assistant", reply)

    return ChatResponse(
        session_id=session.session_id,
        reply=reply,
        known_metrics=session.structured_data,
        clarifying_questions=questions,
        retrieved_knowledge_ids=[e.id for e in retrieved],
    )


@app.get("/api/knowledge-base")
def list_knowledge_base() -> list[dict[str, Any]]:
    """Exposes the underlying knowledge layer for transparency/grading."""
    return [e.__dict__ for e in kb.entries]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
