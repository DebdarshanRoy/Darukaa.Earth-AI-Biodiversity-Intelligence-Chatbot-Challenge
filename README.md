# Darukaa.Earth — AI Biodiversity Intelligence Chatbot

This is my submission for the Darukaa.Earth hackathon. The brief was pretty clear that they
didn't want a wrapper around an LLM with a nice prompt — they wanted something that actually
reasons from real data and can back up what it says. So that's what I focused on: a proper
knowledge layer, hybrid retrieval over it, and an LLM that's only allowed to talk about what
it actually retrieved.

## How it's put together

Roughly, a request goes through four stages:

1. **Extraction** (`app/extraction.py`) — pulls structured numbers/categories out of whatever
   the user typed (soil organic carbon %, pH, rainfall level, land use type, etc.) using
   regex/keyword matching. Nothing fancy here on purpose — I didn't want to burn an LLM call
   just to parse "rainfall is low" out of a sentence.
2. **Session memory** (`app/conversation.py`) — keeps track of what's been said and what
   metrics are known so far for that conversation, and figures out what's still missing.
3. **Retrieval** (`app/rag.py`) — this is the part I spent the most time on. It's a hybrid of:
   - TF-IDF + cosine similarity over the knowledge base text, for free-text queries like
     "biodiversity is declining on my land"
   - a rule-based matcher that evaluates each knowledge entry's condition (e.g.
     `soil_organic_carbon_pct < 0.5`) directly against whatever structured metrics the user
     has given, so numeric/categorical input gets precise matches every time
   
   Both result sets get merged, so the system benefits from the meaning of what someone says
   *and* the actual numbers they give it.
4. **Reasoning** (`app/llm.py`) — calls Claude with only the retrieved knowledge entries as
   context, and the system prompt is explicit that it can't make a claim that isn't traceable
   back to one of those entries, and that every recommendation has to connect at least two
   environmental variables (soil ↔ biodiversity, land use ↔ fragmentation, etc.) — I wanted
   to rule out the "use sustainable practices" kind of non-answer the brief called out
   directly. If there's no API key set, it falls back to composing the same structured output
   directly from the retrieved entries, so the whole thing still works and is still
   grounded — you don't need my API key to try it.

## The knowledge base itself

It's a JSON file (`data/knowledge_base.json`), not a database or a vector store. I went back
and forth on this — a "proper" RAG setup would usually mean embeddings in Chroma or
something similar — but for ~12 curated, cited entries, a flat file is honestly easier to
verify. Anyone reviewing this can open the file and check every fact and citation directly,
rather than trusting that whatever got embedded is accurate. Each entry looks like this:

```json
{
  "id": "kb-001",
  "category": "soil_health",
  "condition": "soil_organic_carbon_pct < 0.5",
  "recommendation": "Introduce legume-based cover crops...",
  "reasoning": "Legumes fix atmospheric nitrogen...",
  "impacted_metrics": ["soil_organic_carbon", "microbial_diversity", "pollinator_support"],
  "expected_effect": "SOC increases ~15-25% over 2-3 years...",
  "time_horizon": "medium_term",
  "confidence": "high",
  "source": "FAO, 'Soil Organic Carbon: the hidden potential', 2017"
}
```

Sources are FAO, IPCC (AR6), IPBES, ICRISAT, and CBD reports — all real, publicly available
assessments, not made-up citations.

If this needed to scale up to hundreds of papers instead of a curated dozen, I'd swap the
TF-IDF vectorizer for `sentence-transformers` embeddings + Chroma or pgvector. I built
`KnowledgeBase.retrieve()` as the one interface everything else talks to specifically so that
swap wouldn't touch `conversation.py` or `main.py` at all.

Session state is just an in-memory dict for now (`SESSION_STORE` in `conversation.py`) —
fine for a demo, but I'd move it to Redis before this saw real traffic.

## Multi-turn behavior

I tried to make the clarifying-question logic feel like it's actually paying attention
rather than firing off a checklist. It tracks 13 environmental fields across five categories,
and if fewer than 3 are known yet, it asks for more — prioritized toward the ones someone
would actually be able to answer off the top of their head (land use, rainfall, region)
before the more specific ones (habitat fragmentation, deforestation rate). Anything the user
mentions earlier in the conversation carries forward, so it's not re-asking things.

## Input formats

- Plain text — `POST /api/chat {"message": "..."}`
- Structured JSON, which takes priority over anything guessed from text —
  `POST /api/chat {"message": "...", "structured_data": {"soil_organic_carbon_pct": 0.3, "rainfall": "low", ...}}`
- Geo-coordinates weren't something I got to (it was listed as a bonus) — but the
  `structured_data` payload is open-ended, so adding `latitude`/`longitude` and routing them
  to a soil/climate lookup API would slot in cleanly.

## Running it

```bash
git clone <this-repo-url>
cd darukaa-biodiversity-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add ANTHROPIC_API_KEY if you want full LLM reasoning — optional, see note above

uvicorn app.main:app --reload --port 8000
# http://localhost:8000
```

Tests: `pytest tests/ -v` — covers the knowledge base (every entry has a source and at least
one impacted metric), both retrieval paths, text extraction, the chat endpoint's grounded
output, the clarifying-question trigger, and that multi-turn memory actually persists across
calls. All 7 pass locally.

## Deploying

There's a `Dockerfile` — `docker build -t darukaa-ai . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... darukaa-ai` and you're running. I set up a basic GitHub Actions workflow
(`.github/workflows/ci.yml`) that runs the test suite on every push/PR to main, mostly so I'd
notice if I broke something while iterating. For hosting, this would drop straight into
Render, Railway, or Fly.io — anything that takes a Dockerfile and an env var.

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/api/chat` | POST | main conversation endpoint |
| `/api/knowledge-base` | GET | dumps the full knowledge layer — useful for checking the grounding directly |
| `/api/health` | GET | health check |
| `/` | GET | the chat UI |

Example request/response:

```json
// Request
{
  "session_id": "optional-uuid-from-previous-turn",
  "message": "Biodiversity is declining on my land",
  "structured_data": { "region": "semi-arid" }
}

// Response
{
  "session_id": "uuid",
  "reply": "...(structured, cited recommendation text)...",
  "known_metrics": { "region": "semi-arid" },
  "clarifying_questions": ["What is the soil organic carbon percentage (SOC %)?", "..."],
  "retrieved_knowledge_ids": ["kb-006"]
}
```

## How this maps to the evaluation criteria

- **Depth of reasoning** — the LLM prompt actively rejects single-variable advice, and every
  KB entry lists at least two impacted metrics, so multi-variable connections are baked into
  the data, not just hoped for from the model.
- **Scientific grounding** — every recommendation traces back to a cited FAO/IPCC/IPBES/
  ICRISAT/CBD source; the LLM isn't allowed to cite anything outside the retrieved set.
- **Knowledge system design** — hybrid retrieval (TF-IDF semantic + rule-based structured)
  over a versioned, auditable knowledge file, not a bare prompt.
- **Conversational intelligence** — real session memory, missing-field tracking, and
  clarifying questions that adapt to what's already been said.
- **Output clarity** — every substantive answer follows the same structure: recommendation,
  why it works, metrics impacted, time horizon, confidence, source.
