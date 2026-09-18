from fastapi.testclient import TestClient

from app.extraction import extract_structured_data
from app.main import app
from app.rag import KnowledgeBase

client = TestClient(app)
kb = KnowledgeBase()


def test_knowledge_base_loads():
    assert len(kb.entries) >= 10
    for e in kb.entries:
        assert e.source  # every entry must be evidence-backed
        assert len(e.impacted_metrics) >= 1


def test_semantic_retrieval_finds_relevant_entry():
    hits = kb.semantic_search("biodiversity is declining, low species richness", top_k=3)
    assert len(hits) > 0
    assert any("species_richness" in h[0].metric or "species_richness" in h[0].impacted_metrics
               for h in hits)


def test_rule_based_retrieval_multi_metric():
    data = {"soil_organic_carbon_pct": 0.3, "rainfall": "low", "land_use_type": "monoculture"}
    hits = kb.rule_match(data)
    assert len(hits) >= 2  # confirms multi-variable reasoning is possible


def test_extraction_pulls_soc_and_ph():
    text = "Soil organic carbon is 0.3% and pH is 5.2, rainfall is low, monoculture wheat, semi-arid region."
    data = extract_structured_data(text)
    assert data["soil_organic_carbon_pct"] == 0.3
    assert data["soil_ph"] == 5.2
    assert data["land_use_type"] == "monoculture"
    assert data["region"] == "semi-arid"
    assert data["rainfall"] == "low"


def test_chat_endpoint_returns_grounded_reply():
    res = client.post("/api/chat", json={
        "message": "Soil organic carbon is 0.3%, rainfall is low, monoculture wheat, semi-arid region.",
    })
    assert res.status_code == 200
    body = res.json()
    assert "session_id" in body
    assert len(body["retrieved_knowledge_ids"]) > 0
    assert "recommendation" in body["reply"].lower() or "Recommendation" in body["reply"]


def test_chat_asks_clarifying_question_when_underspecified():
    res = client.post("/api/chat", json={"message": "Biodiversity is declining on my land."})
    body = res.json()
    assert len(body["clarifying_questions"]) > 0


def test_multiturn_session_retains_context():
    res1 = client.post("/api/chat", json={"message": "Soil organic carbon is 0.3%."})
    sid = res1.json()["session_id"]
    res2 = client.post("/api/chat", json={"session_id": sid, "message": "Rainfall is low and it's a semi-arid region."})
    assert res2.json()["known_metrics"]["soil_organic_carbon_pct"] == 0.3
    assert res2.json()["known_metrics"]["rainfall"] == "low"
