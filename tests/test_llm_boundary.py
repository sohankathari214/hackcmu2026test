from fastapi.testclient import TestClient
from src.api.app import app
def test_parse_state_exposes_parser_provenance_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY",raising=False)
    response=TestClient(app).post("/parse-state",json={"text":"I am at 145 and want a 45 minute run"})
    assert response.status_code==200
    assert response.json()["parser"]["gemini_used"] is False
    assert response.json()["state"]["proposed_action"]["action_type"]=="exercise"

def test_profile_and_policy_parse_are_structured(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY",raising=False); client=TestClient(app)
    assert client.post("/parse-profile",json={"text":"T1D pump"}).json()["data"]["patient_id"]=="patient_001"
    assert client.post("/parse-policy",json={"text":"avoid lows"}).json()["data"]["policy_version"]==1
