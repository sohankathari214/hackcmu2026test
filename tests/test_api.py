from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_validates_request():
    response = client.post("/forecast", json={
        "patient_id": "patient_001",
        "profile_version": 1,
        "demographics": {"age_years": 34},
        "diabetes": {"type": "T1D"},
        "medications": [],
        "conditions": [],
        "baseline_metrics": {},
        "source_metadata": {},
    })
    assert response.status_code == 422
