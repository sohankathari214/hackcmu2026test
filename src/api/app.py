from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.schemas.current_state import CurrentState
from src.schemas.doctor_policy import DoctorPolicy
from src.schemas.medical_profile import MedicalProfile
from src.inference.forecast import forecast_scenario
from src.inference.support_registry import ActionSupportRegistry

app = FastAPI(title="GlucoPilot API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(profile: MedicalProfile, state: CurrentState):
    action_type = state.proposed_action.action_type if state.proposed_action else "none"
    registry = ActionSupportRegistry({
        "exercise": {"supported": True},
        "meal": {"supported": True},
        "alcohol": {"supported": False},
        "caffeine": {"supported": False},
    })
    supported, reason = registry.check(action_type)
    if not supported:
        return {
            "status": "unsupported_action",
            "reason": reason,
            "action_type": action_type,
        }
    return forecast_scenario(profile, state)


@app.post("/compare-scenarios")
def compare(profile: MedicalProfile, state: CurrentState):
    return {"status": "ok", "baseline": forecast_scenario(profile, state), "requested": forecast_scenario(profile, state)}


@app.post("/episodes/create")
def create_episode_endpoint(payload: dict[str, Any]):
    return {"status": "ok", "episode": payload}


@app.post("/episodes/finalize")
def finalize_episode_endpoint(payload: dict[str, Any]):
    return {"status": "ok", "episode": payload}


@app.get("/patients/{patient_id}/model-status")
def model_status(patient_id: str):
    return {"status": "ok", "patient_id": patient_id, "active": False}


@app.post("/patients/{patient_id}/retrain-personal")
def retrain_personal(patient_id: str):
    return {"status": "ok", "patient_id": patient_id}


@app.post("/evaluate-policy")
def evaluate_policy(prediction: dict[str, Any], policy: DoctorPolicy):
    return {
        "status": "ok",
        "risk": "low",
        "prediction": prediction,
    }
