from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.schemas.current_state import CurrentState
from src.schemas.doctor_policy import DoctorPolicy
from src.schemas.medical_profile import MedicalProfile
from src.inference.forecast import forecast_scenario
from src.inference.scenarios import compare_scenarios
from src.episodes.create import create_episode
from src.episodes.finalize import finalize_episode
from src.episodes.store import EpisodeStore
from src.decision import evaluate_permission
from src.decision import evaluate_lifestyle_change, plan_day
from src.safety import evaluate_safety
from src.models.personalization import alpha_for_episode_count
from src.models.registry import ModelRegistry
from src.api.dashboard import HTML
from src.physiology import fit_patient_parameters, monitor_drift
from src.inference.support_registry import ActionSupportRegistry

app = FastAPI(title="GlucoPilot API")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(profile: MedicalProfile, state: CurrentState):
    action_type = state.proposed_action.action_type if state.proposed_action else "none"
    registry = ActionSupportRegistry({k: {"supported": True} for k in ("exercise", "meal", "alcohol", "caffeine")})
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
    return compare_scenarios(profile, state, [state.proposed_action.model_dump()])


@app.post("/episodes/create")
def create_episode_endpoint(payload: dict[str, Any]):
    episode=create_episode(payload["patient_id"], payload["state"], payload.get("planned_action", {}), payload.get("prediction", {}))
    return EpisodeStore().append(episode)


@app.post("/episodes/finalize")
def finalize_episode_endpoint(payload: dict[str, Any]):
    episode=finalize_episode(payload["episode"], payload.get("actual_action", {}), payload.get("observed", {}), payload.get("population_forecast", {}))
    return EpisodeStore().replace(episode)


@app.get("/patients/{patient_id}/model-status")
def model_status(patient_id: str):
    usable=[e for e in EpisodeStore().patient(patient_id) if e.get("quality",{}).get("usable_for_personalization")]
    n=len(usable); return {"status":"ok","patient_id":patient_id,"active":n>=10,"valid_episode_count":n,"alpha":alpha_for_episode_count(n),"versions":ModelRegistry().list_versions(patient_id)}


@app.post("/patients/{patient_id}/retrain-personal")
def retrain_personal(patient_id: str):
    usable=[e for e in EpisodeStore().patient(patient_id) if e.get("quality",{}).get("usable_for_personalization")]
    if len(usable)<10: raise HTTPException(status_code=409,detail="At least 10 usable completed episodes are required before personal-model training.")
    # Residual fitting is intentionally invoked by the training job; endpoint reports real eligibility rather than fabricating a model.
    return {"status":"eligible_for_retraining","patient_id":patient_id,"valid_episode_count":len(usable),"alpha":alpha_for_episode_count(len(usable)),"warning":"Use scripts/train_personalization.py to persist the residual model."}


@app.post("/evaluate-policy")
def evaluate_policy(prediction: dict[str, Any], policy: DoctorPolicy):
    return {"status":"ok","prediction":prediction,"policy_version":policy.policy_version,"interpretation":{"hard_rules":policy.hard_rules,"risk_priorities":policy.risk_priorities,"note":"Policy interprets forecast; it does not alter physiological predictions."}}

@app.post("/permission")
def permission(state: CurrentState, action: dict[str, Any], policy: DoctorPolicy | None = None):
    return evaluate_permission(state, action, policy)

@app.post("/safety-check")
def safety_check(state: CurrentState, action: dict[str, Any], policy: DoctorPolicy | None = None):
    return evaluate_safety(state, action, policy)

@app.post("/plan-day")
def day_plan(state: CurrentState, actions: list[dict[str, Any]], policy: DoctorPolicy | None = None):
    return plan_day(state, actions, policy)

@app.post("/lifestyle-change")
def lifestyle_change(state: CurrentState, events: list[dict[str, Any]]):
    return evaluate_lifestyle_change(state, events)

@app.post("/patients/{patient_id}/fit-parameters")
def fit_parameters(patient_id: str, rows: list[dict[str, Any]]):
    import pandas as pd
    return {"patient_id":patient_id, **fit_patient_parameters(pd.DataFrame(rows))}

@app.post("/patients/{patient_id}/drift-status")
def drift_status(patient_id: str, residuals: list[float]):
    return {"patient_id":patient_id, **monitor_drift(residuals)}
