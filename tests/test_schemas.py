from __future__ import annotations

import json
from pathlib import Path

from src.schemas.current_state import CurrentState
from src.schemas.medical_profile import MedicalProfile


def test_valid_json_accepted():
    profile = MedicalProfile.model_validate(json.loads(Path("examples/medical_profile.json").read_text()))
    state = CurrentState.model_validate(json.loads(Path("examples/current_state.json").read_text()))
    assert profile.patient_id == "patient_001"
    assert state.glucose["current_mg_dl"] == 123


def test_invalid_action_rejected():
    data = json.loads(Path("examples/current_state.json").read_text())
    data["proposed_action"] = {"action_type": "unsupported"}
    try:
        CurrentState.model_validate(data)
    except Exception:
        return
    raise AssertionError("Unsupported action should have been rejected")
