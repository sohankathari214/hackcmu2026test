from __future__ import annotations

import json
from pathlib import Path

from src.inference.forecast import forecast_scenario
from src.schemas.current_state import CurrentState
from src.schemas.medical_profile import MedicalProfile


def test_scenario_and_state_independent():
    profile = MedicalProfile.model_validate(json.loads(Path("examples/medical_profile.json").read_text()))
    state = CurrentState.model_validate(json.loads(Path("examples/current_state.json").read_text()))
    baseline = forecast_scenario(profile, state)
    assert baseline["scenario"]["action_type"] == "exercise"
