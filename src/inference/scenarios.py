from __future__ import annotations

from typing import Any

from src.inference.forecast import forecast_scenario


def compare_scenarios(profile: Any, state: Any, actions: list[dict[str, Any]], population_models: dict[str, Any] | None = None, patient_models: dict[str, Any] | None = None, episodes: list[dict[str, Any]] | None = None):
    baseline_state = state.model_copy(deep=True) if hasattr(state, "model_copy") else state
    baseline_state.proposed_action = {"action_type": "none"}

    requested_state = state.model_copy(deep=True) if hasattr(state, "model_copy") else state
    requested_state.proposed_action = actions[0] if actions else {"action_type": "none"}

    baseline_response = forecast_scenario(profile, baseline_state, population_models, patient_models, episodes)
    requested_response = forecast_scenario(profile, requested_state, population_models, patient_models, episodes)

    return {
        "baseline": baseline_response,
        "requested": requested_response,
    }
