from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


_ACTION_CODES = {"none": 0.0, "exercise": 1.0, "meal": 2.0, "caffeine": 3.0, "alcohol": 4.0, "other": 5.0}


def _feature_row(state: dict[str, Any], action: dict[str, Any]) -> dict[str, float]:
    exercise = action.get("exercise") or {}
    action_type = str(action.get("action_type", "none"))
    return {
        "starting_glucose": float((state.get("glucose") or {}).get("current_mg_dl") or (state.get("glucose") or {}).get("mg_dl") or 0),
        "glucose_delta": 0.0,
        "recent_insulin": 0.0,
        "recent_carbs": 0.0,
        "time_of_day": 0.0,
        "action_code": _ACTION_CODES.get(action_type, _ACTION_CODES["other"]),
        "exercise_duration": float(exercise.get("duration_minutes") or action.get("duration_minutes") or 0),
        "exercise_intensity": {"low": 1.0, "moderate": 2.0, "high": 3.0}.get(str(exercise.get("intensity") or action.get("intensity") or ""), 0.0),
    }


def retrieve_similar_events(episodes: list[dict[str, Any]], current_state: dict[str, Any], k: int = 10):
    if not episodes:
        return []

    rows = []
    for episode in episodes:
        state = episode.get("pre_action_state", {})
        rows.append(_feature_row(state, episode.get("planned_action", {})))

    source = pd.DataFrame(rows)
    query = pd.DataFrame([_feature_row(current_state, current_state.get("proposed_action", {}))]).reindex(columns=source.columns, fill_value=0.0)

    if source.empty:
        return []

    model = NearestNeighbors(n_neighbors=min(k, len(source)))
    model.fit(source.fillna(0).values)
    distances, indices = model.kneighbors(query.fillna(0).values)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        episode = episodes[int(idx)]
        results.append({
            "distance": float(distance),
            "starting_state_summary": {
                "glucose": episode.get("pre_action_state", {}).get("glucose", {}).get("current_mg_dl"),
                "action": episode.get("planned_action", {}).get("action_type"),
            },
            "actual_action": episode.get("actual_action"),
            "actual_glucose_trajectory": episode.get("observed", {}),
            "outcome_deltas": episode.get("residuals", {}),
        })

    return results
