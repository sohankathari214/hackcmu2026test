from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def retrieve_similar_events(episodes: list[dict[str, Any]], current_state: dict[str, Any], k: int = 10):
    if not episodes:
        return []

    rows = []
    for episode in episodes:
        state = episode.get("pre_action_state", {})
        row = {
            "patient_id": episode.get("patient_id"),
            "starting_glucose": state.get("glucose", {}).get("current_mg_dl"),
            "glucose_delta": 0.0,
            "recent_insulin": 0.0,
            "recent_carbs": 0.0,
            "time_of_day": 0.0,
            "action_category": str(episode.get("planned_action", {}).get("action_type", "none")),
            "exercise_duration": 0.0,
            "exercise_intensity": 0.0,
        }
        rows.append(row)

    source = pd.DataFrame(rows)
    query = pd.DataFrame([
        {
            "starting_glucose": current_state.get("glucose", {}).get("current_mg_dl"),
            "glucose_delta": 0.0,
            "recent_insulin": 0.0,
            "recent_carbs": 0.0,
            "time_of_day": 0.0,
            "action_category": current_state.get("proposed_action", {}).get("action_type", "none"),
            "exercise_duration": 0.0,
            "exercise_intensity": 0.0,
        }
    ])

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
