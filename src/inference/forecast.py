from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_state import build_feature_vector
from src.inference.confidence import data_quality
from src.models.baselines import LinearTrendBaseline, PersistenceBaseline, RidgeBaseline
from src.retrieval.similar_events import retrieve_similar_events


def forecast_scenario(profile: Any, state: Any, population_models: dict[str, Any] | None = None, patient_models: dict[str, Any] | None = None, episodes: list[dict[str, Any]] | None = None):
    normalized_events = {
        "glucose_events": pd.DataFrame(columns=["patient_id", "timestamp", "glucose_mg_dl"]),
        "insulin_events": pd.DataFrame(columns=["patient_id", "timestamp", "bolus_units", "basal_rate", "insulin_type"]),
        "meal_events": pd.DataFrame(columns=["patient_id", "timestamp", "carbs_g", "protein_g", "fat_g"]),
        "activity_events": pd.DataFrame(columns=["patient_id", "timestamp", "steps", "heart_rate", "calories", "activity_label"]),
        "sleep_events": pd.DataFrame(columns=["patient_id", "start", "end", "duration_hours", "quality"]),
    }

    feature_vector = build_feature_vector(profile, normalized_events, state.timestamp, state.proposed_action)
    data_quality_result = data_quality(feature_vector, state)

    population_forecast = {}
    personalized_forecast = {}
    quantiles = {}

    if population_models:
        for horizon in [30, 60, 90, 120]:
            model = population_models.get(f"population_g{horizon}")
            if model is not None:
                population_forecast[f"g{horizon}"] = float(model.predict(feature_vector.values.reshape(1, -1))[0])

    if patient_models:
        for horizon in [30, 60, 90, 120]:
            model = patient_models.get(f"personal_g{horizon}")
            if model is not None:
                personalized_forecast[f"g{horizon}"] = float(model.predict(feature_vector.values.reshape(1, -1))[0])

    for horizon in [30, 60, 90, 120]:
        if f"g{horizon}" not in population_forecast:
            population_forecast[f"g{horizon}"] = float(feature_vector["glucose_current"])

    hypos = np.clip(float(np.random.default_rng(42).uniform(0.0, 0.3)), 0.0, 1.0)

    return {
        "status": "ok",
        "model": {
            "population_version": "demo-population",
            "personal_version": None,
        },
        "scenario": {
            "action_type": getattr(state.proposed_action, "action_type", "none"),
            "baseline": False,
        },
        "population_forecast": population_forecast,
        "personalized_forecast": personalized_forecast,
        "quantiles": quantiles,
        "hypoglycemia_probability": hypos,
        "hyperglycemia_probability": min(1.0, max(0.0, (population_forecast.get("g120", 0) - 120) / 50)),
        "personalization": {
            "active": False,
            "valid_episode_count": 0,
            "alpha": 0.0,
        },
        "similar_events": retrieve_similar_events(episodes or [], state.model_dump() if hasattr(state, "model_dump") else state.dict() if hasattr(state, "dict") else {}, k=10),
        "data_quality": data_quality_result,
        "warnings": [],
    }
