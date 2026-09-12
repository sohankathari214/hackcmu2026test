from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.features.build_state import build_feature_vector
from src.schemas.medical_profile import MedicalProfile


class DummyProfile:
    patient_id = "patient_001"
    demographics = {"age_years": 30, "weight_kg": 70, "years_since_diagnosis": 10}
    diabetes = {"insulin_delivery": "pump"}


def test_build_feature_vector_counts_known_features():
    profile = DummyProfile()
    events = {
        "glucose_events": pd.DataFrame([
            {"patient_id": "patient_001", "timestamp": datetime(2026, 1, 1, 11, 45, tzinfo=timezone.utc), "glucose_mg_dl": 120},
            {"patient_id": "patient_001", "timestamp": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), "glucose_mg_dl": 125},
        ]),
        "insulin_events": pd.DataFrame(columns=["patient_id", "timestamp", "bolus_units", "basal_rate", "insulin_type"]),
        "meal_events": pd.DataFrame(columns=["patient_id", "timestamp", "carbs_g", "protein_g", "fat_g"]),
        "activity_events": pd.DataFrame(columns=["patient_id", "timestamp", "steps", "heart_rate", "calories", "activity_label"]),
        "sleep_events": pd.DataFrame(columns=["patient_id", "start", "end", "duration_hours", "quality"]),
    }
    state = {"action_type": "none"}
    features = build_feature_vector(profile, events, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), state)
    assert "glucose_current" in features.index
    assert features["glucose_current"] == 125
