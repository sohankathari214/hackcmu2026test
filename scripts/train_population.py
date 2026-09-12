from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.population import train_and_save_population
from src.features.build_state import build_feature_vector


def build_synthetic_training_frame():
    rows = []
    for patient_id in ["patient_001", "patient_002"]:
        for idx in range(20):
            timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=idx * 15)
            glucose = 120 + idx * 0.5
            rows.append({
                "patient_id": patient_id,
                "timestamp": timestamp,
                "glucose_mg_dl": glucose,
                "age_years": 30,
                "weight_kg": 65,
                "years_since_diagnosis": 10,
                "insulin_delivery_pump": 1,
                "insulin_delivery_injection": 0,
                "insulin_delivery_other": 0,
                "insulin_delivery_unknown": 0,
                "action_type_none": 1,
                "action_type_exercise": 0,
                "action_type_meal": 0,
                "action_type_caffeine": 0,
                "action_type_alcohol": 0,
                "action_type_other": 0,
                "glucose_current": glucose,
                "glucose_lag_15": glucose,
                "glucose_lag_30": glucose,
                "glucose_lag_60": glucose,
                "glucose_lag_120": glucose,
                "glucose_delta_15": 0,
                "glucose_delta_30": 0,
                "glucose_delta_60": 0,
                "glucose_mean_30": glucose,
                "glucose_mean_60": glucose,
                "glucose_mean_120": glucose,
                "glucose_std_30": 0,
                "glucose_std_60": 0,
                "glucose_std_120": 0,
                "glucose_min_60": glucose,
                "glucose_max_60": glucose,
                "bolus_units_30": 0,
                "bolus_units_60": 0,
                "bolus_units_120": 0,
                "bolus_units_240": 0,
                "time_since_last_bolus": 0,
                "basal_rate_current": 0,
                "basal_units_60": 0,
                "basal_units_120": 0,
                "carbs_30": 0,
                "carbs_60": 0,
                "carbs_120": 0,
                "carbs_240": 0,
                "time_since_last_meal": 0,
                "last_meal_carbs": 0,
                "steps_15": 0,
                "steps_60": 0,
                "steps_120": 0,
                "current_hr": 70,
                "mean_hr_15": 70,
                "mean_hr_60": 70,
                "activity_minutes_30": 0,
                "activity_minutes_120": 0,
                "sleep_duration_last_night": 8,
                "sleep_quality_numeric": 1,
                "minutes_since_waking": 0,
                "hour_sin": 0,
                "hour_cos": 0,
                "day_sin": 0,
                "day_cos": 0,
                "y30": glucose + 5,
                "y60": glucose + 10,
                "y90": glucose + 15,
                "y120": glucose + 20,
                "hypo_120": 0,
                "hyper_120": 0,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    training_df = build_synthetic_training_frame()
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text()) if Path("configs/base.yaml").exists() else {}
    feature_names = [c for c in training_df.columns if c not in ["patient_id", "timestamp", "y30", "y60", "y90", "y120", "hypo_120", "hyper_120"]]
    train_and_save_population(training_df, cfg, feature_names)
    print("Population models trained and saved to artifacts/population")
