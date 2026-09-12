from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.risk import train_risk_models


if __name__ == "__main__":
    feature_names = [
        "glucose_current", "glucose_lag_15", "glucose_lag_30", "glucose_lag_60", "glucose_lag_120",
        "glucose_delta_15", "glucose_delta_30", "glucose_delta_60", "glucose_mean_30", "glucose_mean_60",
        "glucose_mean_120", "glucose_std_30", "glucose_std_60", "glucose_std_120", "glucose_min_60", "glucose_max_60",
        "bolus_units_30", "bolus_units_60", "bolus_units_120", "bolus_units_240", "time_since_last_bolus",
        "basal_rate_current", "basal_units_60", "basal_units_120", "carbs_30", "carbs_60", "carbs_120", "carbs_240",
        "time_since_last_meal", "last_meal_carbs", "steps_15", "steps_60", "steps_120", "current_hr", "mean_hr_15",
        "mean_hr_60", "activity_minutes_30", "activity_minutes_120", "sleep_duration_last_night", "sleep_quality_numeric",
        "minutes_since_waking", "hour_sin", "hour_cos", "day_sin", "day_cos", "age_years", "weight_kg",
        "years_since_diagnosis", "insulin_delivery_pump", "insulin_delivery_injection", "insulin_delivery_other",
        "insulin_delivery_unknown", "action_type_none", "action_type_exercise", "action_type_meal", "action_type_caffeine",
        "action_type_alcohol", "action_type_other", "exercise_category_cardio", "exercise_category_resistance",
        "exercise_category_mixed", "exercise_category_other", "exercise_duration_minutes", "exercise_intensity_numeric",
        "exercise_start_delay_minutes",
    ]

    df = pd.read_csv("data/processed/glucose_events.csv")
    if df.empty:
        df = pd.DataFrame({
            "glucose_current": [120, 122],
            "hypo_120": [0, 1],
        })
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0

    cfg = yaml.safe_load(Path("configs/base.yaml").read_text()) if Path("configs/base.yaml").exists() else {}
    models, metrics = train_risk_models(df, cfg, feature_names)
    print("Risk models trained:", list(models.keys()))
    print("Metrics:", metrics)
