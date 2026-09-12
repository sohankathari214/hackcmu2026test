from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.schemas.current_state import ActionType


def build_feature_vector(
    medical_profile: Any,
    normalized_events: dict[str, pd.DataFrame],
    timestamp: datetime,
    proposed_action: Any,
) -> pd.Series:
    """Build a timestamp/window-based feature vector from normalized event tables."""

    timestamp = pd.Timestamp(timestamp)
    feature_names = []
    feature_values = []

    glucose_frame = normalized_events.get("glucose_events", pd.DataFrame(columns=["patient_id", "timestamp", "glucose_mg_dl"]))
    insulin_frame = normalized_events.get("insulin_events", pd.DataFrame(columns=["patient_id", "timestamp", "bolus_units", "basal_rate", "insulin_type"]))
    meal_frame = normalized_events.get("meal_events", pd.DataFrame(columns=["patient_id", "timestamp", "carbs_g", "protein_g", "fat_g"]))
    activity_frame = normalized_events.get("activity_events", pd.DataFrame(columns=["patient_id", "timestamp", "steps", "heart_rate", "calories", "activity_label"]))
    sleep_frame = normalized_events.get("sleep_events", pd.DataFrame(columns=["patient_id", "start", "end", "duration_hours", "quality"]))

    patient_id = getattr(medical_profile, "patient_id", None)

    if patient_id:
        glucose_frame = glucose_frame[glucose_frame["patient_id"] == patient_id]
        insulin_frame = insulin_frame[insulin_frame["patient_id"] == patient_id]
        meal_frame = meal_frame[meal_frame["patient_id"] == patient_id]
        activity_frame = activity_frame[activity_frame["patient_id"] == patient_id]
        sleep_frame = sleep_frame[sleep_frame["patient_id"] == patient_id]

    glucose_frame = glucose_frame.sort_values("timestamp")
    insulin_frame = insulin_frame.sort_values("timestamp")
    meal_frame = meal_frame.sort_values("timestamp")
    activity_frame = activity_frame.sort_values("timestamp")

    def nearest_prior(frame, col, ts, window=None):
        if frame.empty:
            return np.nan
        frame = frame[frame["timestamp"] <= ts].copy()
        if frame.empty:
            return np.nan
        if window is not None:
            frame = frame[frame["timestamp"] >= ts - pd.Timedelta(window)]
        if frame.empty:
            return np.nan
        return float(frame.iloc[-1][col])

    def window_mean(frame, col, start, end, ts):
        if frame.empty:
            return np.nan
        window = frame[(frame["timestamp"] >= ts - pd.Timedelta(end)) & (frame["timestamp"] <= ts - pd.Timedelta(start))]
        if window.empty:
            return np.nan
        return float(window[col].mean())

    def window_sum(frame, col, start, end, ts):
        if frame.empty:
            return np.nan
        window = frame[(frame["timestamp"] >= ts - pd.Timedelta(end)) & (frame["timestamp"] <= ts)]
        if window.empty:
            return np.nan
        return float(window[col].sum())

    def time_since_last(frame, ts):
        if frame.empty:
            return np.nan
        prior = frame[frame["timestamp"] <= ts]
        if prior.empty:
            return np.nan
        return float((ts - prior.iloc[-1]["timestamp"]).total_seconds() / 60)

    def get_recent_activity_minutes(ts):
        if activity_frame.empty:
            return np.nan
        prior = activity_frame[activity_frame["timestamp"] <= ts]
        if prior.empty:
            return np.nan
        recent = prior[prior["timestamp"] >= ts - pd.Timedelta(minutes=120)]
        if recent.empty:
            return np.nan
        return float(recent["steps"].sum()) if "steps" in recent.columns and recent["steps"].notna().any() else np.nan

    glucose_current = nearest_prior(glucose_frame, "glucose_mg_dl", timestamp)
    feature_names += ["glucose_current"]
    feature_values += [glucose_current]

    for lag in [15, 30, 60, 120]:
        value = nearest_prior(glucose_frame, "glucose_mg_dl", timestamp, window=pd.Timedelta(minutes=lag))
        feature_names.append(f"glucose_lag_{lag}")
        feature_values.append(value)

    for delta in [15, 30, 60]:
        current = nearest_prior(glucose_frame, "glucose_mg_dl", timestamp, window=pd.Timedelta(minutes=delta))
        prior = nearest_prior(glucose_frame, "glucose_mg_dl", timestamp - pd.Timedelta(minutes=delta), window=pd.Timedelta(minutes=delta))
        delta_value = np.nan
        if not (pd.isna(current) or pd.isna(prior)):
            delta_value = current - prior
        feature_names.append(f"glucose_delta_{delta}")
        feature_values.append(delta_value)

    for window in [30, 60, 120]:
        feature_names.append(f"glucose_mean_{window}")
        feature_values.append(window_mean(glucose_frame, "glucose_mg_dl", 0, window, timestamp))
        feature_names.append(f"glucose_std_{window}")
        std_window = glucose_frame[(glucose_frame["timestamp"] >= timestamp - pd.Timedelta(minutes=window)) & (glucose_frame["timestamp"] <= timestamp)]
        if std_window.empty or len(std_window) < 2:
            feature_values.append(np.nan)
        else:
            feature_values.append(float(std_window["glucose_mg_dl"].std()))

    recent_60 = glucose_frame[(glucose_frame["timestamp"] >= timestamp - pd.Timedelta(minutes=60)) & (glucose_frame["timestamp"] <= timestamp)]
    feature_names += ["glucose_min_60", "glucose_max_60"]
    feature_values += [np.nanmin(recent_60["glucose_mg_dl"]) if not recent_60.empty else np.nan, np.nanmax(recent_60["glucose_mg_dl"]) if not recent_60.empty else np.nan]

    for window in [30, 60, 120, 240]:
        feature_names.append(f"bolus_units_{window}")
        feature_values.append(window_sum(insulin_frame, "bolus_units", 0, window, timestamp))

    feature_names += ["time_since_last_bolus", "basal_rate_current"]
    feature_values += [time_since_last(insulin_frame, timestamp), nearest_prior(insulin_frame, "basal_rate", timestamp)]

    feature_names += ["basal_units_60", "basal_units_120"]
    feature_values += [window_sum(insulin_frame, "basal_rate", 0, 60, timestamp), window_sum(insulin_frame, "basal_rate", 0, 120, timestamp)]

    if "reported_iob_units" in insulin_frame.columns:
        feature_names.append("reported_iob_units")
        feature_values.append(nearest_prior(insulin_frame, "reported_iob_units", timestamp))

    for window in [30, 60, 120, 240]:
        feature_names.append(f"carbs_{window}")
        feature_values.append(window_sum(meal_frame, "carbs_g", 0, window, timestamp))

    feature_names += ["time_since_last_meal", "last_meal_carbs"]
    feature_values += [time_since_last(meal_frame, timestamp), nearest_prior(meal_frame, "carbs_g", timestamp)]

    for window in [15, 60, 120]:
        feature_names.append(f"steps_{window}")
        vals = activity_frame[(activity_frame["timestamp"] >= timestamp - pd.Timedelta(minutes=window)) & (activity_frame["timestamp"] <= timestamp)]
        feature_values.append(float(vals["steps"].sum()) if not vals.empty and vals["steps"].notna().any() else np.nan)

    if not activity_frame.empty:
        recent_hr = activity_frame[(activity_frame["timestamp"] <= timestamp)]["heart_rate"]
        feature_names += ["current_hr", "mean_hr_15", "mean_hr_60"]
        feature_values += [float(recent_hr.iloc[-1]) if not recent_hr.empty else np.nan, float(activity_frame[(activity_frame["timestamp"] >= timestamp - pd.Timedelta(minutes=15)) & (activity_frame["timestamp"] <= timestamp)]["heart_rate"].mean()) if not activity_frame.empty else np.nan, float(activity_frame[(activity_frame["timestamp"] >= timestamp - pd.Timedelta(minutes=60)) & (activity_frame["timestamp"] <= timestamp)]["heart_rate"].mean()) if not activity_frame.empty else np.nan]

    feature_names += ["activity_minutes_30", "activity_minutes_120"]
    feature_values += [np.nan, np.nan]

    if "sleep_events" in normalized_events:
        sleep_prior = sleep_frame[sleep_frame["end"] <= timestamp]
        if not sleep_prior.empty:
            last_sleep = sleep_prior.iloc[-1]
            feature_names.append("sleep_duration_last_night")
            feature_values.append(float(last_sleep.get("duration_hours", np.nan)))
            if "quality" in last_sleep and pd.notna(last_sleep.get("quality")):
                try:
                    quality_numeric = float(last_sleep["quality"])
                except Exception:
                    quality_numeric = np.nan
                feature_names.append("sleep_quality_numeric")
                feature_values.append(quality_numeric)
            feature_names.append("minutes_since_waking")
            feature_values.append(float((timestamp - pd.Timestamp(last_sleep["end"])).total_seconds() / 60))
        else:
            feature_names += ["sleep_duration_last_night", "sleep_quality_numeric", "minutes_since_waking"]
            feature_values += [np.nan, np.nan, np.nan]

    for name in ["age_years", "weight_kg", "years_since_diagnosis"]:
        feature_names.append(name)
        feature_values.append(medical_profile.demographics.get(name) if hasattr(medical_profile, "demographics") else np.nan)

    for delivery in ["pump", "injection", "other", "unknown"]:
        feature_names.append(f"insulin_delivery_{delivery}")
        feature_values.append(1.0 if getattr(medical_profile.diabetes, "insulin_delivery", None) == delivery else 0.0)

    if isinstance(proposed_action, dict):
        action_type = proposed_action.get("action_type", ActionType.NONE.value)
    else:
        action_type = getattr(proposed_action, "action_type", ActionType.NONE.value)

    for action in ["none", "exercise", "meal", "caffeine", "alcohol", "other"]:
        feature_names.append(f"action_type_{action}")
        feature_values.append(1.0 if action_type == action else 0.0)

    if action_type == "exercise":
        exercise = proposed_action.get("exercise", {}) if isinstance(proposed_action, dict) else getattr(proposed_action, "exercise", {})
        exercise_dict = exercise.model_dump() if hasattr(exercise, "model_dump") else exercise
        if not isinstance(exercise_dict, dict):
            exercise_dict = {}
        for category in ["cardio", "resistance", "mixed", "other"]:
            feature_names.append(f"exercise_category_{category}")
            feature_values.append(1.0 if str(exercise_dict.get("category", "")).lower() == category else 0.0)
        feature_names += ["exercise_duration_minutes", "exercise_intensity_numeric", "exercise_start_delay_minutes"]
        feature_values += [exercise_dict.get("duration_minutes", np.nan), {
            "low": 1,
            "moderate": 2,
            "high": 3,
            "unknown": np.nan,
        }.get(str(exercise_dict.get("intensity", "unknown")).lower(), np.nan), exercise_dict.get("start_delay_minutes", np.nan)]

    unrecognized = []
    if len(feature_names) != len(feature_values):
        raise RuntimeError("Feature vector build failed: names and values length mismatch")

    feature_series = pd.Series(feature_values, index=feature_names, dtype="float64")
    feature_series.index = feature_series.index.astype(str)
    return feature_series
