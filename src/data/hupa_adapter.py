from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base_adapter import BaseDatasetAdapter


class HUPAAdapter(BaseDatasetAdapter):
    """A generic placeholder adapter for HUPA-like data.

    This adapter intentionally supports configurable paths and column mapping while
    failing gracefully if no HUPA files exist in the local workspace.
    """

    def __init__(self, root_dir: str | Path | None = None, mapping: dict[str, Any] | None = None):
        super().__init__(source_name="hupa", root_dir=root_dir)
        self.mapping = mapping or {}

    def load_raw(self) -> dict[str, pd.DataFrame]:
        root = Path(self.root_dir)
        if not root.exists():
            raise FileNotFoundError(
                f"HUPA data directory not found: {root}. "
                "Place HUPA files under data/raw/hupa/ or update the adapter paths in configuration."
            )

        glucose_path = root / "glucose.csv"
        insulin_path = root / "insulin.csv"
        meal_path = root / "meals.csv"
        activity_path = root / "activity.csv"
        sleep_path = root / "sleep.csv"

        if not glucose_path.exists() and not insulin_path.exists() and not meal_path.exists():
            raise FileNotFoundError(
                f"No HUPA-like files found in {root}. Expected glucose.csv, insulin.csv, meals.csv, activity.csv, and sleep.csv."
            )

        glucose = pd.read_csv(glucose_path) if glucose_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "glucose_mg_dl"])
        insulin = pd.read_csv(insulin_path) if insulin_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "bolus_units", "basal_rate", "insulin_type"])
        meal = pd.read_csv(meal_path) if meal_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "carbs_g", "protein_g", "fat_g"])
        activity = pd.read_csv(activity_path) if activity_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "steps", "heart_rate", "calories", "activity_label"])
        sleep = pd.read_csv(sleep_path) if sleep_path.exists() else pd.DataFrame(columns=["patient_id", "start", "end", "duration_hours", "quality"])

        frames = {
            "glucose_events": self._normalize_glucose(glucose),
            "insulin_events": self._normalize_insulin(insulin),
            "meal_events": self._normalize_meal(meal),
            "activity_events": self._normalize_activity(activity),
            "sleep_events": self._normalize_sleep(sleep),
        }
        return frames

    def _normalize_glucose(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out = out.rename(columns={
            self.mapping.get("glucose_patient_id", "patient_id"): "patient_id",
            self.mapping.get("glucose_timestamp", "timestamp"): "timestamp",
            self.mapping.get("glucose_value", "glucose_mg_dl"): "glucose_mg_dl",
        })
        return out[["patient_id", "timestamp", "glucose_mg_dl"]]

    def _normalize_insulin(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out = out.rename(columns={
            self.mapping.get("insulin_patient_id", "patient_id"): "patient_id",
            self.mapping.get("insulin_timestamp", "timestamp"): "timestamp",
            self.mapping.get("insulin_bolus_units", "bolus_units"): "bolus_units",
            self.mapping.get("insulin_basal_rate", "basal_rate"): "basal_rate",
            self.mapping.get("insulin_type", "insulin_type"): "insulin_type",
        })
        return out[["patient_id", "timestamp", "bolus_units", "basal_rate", "insulin_type"]]

    def _normalize_meal(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out = out.rename(columns={
            self.mapping.get("meal_patient_id", "patient_id"): "patient_id",
            self.mapping.get("meal_timestamp", "timestamp"): "timestamp",
            self.mapping.get("meal_carbs", "carbs_g"): "carbs_g",
            self.mapping.get("meal_protein", "protein_g"): "protein_g",
            self.mapping.get("meal_fat", "fat_g"): "fat_g",
        })
        return out[["patient_id", "timestamp", "carbs_g", "protein_g", "fat_g"]]

    def _normalize_activity(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out = out.rename(columns={
            self.mapping.get("activity_patient_id", "patient_id"): "patient_id",
            self.mapping.get("activity_timestamp", "timestamp"): "timestamp",
            self.mapping.get("activity_steps", "steps"): "steps",
            self.mapping.get("activity_heart_rate", "heart_rate"): "heart_rate",
            self.mapping.get("activity_calories", "calories"): "calories",
            self.mapping.get("activity_label", "activity_label"): "activity_label",
        })
        return out[["patient_id", "timestamp", "steps", "heart_rate", "calories", "activity_label"]]

    def _normalize_sleep(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out = out.rename(columns={
            self.mapping.get("sleep_patient_id", "patient_id"): "patient_id",
            self.mapping.get("sleep_start", "start"): "start",
            self.mapping.get("sleep_end", "end"): "end",
            self.mapping.get("sleep_duration", "duration_hours"): "duration_hours",
            self.mapping.get("sleep_quality", "quality"): "quality",
        })
        return out[["patient_id", "start", "end", "duration_hours", "quality"]]
