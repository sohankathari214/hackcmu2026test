from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base_adapter import BaseDatasetAdapter


class OhioAdapter(BaseDatasetAdapter):
    """Interface stub for Ohio data.

    The raw Ohio dataset is not bundled with this repository. This adapter exposes the
    expected contract so downstream code can normalize Ohio sources once those files are made available.
    """

    def __init__(self, root_dir: str | Path | None = None, mapping: dict[str, Any] | None = None):
        super().__init__(source_name="ohio", root_dir=root_dir)
        self.mapping = mapping or {}

    def load_raw(self) -> dict[str, pd.DataFrame]:
        root = Path(self.root_dir)
        if not root.exists():
            raise FileNotFoundError(
                f"Ohio data directory not found: {root}. "
                "Place the Ohio dataset under data/raw/ohio/ or update the adapter configuration."
            )

        glucose_path = root / "glucose.csv"
        insulin_path = root / "insulin.csv"
        meal_path = root / "meals.csv"
        activity_path = root / "activity.csv"
        sleep_path = root / "sleep.csv"

        if not any(path.exists() for path in [glucose_path, insulin_path, meal_path, activity_path, sleep_path]):
            raise FileNotFoundError(
                f"Ohio files not found under {root}. Expected at least one of the normalized event CSVs."
            )

        return {
            "glucose_events": pd.read_csv(glucose_path) if glucose_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "glucose_mg_dl"]),
            "insulin_events": pd.read_csv(insulin_path) if insulin_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "bolus_units", "basal_rate", "insulin_type"]),
            "meal_events": pd.read_csv(meal_path) if meal_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "carbs_g", "protein_g", "fat_g"]),
            "activity_events": pd.read_csv(activity_path) if activity_path.exists() else pd.DataFrame(columns=["patient_id", "timestamp", "steps", "heart_rate", "calories", "activity_label"]),
            "sleep_events": pd.read_csv(sleep_path) if sleep_path.exists() else pd.DataFrame(columns=["patient_id", "start", "end", "duration_hours", "quality"]),
        }
