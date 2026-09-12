from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BaseDatasetAdapter:
    """Generic adapter for any dataset source."""

    source_name: str = "base"
    root_dir: Path | str | None = None

    def __post_init__(self):
        if self.root_dir is None:
            self.root_dir = Path("data/raw")
        self.root_dir = Path(self.root_dir)

    def load_raw(self) -> dict[str, pd.DataFrame]:
        return {
            "glucose_events": pd.DataFrame(columns=["patient_id", "timestamp", "glucose_mg_dl"]),
            "insulin_events": pd.DataFrame(columns=["patient_id", "timestamp", "bolus_units", "basal_rate", "insulin_type"]),
            "meal_events": pd.DataFrame(columns=["patient_id", "timestamp", "carbs_g", "protein_g", "fat_g"]),
            "activity_events": pd.DataFrame(columns=["patient_id", "timestamp", "steps", "heart_rate", "calories", "activity_label"]),
            "sleep_events": pd.DataFrame(columns=["patient_id", "start", "end", "duration_hours", "quality"]),
        }

    def normalize(self) -> dict[str, pd.DataFrame]:
        frames = self.load_raw()
        for key, frame in frames.items():
            frame["patient_id"] = frame.get("patient_id", pd.Series(dtype="object"))
        return frames

    def get_patient_ids(self) -> list[str]:
        frames = self.normalize()
        ids = []
        for frame in frames.values():
            if not frame.empty:
                ids.extend(frame["patient_id"].dropna().unique().tolist())
        return sorted(set(ids))
