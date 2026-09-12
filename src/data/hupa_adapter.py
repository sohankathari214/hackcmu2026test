"""Adapter for the semicolon-delimited HUPA-UCM preprocessed exports."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from .base_adapter import BaseDatasetAdapter

REQUIRED_COLUMNS = {"time", "glucose", "calories", "heart_rate", "steps", "basal_rate", "bolus_volume_delivered", "carb_input"}

class HUPAAdapter(BaseDatasetAdapter):
    def __init__(self, root_dir: str | Path = "data/raw/Preprocessed", config: dict[str, Any] | None = None, mapping: dict[str, Any] | None = None):
        super().__init__(source_name="hupa", root_dir=root_dir)
        self.config, self.mapping = config or {}, mapping or {}

    def discover_files(self) -> list[Path]:
        return sorted(Path(self.root_dir).glob("HUPA*P.csv"))

    def load_preprocessed(self) -> pd.DataFrame:
        files = self.discover_files()
        if not files:
            raise FileNotFoundError(f"No HUPA preprocessed patient CSVs found in {self.root_dir}")
        frames, delimiter = [], self.config.get("delimiter", ";")
        for path in files:
            raw = pd.read_csv(path, sep=delimiter)
            missing = REQUIRED_COLUMNS - set(raw.columns)
            if missing:
                raise ValueError(f"{path.name} is missing required HUPA columns: {sorted(missing)}")
            out = raw.rename(columns={"time":"timestamp", "glucose":"glucose_mg_dl", "heart_rate":"heart_rate_bpm", "bolus_volume_delivered":"bolus_raw", "carb_input":"carb_input_raw", "basal_rate":"basal_raw"}).copy()
            out.insert(0, "patient_id", path.stem)
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
            for col in ["glucose_mg_dl","calories","heart_rate_bpm","steps","basal_raw","bolus_raw","carb_input_raw"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            carb_cfg = self.config.get("carb_input", {})
            out["carbs_g"] = out["carb_input_raw"] * float(carb_cfg.get("grams_per_serving", 1)) if carb_cfg.get("mode") == "servings" else pd.NA
            out["bolus_units"] = out["bolus_raw"] if self.config.get("bolus", {}).get("assume_units", False) else pd.NA
            out["basal_value"] = out["basal_raw"]
            frames.append(out[["patient_id","timestamp","glucose_mg_dl","calories","heart_rate_bpm","steps","basal_raw","basal_value","bolus_raw","bolus_units","carb_input_raw","carbs_g"]].sort_values("timestamp"))
        return pd.concat(frames, ignore_index=True).sort_values(["patient_id","timestamp"]).reset_index(drop=True)

    def load_raw(self) -> dict[str, pd.DataFrame]:
        table = self.load_preprocessed()
        insulin = table[["patient_id","timestamp","bolus_units","basal_value","bolus_raw","basal_raw"]].rename(columns={"basal_value":"basal_rate"})
        meals = table[["patient_id","timestamp","carbs_g","carb_input_raw"]].copy(); meals["protein_g"] = pd.NA; meals["fat_g"] = pd.NA
        activity = table[["patient_id","timestamp","steps","calories","heart_rate_bpm"]].rename(columns={"heart_rate_bpm":"heart_rate"}); activity["activity_label"] = pd.NA
        return {"glucose_events":table[["patient_id","timestamp","glucose_mg_dl"]], "insulin_events":insulin, "meal_events":meals, "activity_events":activity, "sleep_events":pd.DataFrame(columns=["patient_id","start","end","duration_hours","quality"])}
