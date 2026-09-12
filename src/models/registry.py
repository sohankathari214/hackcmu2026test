from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class ModelRegistry:
    def __init__(self, root_dir: str | Path = "artifacts"):
        self.root_dir = Path(root_dir)

    def _ensure_dirs(self):
        (self.root_dir / "population").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "patients").mkdir(parents=True, exist_ok=True)

    def save_population(self, models: dict[str, Any], metadata: dict[str, Any]):
        self._ensure_dirs()
        population_dir = self.root_dir / "population"
        for name, model in models.items():
            joblib.dump(model, population_dir / "models" / f"{name}.joblib")
        (population_dir / "models").mkdir(parents=True, exist_ok=True)
        (population_dir / "metadata").mkdir(parents=True, exist_ok=True)
        with open(population_dir / "metadata" / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_patient(self, patient_id: str, models: dict[str, Any], metadata: dict[str, Any]):
        self._ensure_dirs()
        patient_dir = self.root_dir / "patients" / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, patient_dir / f"{name}.joblib")
        with open(patient_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_latest(self, patient_id: str | None = None):
        if patient_id is None:
            return self.root_dir / "population"
        return self.root_dir / "patients" / patient_id

    def list_versions(self, patient_id: str | None = None):
        if patient_id is None:
            return sorted(str(p.name) for p in (self.root_dir / "population" / "models").glob("*.joblib"))
        return sorted(str(p.name) for p in (self.root_dir / "patients" / patient_id).glob("*.joblib"))

    def promote(self, patient_id: str, models: dict[str, Any], metadata: dict[str, Any]):
        self.save_patient(patient_id, models, metadata)

    def rollback(self, patient_id: str):
        patient_dir = self.root_dir / "patients" / patient_id
        if patient_dir.exists():
            return patient_dir
        return None
