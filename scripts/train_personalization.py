"""Persist one general residual Ridge model per patient and horizon."""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.io import read_table
from src.features.build_state import FEATURE_NAMES
from src.models.personalization import train_personal_model
from src.models.registry import ModelRegistry

if __name__=="__main__":
    path=ROOT/"data/processed/hybrid_training_episodes.parquet"
    if not path.exists(): raise SystemExit("Hybrid episodes missing. Run: python3 scripts/train_full_poc.py")
    df=read_table(path); pop={p.stem:joblib.load(p) for p in (ROOT/"artifacts/population/models").glob("population_g*.joblib")}; registry=ModelRegistry(ROOT/"artifacts")
    for pid, rows in df.groupby("patient_id"):
        models={}
        for h in (30,60,90,120):
            fit=rows.copy(); fit["residual_g"+str(h)]=fit[f"y{h}"]-pop[f"population_g{h}"].predict(fit[FEATURE_NAMES].fillna(0)); models[f"personal_g{h}"]=train_personal_model(fit,FEATURE_NAMES,f"residual_g{h}")
        registry.save_patient(pid,models,{"model_type":"general_ridge_residual","feature_names":FEATURE_NAMES,"episode_count":len(rows),"hidden_traits_excluded":True,"clinical_validation":False})
    print(f"Saved personal residual models for {df.patient_id.nunique()} patients.")
