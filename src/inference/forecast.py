from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from src.features.build_state import FEATURE_NAMES, build_feature_vector
from src.inference.confidence import data_quality
from src.retrieval.similar_events import retrieve_similar_events
from src.models.personalization import alpha_for_episode_count

def _events(profile,state):
    ts=pd.Timestamp(state.timestamp); pid=profile.patient_id; glucose=[{"patient_id":pid,"timestamp":ts,"glucose_mg_dl":state.glucose.get("current_mg_dl",state.glucose.get("mg_dl"))}]
    glucose += [{"patient_id":pid,"timestamp":ts-pd.Timedelta(minutes=r.get("minutes_ago",0)),"glucose_mg_dl":r.get("mg_dl")} for r in state.glucose.get("recent_readings",[])]
    ins=[{"patient_id":pid,"timestamp":ts-pd.Timedelta(minutes=d.get("minutes_ago",0)),"bolus_units":d.get("units"),"basal_rate":d.get("basal_rate_units_hr",state.insulin.get("basal_rate_units_hr"))} for d in state.insulin.get("recent_doses",[])]
    meals=[{"patient_id":pid,"timestamp":ts-pd.Timedelta(minutes=x.get("minutes_ago",0)),"carbs_g":x.get("carbs_g")} for x in state.food.get("recent_meals",[])]
    act=[{"patient_id":pid,"timestamp":ts,"steps":state.activity.get("recent_steps"),"heart_rate":state.activity.get("current_hr"),"calories":np.nan}]
    return {"glucose_events":pd.DataFrame(glucose),"insulin_events":pd.DataFrame(ins,columns=["patient_id","timestamp","bolus_units","basal_rate"]),"meal_events":pd.DataFrame(meals,columns=["patient_id","timestamp","carbs_g"]),"activity_events":pd.DataFrame(act)}

def _load_models():
    d=Path("artifacts/population/models")
    if not d.exists(): raise FileNotFoundError("Population models are absent. Run: python3 scripts/train_full_poc.py")
    return {p.stem:joblib.load(p) for p in d.glob("*.joblib")}

def _load_patient_models(patient_id):
    d=Path("artifacts/patients")/patient_id
    return ({p.stem:joblib.load(p) for p in d.glob("personal_g*.joblib")} if d.exists() else {})

def forecast_scenario(profile:Any,state:Any,population_models=None,patient_models=None,episodes=None):
    models=population_models or _load_models(); f=build_feature_vector(profile,_events(profile,state),state.timestamp,state.proposed_action).reindex(FEATURE_NAMES).fillna(0); fc={}; personal=patient_models or _load_patient_models(profile.patient_id); count=len(episodes or []); alpha=alpha_for_episode_count(count)
    for h in (30,60,90,120):
        m=models.get(f"population_g{h}")
        if m is None: raise FileNotFoundError(f"Missing population_g{h}; run python3 scripts/train_full_poc.py")
        fc[f"g{h}"]=float(m.predict(f.to_frame().T)[0])
    quant={f"g{h}":{q:float(models[f"g{h}_{q}"].predict(f.to_frame().T)[0]) for q in ("q10","q50","q90") if f"g{h}_{q}" in models} for h in (30,60,90,120)}
    action=state.proposed_action.model_dump() if hasattr(state.proposed_action,"model_dump") else state.proposed_action; raw_type=action.get("action_type","none"); typ=getattr(raw_type,"value",str(raw_type)); warning="Action response learned from proof-of-concept synthetic augmentation." if typ!="none" else None
    corrected={f"g{h}":float(fc[f"g{h}"]+alpha*personal[f"personal_g{h}"].predict(f.to_frame().T)[0]) for h in (30,60,90,120) if f"personal_g{h}" in personal}
    return {"status":"ok","scenario":{"action_type":typ,"baseline":typ=="none"},"population_forecast":fc,"personalized_forecast":corrected,"quantiles":quant,"hypoglycemia_probability":float(models["hypo_120"].predict_proba(f.to_frame().T)[0,1]) if "hypo_120" in models else None,"hyperglycemia_probability":float(models["hyper_120"].predict_proba(f.to_frame().T)[0,1]) if "hyper_120" in models else None,"personalization":{"active":bool(corrected),"episode_count":count,"alpha":alpha,"personal_model_version":"general-ridge-residual" if corrected else None},"support":{"supported":True,"support_type":"real_hupa" if typ=="none" else "synthetic_poc"},"data_quality":data_quality(f,state),"warnings":[warning] if warning else [],"provenance":{"real_physiology":"HUPA-UCM","action_augmentation":"synthetic_poc" if typ!="none" else None},"similar_events":retrieve_similar_events(episodes or [],{},k=10)}
