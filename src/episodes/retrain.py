"""Train persistent general Ridge residual models from finalized feature snapshots."""
from __future__ import annotations
import pandas as pd
from src.features.build_state import FEATURE_NAMES
from src.models.personalization import train_personal_model, alpha_for_episode_count
from src.models.registry import ModelRegistry

def retrain_from_episodes(patient_id, episodes):
    usable=[x for x in episodes if x.get("quality",{}).get("usable_for_personalization") and x.get("feature_vector")]
    if len(usable)<10: raise ValueError("At least 10 finalized episodes with feature snapshots are required.")
    frame=pd.DataFrame([{**x["feature_vector"],**x.get("residuals",{})} for x in usable]); models={}
    for h in (30,60,90,120):
        target=f"r{h}"
        if target in frame and frame[target].notna().sum()>=10: models[f"personal_g{h}"]=train_personal_model(frame,FEATURE_NAMES,target)
    if not models: raise ValueError("Episodes have no usable horizon residuals.")
    ModelRegistry().save_patient(patient_id,models,{"model_type":"general_ridge_residual","episode_count":len(usable),"feature_names":FEATURE_NAMES,"hidden_traits_excluded":True,"clinical_validation":False})
    return {"status":"retrained","patient_id":patient_id,"models":sorted(models),"valid_episode_count":len(usable),"alpha":alpha_for_episode_count(len(usable))}
