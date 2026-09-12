"""Canonical, strictly historical feature construction for training and inference."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd

BASE_FEATURES = ["glucose_current", *[f"glucose_lag_{x}" for x in (15,30,60,120)], *[f"glucose_delta_{x}" for x in (15,30,60)], *[f"glucose_mean_{x}" for x in (30,60,120)], *[f"glucose_std_{x}" for x in (30,60,120)], "glucose_min_60","glucose_max_60", *[f"bolus_{x}" for x in (30,60,120,240)], "time_since_last_bolus", *[f"basal_sum_{x}" for x in (30,60,120)], "recent_insulin_exposure", *[f"carbs_{x}" for x in (30,60,120,240)], "time_since_last_carb_event","last_carb_amount", *[f"steps_{x}" for x in (15,30,60,120)], "heart_rate_current", *[f"heart_rate_mean_{x}" for x in (15,30,60)], *[f"heart_rate_max_{x}" for x in (30,60)], *[f"calories_{x}" for x in (30,60,120)], "hour_sin","hour_cos","day_sin","day_cos","minutes_since_latest_glucose","recent_glucose_count_60","missing_hr_fraction_60","missing_activity_fraction_60"]
ACTION_FEATURES = ["action_type_none","action_type_exercise","action_type_meal","action_type_caffeine","action_type_alcohol", "exercise_cardio","exercise_resistance","exercise_mixed","exercise_duration_minutes","exercise_intensity_numeric","exercise_start_delay_minutes", "caffeine_mg","alcohol_grams","alcohol_drink_count","stress_level","illness","hydration_low","hydration_normal","hydration_high","sleep_hours"]
FEATURE_NAMES = BASE_FEATURES + ACTION_FEATURES

def _dict(value: Any) -> dict:
    return value.model_dump() if hasattr(value,"model_dump") else value if isinstance(value,dict) else {}

def build_feature_vector(medical_profile: Any, normalized_events: dict[str,pd.DataFrame], timestamp: datetime, proposed_action: Any) -> pd.Series:
    ts=pd.Timestamp(timestamp)
    if ts.tz is not None: ts=ts.tz_localize(None)
    pid=getattr(medical_profile,"patient_id",None)
    def frame(name,cols):
        x=normalized_events.get(name,pd.DataFrame(columns=cols)).copy()
        if pid and "patient_id" in x: x=x[x.patient_id==pid]
        if "timestamp" in x:
            x["timestamp"]=pd.to_datetime(x.timestamp,errors="coerce",utc=True).dt.tz_localize(None)
            x=x[x.timestamp<=ts].sort_values("timestamp") # the no-future boundary
        return x
    g=frame("glucose_events",["timestamp","glucose_mg_dl"]); i=frame("insulin_events",["timestamp","bolus_units","basal_rate"]); m=frame("meal_events",["timestamp","carbs_g"]); a=frame("activity_events",["timestamp","steps","heart_rate","calories"])
    def prior(x,col,at=ts):
        z=x[x.timestamp<=at]; return float(z.iloc[-1][col]) if len(z) and pd.notna(z.iloc[-1].get(col)) else np.nan
    def win(x,col,w):
        z=x[(x.timestamp>=ts-pd.Timedelta(minutes=w))&(x.timestamp<=ts)]; return z[col] if col in z else pd.Series(dtype=float)
    def total(x,col,w):
        z=win(x,col,w); return float(z.sum()) if z.notna().any() else np.nan
    def since(x,col):
        z=x[x[col].fillna(0)>0] if col in x else x.iloc[0:0]; return float((ts-z.iloc[-1].timestamp).total_seconds()/60) if len(z) else np.nan
    out={k:np.nan for k in FEATURE_NAMES}; out["glucose_current"]=prior(g,"glucose_mg_dl")
    for lag in (15,30,60,120): out[f"glucose_lag_{lag}"]=prior(g,"glucose_mg_dl",ts-pd.Timedelta(minutes=lag))
    for lag in (15,30,60): out[f"glucose_delta_{lag}"]=out["glucose_current"]-out[f"glucose_lag_{lag}"]
    for w in (30,60,120):
        z=win(g,"glucose_mg_dl",w); out[f"glucose_mean_{w}"]=float(z.mean()) if len(z) else np.nan; out[f"glucose_std_{w}"]=float(z.std()) if len(z)>1 else np.nan
    z=win(g,"glucose_mg_dl",60); out["glucose_min_60"]=float(z.min()) if len(z) else np.nan; out["glucose_max_60"]=float(z.max()) if len(z) else np.nan
    for w in (30,60,120,240): out[f"bolus_{w}"]=total(i,"bolus_units",w); out[f"carbs_{w}"]=total(m,"carbs_g",w)
    out["time_since_last_bolus"]=since(i,"bolus_units"); out["time_since_last_carb_event"]=since(m,"carbs_g"); out["last_carb_amount"]=prior(m,"carbs_g")
    for w in (30,60,120): out[f"basal_sum_{w}"]=total(i,"basal_rate",w)
    out["recent_insulin_exposure"]=(out["bolus_240"] if pd.notna(out["bolus_240"]) else 0)+(out["basal_sum_120"] if pd.notna(out["basal_sum_120"]) else 0)
    for w in (15,30,60,120): out[f"steps_{w}"]=total(a,"steps",w)
    out["heart_rate_current"]=prior(a,"heart_rate")
    for w in (15,30,60):
        z=win(a,"heart_rate",w); out[f"heart_rate_mean_{w}"]=float(z.mean()) if z.notna().any() else np.nan
    for w in (30,60):
        z=win(a,"heart_rate",w); out[f"heart_rate_max_{w}"]=float(z.max()) if z.notna().any() else np.nan
    for w in (30,60,120): out[f"calories_{w}"]=total(a,"calories",w)
    out.update(hour_sin=np.sin(2*np.pi*ts.hour/24),hour_cos=np.cos(2*np.pi*ts.hour/24),day_sin=np.sin(2*np.pi*ts.dayofweek/7),day_cos=np.cos(2*np.pi*ts.dayofweek/7),recent_glucose_count_60=float(len(win(g,"glucose_mg_dl",60))))
    out["minutes_since_latest_glucose"]=float((ts-g.iloc[-1].timestamp).total_seconds()/60) if len(g) else np.nan; out["missing_hr_fraction_60"]=float(win(a,"heart_rate",60).isna().mean()) if len(win(a,"heart_rate",60)) else 1.; out["missing_activity_fraction_60"]=float(win(a,"steps",60).isna().mean()) if len(win(a,"steps",60)) else 1.
    action=_dict(proposed_action); typ=str(action.get("action_type","none")).lower().replace("actiontype.",""); key=f"action_type_{typ}"; out[key]=1. if key in out else np.nan
    ex=_dict(action.get("exercise")); cat=str(ex.get("category",action.get("category",""))).lower(); key=f"exercise_{cat}"; out[key]=1. if key in out else np.nan; out["exercise_duration_minutes"]=ex.get("duration_minutes",action.get("duration_minutes",np.nan)); intensity=str(ex.get("intensity",action.get("intensity",""))).lower(); out["exercise_intensity_numeric"]={"low":1,"moderate":2,"high":3}.get(intensity,np.nan); out["exercise_start_delay_minutes"]=ex.get("start_delay_minutes",action.get("start_delay_minutes",np.nan))
    out["caffeine_mg"]=action.get("caffeine_mg",action.get("dose_mg",np.nan)); out["alcohol_grams"]=action.get("alcohol_grams",np.nan); out["alcohol_drink_count"]=action.get("alcohol_drink_count",action.get("drink_count",np.nan)); ctx=_dict(action.get("context")); out["stress_level"]=ctx.get("stress_level",action.get("stress_level",np.nan)); out["illness"]=float(bool(ctx.get("illness",action.get("illness",False)))); hyd=str(ctx.get("hydration",action.get("hydration",""))).lower(); key=f"hydration_{hyd}"; out[key]=1. if key in out else np.nan; out["sleep_hours"]=ctx.get("sleep_hours",action.get("sleep_hours",np.nan))
    return pd.Series(out,index=FEATURE_NAMES,dtype="float64")
