"""Normalize real HUPA data and build the leakage-safe real forecasting table."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.hupa_adapter import HUPAAdapter
from src.data.io import write_table
from src.features.build_state import FEATURE_NAMES, build_feature_vector

class _Profile:
    def __init__(self,pid): self.patient_id=pid

def _features_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """Vectorized counterpart of the canonical feature function for regular HUPA rows."""
    x=group.set_index("timestamp").copy(); out=pd.DataFrame(index=x.index)
    g=x.glucose_mg_dl
    out["glucose_current"]=g
    for lag in (15,30,60,120): out[f"glucose_lag_{lag}"]=g.shift(lag//5)
    for lag in (15,30,60): out[f"glucose_delta_{lag}"]=g-out[f"glucose_lag_{lag}"]
    for w in (30,60,120): out[f"glucose_mean_{w}"]=g.rolling(f"{w}min",closed="both").mean(); out[f"glucose_std_{w}"]=g.rolling(f"{w}min",closed="both").std()
    out["glucose_min_60"]=g.rolling("60min",closed="both").min(); out["glucose_max_60"]=g.rolling("60min",closed="both").max()
    for raw,prefix in [(x.bolus_units,"bolus"),(x.carbs_g,"carbs")]:
        for w in (30,60,120,240): out[f"{prefix}_{w}"]=raw.rolling(f"{w}min",closed="both").sum()
    positive=x.bolus_units.fillna(0)>0; out["time_since_last_bolus"]=(x.index.to_series()-x.index.to_series().where(positive).ffill()).dt.total_seconds()/60
    positive=x.carbs_g.fillna(0)>0; out["time_since_last_carb_event"]=(x.index.to_series()-x.index.to_series().where(positive).ffill()).dt.total_seconds()/60; out["last_carb_amount"]=x.carbs_g.where(positive).ffill()
    for w in (30,60,120): out[f"basal_sum_{w}"]=x.basal_value.rolling(f"{w}min",closed="both").sum()
    out["recent_insulin_exposure"]=out.bolus_240.fillna(0)+out.basal_sum_120.fillna(0)
    for w in (15,30,60,120): out[f"steps_{w}"]=x.steps.rolling(f"{w}min",closed="both").sum()
    out["heart_rate_current"]=x.heart_rate_bpm
    for w in (15,30,60): out[f"heart_rate_mean_{w}"]=x.heart_rate_bpm.rolling(f"{w}min",closed="both").mean()
    for w in (30,60): out[f"heart_rate_max_{w}"]=x.heart_rate_bpm.rolling(f"{w}min",closed="both").max()
    for w in (30,60,120): out[f"calories_{w}"]=x.calories.rolling(f"{w}min",closed="both").sum()
    out["hour_sin"]=np.sin(2*np.pi*x.index.hour/24);out["hour_cos"]=np.cos(2*np.pi*x.index.hour/24);out["day_sin"]=np.sin(2*np.pi*x.index.dayofweek/7);out["day_cos"]=np.cos(2*np.pi*x.index.dayofweek/7)
    out["minutes_since_latest_glucose"]=0.;out["recent_glucose_count_60"]=g.rolling("60min",closed="both").count();out["missing_hr_fraction_60"]=x.heart_rate_bpm.isna().rolling("60min",closed="both").mean();out["missing_activity_fraction_60"]=x.steps.isna().rolling("60min",closed="both").mean()
    for name in FEATURE_NAMES:
        if name not in out: out[name]=0. if name.startswith(("action_type_","exercise_","hydration_")) else np.nan
    out["action_type_none"]=1.
    return out[FEATURE_NAMES].reset_index(drop=True)

def _targets(frame: pd.DataFrame, tolerance: int, hypo: float, hyper: float) -> pd.DataFrame:
    base=frame[["timestamp"]].copy(); source=frame[["timestamp","glucose_mg_dl"]].sort_values("timestamp"); result={}
    for h in (30,60,90,120):
        desired=base.assign(desired=lambda x:x.timestamp+pd.Timedelta(minutes=h)).sort_values("desired")
        joined=pd.merge_asof(desired,source,left_on="desired",right_on="timestamp",direction="nearest",tolerance=pd.Timedelta(minutes=tolerance),suffixes=("_base","_future")).sort_index()
        result[f"y{h}"]=joined.glucose_mg_dl.to_numpy()
    # reverse rolling makes the next 120 minutes available without inspecting future in features.
    rev=frame.set_index("timestamp").glucose_mg_dl.iloc[::-1]
    future_min=rev.rolling("120min",closed="left").min().iloc[::-1]; future_max=rev.rolling("120min",closed="left").max().iloc[::-1]
    result["hypo_120"]=(future_min<hypo).fillna(False).astype(int).to_numpy(); result["hyper_120"]=(future_max>hyper).fillna(False).astype(int).to_numpy()
    return pd.DataFrame(result,index=frame.index)

def prepare(cfg: dict | None=None) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    cfg=cfg or yaml.safe_load((ROOT/"configs/base.yaml").read_text()); hcfg=cfg.get("hupa",{})
    raw=HUPAAdapter(ROOT/"data/raw/Preprocessed",hcfg).load_preprocessed(); write_table(raw,ROOT/"data/processed/hupa_real.parquet")
    rows=[]; intervals={}
    for pid, group in raw.groupby("patient_id",sort=True):
        group=group.sort_values("timestamp").reset_index(drop=True); intervals[pid]=group.timestamp.diff().dt.total_seconds().div(60).dropna().round(3).value_counts().to_dict()
        table=_features_for_group(group); table.insert(0,"timestamp",group.timestamp); table.insert(0,"patient_id",pid); rows.append(pd.concat([table,_targets(group,int(cfg.get("target_tolerance_minutes",15)),float(cfg.get("hypoglycemia_threshold",70)),float(cfg.get("hyperglycemia_threshold",180)))],axis=1))
    training=pd.concat(rows,ignore_index=True); training["data_source"]="real"; write_table(training,ROOT/"data/processed/hupa_training_real.parquet")
    summary={"patient_count":int(raw.patient_id.nunique()),"rows":int(len(raw)),"rows_per_patient":raw.groupby("patient_id").size().to_dict(),"date_range":{"min":str(raw.timestamp.min()),"max":str(raw.timestamp.max())},"sampling_interval_minutes":intervals,"missing_values":raw.isna().sum().to_dict(),"numeric_min":raw.select_dtypes("number").min().to_dict(),"numeric_max":raw.select_dtypes("number").max().to_dict()}
    report=ROOT/"artifacts/reports"; report.mkdir(parents=True,exist_ok=True); (report/"hupa_data_summary.json").write_text(json.dumps(summary,indent=2,default=str)); raw.isna().mean().rename("missing_fraction").to_csv(report/"hupa_missingness.csv")
    return raw,training,summary

if __name__=="__main__":
    raw,training,summary=prepare(); print(f"Prepared {len(raw):,} normalized rows for {summary['patient_count']} HUPA patients; {len(training):,} real training rows.")
