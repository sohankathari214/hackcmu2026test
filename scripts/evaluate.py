"""Produce separate real-physiology and synthetic-action evaluation reports."""
from __future__ import annotations
import json,sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.io import read_table
from src.features.build_state import FEATURE_NAMES
def metric(name,h,y,p): return {"model":name,"horizon":h,"MAE":float(mean_absolute_error(y,p)),"RMSE":float(mean_squared_error(y,p)**.5),"rows":len(y)}
def chronological(df): return df.groupby("patient_id",group_keys=False).apply(lambda x:x.sort_values("timestamp").iloc[int(.8*len(x)):],include_groups=False)
if __name__=="__main__":
    report=ROOT/"artifacts/reports";report.mkdir(parents=True,exist_ok=True); real=read_table(ROOT/"data/processed/hupa_training_real.parquet"); train=real.groupby("patient_id",group_keys=False).apply(lambda x:x.sort_values("timestamp").iloc[:int(.8*len(x))],include_groups=False); test=chronological(real); records=[]
    for h in (30,60,90,120):
        target=f"y{h}"; tr=train.dropna(subset=[target]); te=test.dropna(subset=[target]); Xtr=tr[FEATURE_NAMES].fillna(0); Xte=te[FEATURE_NAMES].fillna(0)
        records.append(metric("persistence",h,te[target],te.glucose_current)); records.append(metric("linear_trend",h,te[target],te.glucose_current+te.get("glucose_delta_30",0).fillna(0)*(h/30)))
        ridge=Ridge(alpha=5).fit(Xtr,tr[target]);records.append(metric("ridge",h,te[target],ridge.predict(Xte)))
        model=joblib.load(ROOT/f"artifacts/population/models/population_g{h}.joblib");records.append(metric("lightgbm",h,te[target],model.predict(Xte)))
    out=pd.DataFrame(records);out.to_csv(report/"real_forecast_metrics.csv",index=False);(report/"real_forecast_metrics.json").write_text(json.dumps(records,indent=2)); print(out.to_string(index=False))
