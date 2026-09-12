"""Chronological reveal experiment for general (not action-specific) residual models."""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.io import read_table
from src.features.build_state import FEATURE_NAMES
from src.models.personalization import train_personal_model,alpha_for_episode_count
if __name__=="__main__":
    df=read_table(ROOT/"data/processed/hybrid_training_episodes.parquet"); pops={p.stem:joblib.load(p) for p in (ROOT/"artifacts/population/models").glob("population_g*.joblib")}; output=[]
    for pid,x in df.groupby("patient_id"):
        x=x.sort_values("timestamp").reset_index(drop=True); history=x.iloc[:int(.8*len(x))]; test=x.iloc[int(.8*len(x)):]
        if test.empty: continue
        for n in (0,5,10,20,40):
            errors=[]
            for h in (30,60,90,120):
                base=pops[f"population_g{h}"].predict(test[FEATURE_NAMES].fillna(0)); pred=base.copy()
                if n>=10 and len(history)>=n:
                    fit=history.iloc[:n].copy(); fit["residual"]=fit[f"y{h}"]-pops[f"population_g{h}"].predict(fit[FEATURE_NAMES].fillna(0)); pred += alpha_for_episode_count(n)*train_personal_model(fit,FEATURE_NAMES,"residual").predict(test[FEATURE_NAMES].fillna(0)).to_numpy()
                errors.append(mean_absolute_error(test[f"y{h}"],pred))
            output.append({"patient_id":pid,"episodes_revealed":n,"mae":sum(errors)/len(errors)})
    out=ROOT/"artifacts/reports"; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(output).to_csv(out/"personalization_curve.csv",index=False); print(f"Wrote {out/'personalization_curve.csv'}")
