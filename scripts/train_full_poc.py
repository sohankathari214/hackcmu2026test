"""Run the real-HUPA plus synthetic-action proof-of-concept pipeline."""
from __future__ import annotations
import json,sys,subprocess
from pathlib import Path
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_hupa import prepare
from src.data.io import write_table
from src.features.build_state import FEATURE_NAMES
from src.synthetic.episode_generator import generate_hybrid_episodes
from src.models.population import train_and_save_population
from src.models.quantile import train_quantile_models
from src.models.risk import train_risk_models
def main():
 cfg=yaml.safe_load((ROOT/"configs/base.yaml").read_text()); raw,real,summary=prepare(cfg); report=ROOT/"artifacts/reports"; report.mkdir(parents=True,exist_ok=True)
 test=real.groupby("patient_id",group_keys=False).apply(lambda x:x.sort_values("timestamp").iloc[int(len(x)*.8):],include_groups=False); metrics=[]
 for h in (30,60,90,120):
  z=test.dropna(subset=[f"y{h}","glucose_current"]); metrics.append({"model":"persistence","horizon":h,"MAE":float(mean_absolute_error(z[f"y{h}"],z.glucose_current)),"rows":len(z)})
 pd.DataFrame(metrics).to_csv(report/"real_forecast_metrics.csv",index=False); (report/"real_forecast_metrics.json").write_text(json.dumps(metrics,indent=2))
 s=cfg.get("synthetic",{}); hybrid,traits=generate_hybrid_episodes(real,cfg.get("seed",42),s.get("sample_fraction",.1),s.get("simulator_version","poc-v1")); write_table(hybrid,ROOT/"data/processed/hybrid_training_episodes.parquet"); (ROOT/"data/processed/synthetic_patient_traits.json").write_text(json.dumps({"warning":"SIMULATOR-ONLY HIDDEN VARIABLES; never model features","traits":traits},indent=2))
 groups=hybrid.base_state_id.drop_duplicates().tolist(); cut=int(len(groups)*.8); train=hybrid[hybrid.base_state_id.isin(groups[:cut])]; held=hybrid[hybrid.base_state_id.isin(groups[cut:])]
 models,pop_metrics=train_and_save_population(train,cfg,FEATURE_NAMES); qmodels=train_quantile_models(train,cfg,FEATURE_NAMES); risks,risk_metrics=train_risk_models(train,cfg,FEATURE_NAMES)
 import joblib
 for h,qs in qmodels.items():
  for q,m in qs.items(): joblib.dump(m,ROOT/f"artifacts/population/models/{h}_{q}.joblib")
 for n,m in risks.items(): joblib.dump(m,ROOT/f"artifacts/population/models/{n}.joblib")
 synthetic=[]
 for h in (30,60,90,120):
  z=held.dropna(subset=[f"y{h}"]); pred=models[f"population_g{h}"].predict(z[FEATURE_NAMES].fillna(0)); synthetic.append({"label":"synthetic proof-of-concept evaluation","horizon":h,"MAE":float(mean_absolute_error(z[f"y{h}"],pred)),"rows":len(z)})
 pd.DataFrame(synthetic).to_csv(report/"synthetic_action_metrics.csv",index=False); (report/"synthetic_action_metrics.json").write_text(json.dumps(synthetic,indent=2)); (report/"model_summary.json").write_text(json.dumps({"real_data":"HUPA-UCM","synthetic_additions":["exercise labels/effects","caffeine","alcohol","stress","illness","hydration"],"clinical_validation":False,"proof_of_concept":True,"population_metrics":pop_metrics,"risk_metrics":risk_metrics},indent=2)); subprocess.run([sys.executable,str(ROOT/"scripts/train_personalization.py")],check=True); subprocess.run([sys.executable,str(ROOT/"scripts/run_personalization_experiment.py")],check=True); print(f"Complete: {len(raw)} real rows, {len(real)} real states, {len(hybrid)} hybrid episodes")
if __name__=="__main__": main()
