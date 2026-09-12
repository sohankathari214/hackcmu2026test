from __future__ import annotations
import json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.io import read_table,write_table
from src.synthetic.episode_generator import generate_hybrid_episodes
if __name__=="__main__":
 cfg=yaml.safe_load((ROOT/"configs/base.yaml").read_text()); real=read_table(ROOT/"data/processed/hupa_training_real.parquet"); s=cfg.get("synthetic",{}); episodes,traits=generate_hybrid_episodes(real,cfg.get("seed",42),s.get("sample_fraction",.1),s.get("simulator_version","poc-v1")); write_table(episodes,ROOT/"data/processed/hybrid_training_episodes.parquet"); (ROOT/"data/processed/synthetic_patient_traits.json").write_text(json.dumps({"warning":"SIMULATOR-ONLY HIDDEN VARIABLES; never model features", "traits":traits},indent=2)); print(f"Generated {len(episodes):,} hybrid episodes.")
