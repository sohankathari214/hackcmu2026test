from __future__ import annotations
import sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.io import read_table
from src.features.build_state import FEATURE_NAMES
from src.models.population import train_and_save_population
if __name__=="__main__":
 cfg=yaml.safe_load((ROOT/"configs/base.yaml").read_text()); path=ROOT/"data/processed/hybrid_training_episodes.parquet"
 if not path.exists(): raise SystemExit("Hybrid data missing. Run: python3 scripts/train_full_poc.py")
 df=read_table(path); _,metrics=train_and_save_population(df,cfg,FEATURE_NAMES); print(metrics)
