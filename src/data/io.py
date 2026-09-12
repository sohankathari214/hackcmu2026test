"""Portable table IO; a pickle fallback keeps this POC runnable without pyarrow."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try: frame.to_parquet(path, index=False)
    except ImportError: frame.to_pickle(path)

def read_table(path: str | Path) -> pd.DataFrame:
    try: return pd.read_parquet(path)
    except (ImportError, ValueError): return pd.read_pickle(path)
