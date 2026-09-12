from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base_adapter import BaseDatasetAdapter


if __name__ == "__main__":
    adapter = BaseDatasetAdapter(root_dir=Path("data/raw"))
    normalized = adapter.normalize()
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in normalized.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    print("Prepared synthetic normalized data tables in data/processed")
