from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    output_dir = Path("artifacts/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": ["patient_001"], "mae": [0.0]}).to_csv(output_dir / "personalization_curve.csv", index=False)
    print("Personalization experiment placeholder generated.")
