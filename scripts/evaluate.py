from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import LinearTrendBaseline, PersistenceBaseline


if __name__ == "__main__":
    df = pd.DataFrame({
        "glucose_current": [120, 121, 122],
        "y30": [125, 126, 127],
        "y60": [130, 131, 132],
        "y90": [135, 136, 137],
        "y120": [140, 141, 142],
    })
    persistence = PersistenceBaseline()
    trend = LinearTrendBaseline()
    results = {
        "persistence": persistence.predict(df),
        "linear_trend": trend.predict(df),
    }
    (Path("artifacts/reports")).mkdir(parents=True, exist_ok=True)
    with open("artifacts/reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [results["persistence"], results["linear_trend"]])
    ax.set_title("Synthetic evaluation")
    fig.savefig("artifacts/reports/evaluation.png")
    print(json.dumps(results, indent=2))
