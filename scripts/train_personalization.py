from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.personalization import alpha_for_episode_count, train_personal_model


if __name__ == "__main__":
    synthetic = pd.DataFrame({
        "glucose_current": [120, 121, 122, 123],
        "y30": [125, 126, 127, 128],
        "y60": [130, 131, 132, 133],
        "y90": [135, 136, 137, 138],
        "y120": [140, 141, 142, 143],
    })
    feature_names = ["glucose_current"]
    model = train_personal_model(synthetic, feature_names, "y30")
    alpha = alpha_for_episode_count(12)
    print({"alpha": alpha, "model_type": type(model).__name__})
