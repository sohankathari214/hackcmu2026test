from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.population import train_population_models


def test_population_models_train():
    training_df = pd.DataFrame({
        "glucose_current": [120, 121, 122, 123],
        "y30": [125, 126, 127, 128],
        "y60": [130, 131, 132, 133],
        "y90": [135, 136, 137, 138],
        "y120": [140, 141, 142, 143],
    })
    feature_names = ["glucose_current"]
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    models, metrics = train_population_models(training_df, cfg, feature_names)
    assert "population_g30" in models
    assert "g30" in metrics
