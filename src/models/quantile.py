from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def train_quantile_models(training_df: pd.DataFrame, cfg: dict, feature_names: list[str]):
    horizons = cfg.get("prediction_horizons", [30, 60, 90, 120])
    quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])
    models = {}

    for horizon in horizons:
        target_col = f"y{horizon}"
        subset = training_df.dropna(subset=[target_col] + feature_names)
        X = subset[feature_names]
        y = subset[target_col]
        horizon_models = {}

        for q in quantiles:
            params = {
                "objective": "quantile",
                "alpha": q,
                "n_estimators": cfg.get("lightgbm", {}).get("n_estimators", 150),
                "learning_rate": cfg.get("lightgbm", {}).get("learning_rate", 0.05),
                "num_leaves": cfg.get("lightgbm", {}).get("num_leaves", 31),
                "max_depth": cfg.get("lightgbm", {}).get("max_depth", -1),
                "subsample": cfg.get("lightgbm", {}).get("subsample", 0.9),
                "colsample_bytree": cfg.get("lightgbm", {}).get("colsample_bytree", 0.9),
                "random_state": cfg.get("seed", 42),
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            horizon_models[f"q{int(q*100)}"] = model

        models[f"g{horizon}"] = horizon_models

    return models
