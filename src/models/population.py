from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.registry import ModelRegistry


def _safe_float(value):
    if pd.isna(value):
        return np.nan
    return float(value)


def train_population_models(training_df: pd.DataFrame, cfg: dict, feature_names: list[str]) -> tuple[dict[str, lgb.LGBMRegressor], dict[str, dict]]:
    seed = cfg.get("seed", 42)
    horizons = cfg.get("prediction_horizons", [30, 60, 90, 120])
    models = {}
    metrics = {}

    for horizon in horizons:
        target_col = f"y{horizon}"
        subset = training_df.dropna(subset=[target_col])
        X = subset[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = subset[target_col]

        params = {
            "objective": "regression",
            "n_estimators": cfg.get("lightgbm", {}).get("n_estimators", 150),
            "learning_rate": cfg.get("lightgbm", {}).get("learning_rate", 0.05),
            "num_leaves": cfg.get("lightgbm", {}).get("num_leaves", 31),
            "max_depth": cfg.get("lightgbm", {}).get("max_depth", -1),
            "subsample": cfg.get("lightgbm", {}).get("subsample", 0.9),
            "colsample_bytree": cfg.get("lightgbm", {}).get("colsample_bytree", 0.9),
            "random_state": seed,
            "verbose": -1,
            "n_jobs": 1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        models[f"population_g{horizon}"] = model
        pred = model.predict(X)
        metrics[f"g{horizon}"] = {
            "MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "rows": int(len(subset)),
        }

    return models, metrics


def load_population_models(model_dir: str | Path) -> dict[str, object]:
    model_dir = Path(model_dir)
    models = {}
    for file in model_dir.glob("*.joblib"):
        models[file.stem] = file
    return models


def save_population_models(models: dict[str, object], model_dir: str | Path, metadata: dict):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(root_dir=str(model_dir.parent.parent))
    registry.save_population(models, metadata)


def train_and_save_population(training_df: pd.DataFrame, cfg: dict, feature_names: list[str], registry_root: str | Path = "artifacts"):
    models, metrics = train_population_models(training_df, cfg, feature_names)
    registry = ModelRegistry(root_dir=str(registry_root))
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "config": cfg,
        "metrics": metrics,
        "model_type": "population_lightgbm",
    }
    registry.save_population(models, metadata)
    return models, metrics
