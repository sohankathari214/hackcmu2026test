from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def train_risk_models(training_df: pd.DataFrame, cfg: dict, feature_names: list[str]):
    models = {}
    metrics = {}

    for target_name in ["hypo_120", "hyper_120"]:
        if target_name not in training_df.columns:
            continue
        subset = training_df.dropna(subset=[target_name] + feature_names)
        X = subset[feature_names]
        y = subset[target_name].astype(int)

        if y.nunique() < 2:
            continue

        params = {
            "objective": "binary",
            "n_estimators": cfg.get("lightgbm", {}).get("n_estimators", 150),
            "learning_rate": cfg.get("lightgbm", {}).get("learning_rate", 0.05),
            "num_leaves": cfg.get("lightgbm", {}).get("num_leaves", 31),
            "max_depth": cfg.get("lightgbm", {}).get("max_depth", -1),
            "subsample": cfg.get("lightgbm", {}).get("subsample", 0.9),
            "colsample_bytree": cfg.get("lightgbm", {}).get("colsample_bytree", 0.9),
            "random_state": cfg.get("seed", 42),
            "verbose": -1,
            "is_unbalance": True,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        models[target_name] = model

        probs = model.predict_proba(X)[:, 1]
        metrics[target_name] = {
            "AUROC": float(roc_auc_score(y, probs)),
            "PR_AUC": float(average_precision_score(y, probs)),
            "Brier": float(brier_score_loss(y, probs)),
        }

    return models, metrics
