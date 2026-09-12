from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


class PersonalResidualModel:
    def __init__(self, horizon: int, alpha: float = 1.0):
        self.horizon = horizon
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if not self.feature_names:
            return pd.Series(np.zeros(len(X)), index=X.index)
        return pd.Series(self.model.predict(X[self.feature_names]), index=X.index)


def alpha_for_episode_count(n: int, k: int = 20) -> float:
    if n < 10:
        return 0.0
    return min(1.0, n / (n + k))


def train_personal_model(training_df: pd.DataFrame, feature_names: list[str], target_name: str, alpha: float = 1.0):
    subset = training_df.dropna(subset=[target_name])
    X = subset[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = subset[target_name]
    model = PersonalResidualModel(horizon=int(target_name.split("_")[-1][1:]) if target_name.startswith("residual_g") else 30, alpha=alpha)
    model.fit(X, y)
    return model


def evaluate_personal_model(model, X: pd.DataFrame, y: pd.Series):
    pred = model.predict(X)
    return float(mean_absolute_error(y, pred))
