from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceBaseline:
    def predict(self, X: pd.DataFrame, horizon: int = 30):
        if X.empty:
            return np.nan
        return float(X.iloc[0]["glucose_current"])


class LinearTrendBaseline:
    def predict(self, X: pd.DataFrame, horizon: int = 30):
        if X.empty:
            return np.nan
        slope = 0.0
        recent = X[["glucose_lag_15", "glucose_lag_30", "glucose_lag_60", "glucose_lag_120"]].dropna()
        if recent.shape[0] >= 2:
            slope = float((recent.iloc[-1].mean() - recent.iloc[0].mean()) / max(1, len(recent) - 1))
        current = float(X["glucose_current"].iloc[0])
        return current + slope * (horizon / 30)


class RidgeBaseline:
    def __init__(self, model=None):
        self.model = model

    def predict(self, X: pd.DataFrame, horizon: int = 30):
        if self.model is None:
            return float(X["glucose_current"])
        return float(self.model.predict(X.values.reshape(1, -1))[0])
