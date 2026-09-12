from __future__ import annotations

import pandas as pd

from src.models.baselines import PersistenceBaseline, LinearTrendBaseline


def test_persistence_returns_current_glucose():
    df = pd.DataFrame({"glucose_current": [120.0]})
    pred = PersistenceBaseline().predict(df)
    assert pred == 120.0


def test_linear_trend_works():
    df = pd.DataFrame({"glucose_current": [120.0], "glucose_lag_15": [110.0], "glucose_lag_30": [100.0], "glucose_lag_60": [95.0], "glucose_lag_120": [90.0]})
    pred = LinearTrendBaseline().predict(df, horizon=30)
    assert pred >= 100.0
