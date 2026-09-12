"""Clearly labelled POC-only observed values for demonstrating the episode loop."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def synthetic_observation(episode: dict[str, Any]) -> dict[str, Any]:
    forecast = episode.get("prediction", {})
    seed = int(hashlib.sha256(episode["episode_id"].encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    observed = {f"g{h}": float(np.clip(float(forecast.get(f"g{h}", 145)) + rng.normal(0, 5 + h / 20), 40, 400)) for h in (30, 60, 90, 120)}
    return {"observed": observed, "provenance": {"data_source": "synthetic_poc", "generator": "forecast_residual_noise_v1", "random_seed": seed, "not_for_personalization": True}}
