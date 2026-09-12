from __future__ import annotations

from typing import Any


def finalize_episode(episode: dict[str, Any], actual_action: dict[str, Any], observed: dict[str, Any], population_forecast: dict[str, Any]):
    episode["actual_action"] = actual_action
    episode["observed"] = observed
    episode["quality"]["complete"] = True
    csv = {}
    for horizon in [30, 60, 90, 120]:
        key = f"g{horizon}"
        obs = observed.get(key)
        pred = population_forecast.get(key)
        if obs is not None and pred is not None:
            csv[f"r{horizon}"] = obs - pred
    episode["residuals"] = csv
    return episode
