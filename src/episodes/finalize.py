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
    unexpected = episode.get("unexpected_events", [])
    episode["quality"]["confounded"] = bool(unexpected)
    synthetic = bool(episode.get("observation_provenance", {}).get("not_for_personalization"))
    episode["quality"]["usable_for_personalization"] = bool(csv) and not bool(unexpected) and not synthetic
    if synthetic:
        episode["quality"].setdefault("exclusion_reasons", []).append("synthetic_poc_observation")
    return episode
