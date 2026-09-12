from __future__ import annotations

from src.episodes.create import create_episode
from src.episodes.finalize import finalize_episode


def test_episode_finalization():
    episode = create_episode("patient_001", {"timestamp": "2026-01-01T00:00:00Z"}, {"action_type": "exercise"}, {"population_model_version": "demo"})
    finalized = finalize_episode(episode, {"action_type": "exercise"}, {"g30": 120, "g60": 121, "g90": 122, "g120": 123}, {"g30": 110, "g60": 111, "g90": 112, "g120": 113})
    assert finalized["actual_action"]["action_type"] == "exercise"
    assert finalized["residuals"]["r30"] == 10
