from __future__ import annotations

from src.models.personalization import alpha_for_episode_count


def test_alpha_zero_below_threshold():
    assert alpha_for_episode_count(5) == 0.0


def test_alpha_formula_correct():
    assert alpha_for_episode_count(10) > 0
