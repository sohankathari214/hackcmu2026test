from __future__ import annotations

import pandas as pd


def test_targets_shape():
    df = pd.DataFrame({
        "glucose_current": [120, 121],
        "y30": [125, 126],
        "y60": [130, 131],
        "y90": [135, 136],
        "y120": [140, 141],
        "hypo_120": [0, 1],
    })
    assert "y120" in df.columns
    assert "hypo_120" in df.columns
