from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.validation import validate_event_timestamps


def test_no_future_event_leakage():
    frame = pd.DataFrame({
        "timestamp": [datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)],
    })
    assert validate_event_timestamps(frame, datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)) is False
