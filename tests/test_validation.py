from __future__ import annotations

from datetime import datetime, timezone

from src.data.validation import validate_event_timestamps
import pandas as pd


def test_validate_event_timestamps():
    frame = pd.DataFrame({"timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)]})
    assert validate_event_timestamps(frame, datetime(2026, 1, 1, tzinfo=timezone.utc)) is True
