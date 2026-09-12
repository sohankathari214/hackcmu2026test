from __future__ import annotations

from datetime import datetime

import pandas as pd


def validate_event_timestamps(frame: pd.DataFrame, snapshot_time: datetime) -> bool:
    """Validate that no event timestamp is in the future relative to the snapshot."""
    if frame.empty:
        return True

    ts_col = None
    for candidate in ["timestamp", "start", "end"]:
        if candidate in frame.columns:
            ts_col = candidate
            break

    if ts_col is None:
        return True

    return bool((pd.to_datetime(frame[ts_col], errors="coerce") <= pd.Timestamp(snapshot_time)).all())
