from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    patient_id: str
    start_timestamp: datetime
    pre_action_state: dict[str, Any]
    planned_action: dict[str, Any]
    actual_action: Optional[dict[str, Any]] = None
    prediction: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    residuals: dict[str, Any] = Field(default_factory=dict)
    unexpected_events: list[dict[str, Any]] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)
