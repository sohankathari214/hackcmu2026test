from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrendUserReported(str, Enum):
    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    NONE = "none"
    EXERCISE = "exercise"
    MEAL = "meal"
    CAFFEINE = "caffeine"
    ALCOHOL = "alcohol"
    OTHER = "other"


class ExerciseCategory(str, Enum):
    CARDIO = "cardio"
    RESISTANCE = "resistance"
    MIXED = "mixed"
    OTHER = "other"


class ExerciseIntensity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class TimestampAwareModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    @field_validator("timestamp", mode="before", check_fields=False)
    @classmethod
    def ensure_tz(cls, value):
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        return value


class GlucoseReading(TimestampAwareModel):
    timestamp: datetime | None = None
    minutes_ago: Optional[int] = None
    mg_dl: float


class InsulinDose(TimestampAwareModel):
    timestamp: datetime | None = None
    minutes_ago: Optional[int] = None
    units: float
    type: Optional[str] = None
    basal_rate_units_hr: Optional[float] = None
    reported_iob_units: Optional[float] = None


class MealEvent(TimestampAwareModel):
    timestamp: datetime | None = None
    minutes_ago: Optional[int] = None
    carbs_g: float
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None


class ActivityEvent(TimestampAwareModel):
    timestamp: datetime | None = None
    minutes_ago: Optional[int] = None
    recent_activity_minutes: Optional[int] = None
    recent_steps: Optional[int] = None
    current_hr: Optional[int] = None
    recent_hr_samples: Optional[list[int]] = None


class SleepContext(TimestampAwareModel):
    last_sleep_hours: Optional[float] = None
    quality: Optional[str] = None
    minutes_since_waking: Optional[int] = None


class ContextEvent(TimestampAwareModel):
    stress_level: Optional[int] = None
    illness: Optional[bool] = None
    hydration: Optional[str] = None
    caffeine_events: list[dict[str, Any]] = Field(default_factory=list)
    alcohol_events: list[dict[str, Any]] = Field(default_factory=list)


class ExerciseAction(TimestampAwareModel):
    category: ExerciseCategory | str | None = None
    subtype: Optional[str] = None
    duration_minutes: Optional[int] = None
    intensity: ExerciseIntensity | str | None = None
    start_delay_minutes: Optional[int] = None


class ProposedAction(TimestampAwareModel):
    action_type: ActionType
    exercise: Optional[ExerciseAction] = None
    category: Optional[str] = None
    subtype: Optional[str] = None
    duration_minutes: Optional[int] = None
    intensity: Optional[str] = None
    start_delay_minutes: Optional[int] = None


class CurrentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    glucose: dict[str, Any]
    recent_readings: list[GlucoseReading] = Field(default_factory=list)
    insulin: dict[str, Any] = Field(default_factory=dict)
    food: dict[str, Any] = Field(default_factory=dict)
    activity: dict[str, Any] = Field(default_factory=dict)
    sleep: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    proposed_action: ProposedAction = Field(default_factory=lambda: ProposedAction(action_type=ActionType.NONE))
    parser_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value):
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        return value

    @field_validator("glucose", mode="before")
    @classmethod
    def validate_glucose(cls, value):
        if value is None:
            raise ValueError("glucose section is required")
        return value

    @field_validator("proposed_action", mode="before")
    @classmethod
    def normalize_action(cls, value):
        if value is None:
            return {"action_type": ActionType.NONE}
        if isinstance(value, dict):
            if "action_type" not in value:
                value = {"action_type": ActionType.NONE, **value}
            return value
        return value
