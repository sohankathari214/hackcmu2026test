from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model: dict[str, Any] = Field(default_factory=dict)
    scenario: dict[str, Any] = Field(default_factory=dict)
    population_forecast: dict[str, Any] = Field(default_factory=dict)
    personalized_forecast: dict[str, Any] = Field(default_factory=dict)
    quantiles: dict[str, Any] = Field(default_factory=dict)
    hypoglycemia_probability: Optional[float] = None
    hyperglycemia_probability: Optional[float] = None
    personalization: dict[str, Any] = Field(default_factory=dict)
    similar_events: list[dict[str, Any]] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
