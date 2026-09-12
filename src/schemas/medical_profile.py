from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiabetesType(str, Enum):
    T1D = "T1D"


class InsulinDelivery(str, Enum):
    PUMP = "pump"
    INJECTION = "injection"
    OTHER = "other"
    UNKNOWN = "unknown"


class SexAtBirth(str, Enum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    UNKNOWN = "unknown"


class Medication(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    category: str
    dose: Optional[float] = None
    unit: Optional[str] = None


class MedicalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    profile_version: int
    demographics: dict[str, Any] = Field(default_factory=dict)
    diabetes: dict[str, Any] = Field(default_factory=dict)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("demographics", mode="before")
    @classmethod
    def normalize_demographics(cls, value):
        if value is None:
            return {}
        return value

    @field_validator("diabetes", mode="before")
    @classmethod
    def normalize_diabetes(cls, value):
        if value is None:
            return {}
        return value

    @field_validator("baseline_metrics", mode="before")
    @classmethod
    def normalize_baseline_metrics(cls, value):
        if value is None:
            return {}
        return value

    @field_validator("source_metadata", mode="before")
    @classmethod
    def normalize_source_metadata(cls, value):
        if value is None:
            return {}
        return value

    @field_validator("patient_id")
    @classmethod
    def patient_id_non_empty(cls, value):
        if not value or not value.strip():
            raise ValueError("patient_id must not be empty")
        return value

    @field_validator("profile_version")
    @classmethod
    def validate_profile_version(cls, value):
        if value < 1:
            raise ValueError("profile_version must be >= 1")
        return value

    @field_validator("medications")
    @classmethod
    def validate_medications(cls, value):
        if value is None:
            return []
        return value

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)
