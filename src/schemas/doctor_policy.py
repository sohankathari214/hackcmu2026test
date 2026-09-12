from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DoctorPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    policy_version: int
    risk_priorities: dict[str, int] = Field(default_factory=dict)
    clinical_facts: list[dict[str, Any]] = Field(default_factory=list)
    hard_rules: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value):
        if not value or not value.strip():
            raise ValueError("patient_id must not be empty")
        return value

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value):
        if value < 1:
            raise ValueError("policy_version must be >= 1")
        return value

    @field_validator("risk_priorities")
    @classmethod
    def validate_risk_priorities(cls, value):
        for key, pr in value.items():
            if pr < 0 or pr > 3:
                raise ValueError(f"risk priority for {key} must be between 0 and 3")
        return value

    @field_validator("provenance", mode="before")
    @classmethod
    def normalize_provenance(cls, value):
        if value is None:
            return {}
        return value
