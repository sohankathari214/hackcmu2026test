"""LLM boundary: language becomes validated data; the LLM never forecasts glucose."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from src.schemas.current_state import CurrentState
from src.schemas.medical_profile import MedicalProfile
from src.schemas.doctor_policy import DoctorPolicy

load_dotenv()


SYSTEM = """Extract only explicitly stated information into CurrentState JSON.
Never invent values. Unknown values must be omitted or null. Do not give medical
advice, insulin recommendations, safety decisions, or forecasts. Set
proposed_action only when the user clearly states one."""


def _schema(model: type) -> dict[str, Any]:
    """Return a JSON schema compatible with OpenAI Structured Outputs."""
    return model.model_json_schema()


def _openai_extract(text: str, schema: type, instructions: str) -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI

        response = OpenAI(api_key=key).responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
            instructions=instructions,
            input=text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__.lower(),
                    "schema": _schema(schema),
                    "strict": False,
                }
            },
        )
        if not response.output_text:
            raise RuntimeError("OpenAI returned no structured output.")
        return schema.model_validate_json(response.output_text).model_dump(mode="json")
    except Exception as exc:
        raise RuntimeError(f"OpenAI structured extraction failed; no fallback was used: {exc}") from exc


def _demo_parse(text: str) -> dict[str, Any]:
    lower = text.lower()
    glucose = re.search(r"(?:at|glucose(?: is)?)\s*(\d{2,3})", lower)
    minutes = re.search(r"(\d+)\s*(?:minute|min)", lower)
    duration = int(minutes.group(1)) if minutes else None
    action = "exercise" if any(x in lower for x in ("run", "workout", "exercise", "gym")) else "caffeine" if any(x in lower for x in ("coffee", "caffeine")) else "alcohol" if any(x in lower for x in ("beer", "wine", "drink")) else "none"
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "glucose": {"current_mg_dl": int(glucose.group(1)) if glucose else None}, "insulin": {}, "food": {}, "activity": {}, "sleep": {}, "context": {}, "proposed_action": {"action_type": action}}
    if action == "exercise":
        result["proposed_action"]["exercise"] = {"category": "cardio", "duration_minutes": duration, "intensity": "moderate", "start_delay_minutes": 0}
    if action == "caffeine":
        result["proposed_action"]["dose_mg"] = None
    return result


def _provenance(used: bool, warning: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"provider": "openai" if used else "deterministic_demo", "openai_used": used}
    if warning:
        result["warning"] = warning
    return result


def parse_state_text(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("text is required")
    if not os.getenv("OPENAI_API_KEY"):
        parsed = _demo_parse(text)
        return {"state": CurrentState.model_validate(parsed).model_dump(mode="json"), "parser": _provenance(False, "Set OPENAI_API_KEY to enable OpenAI structured extraction.")}
    return {"state": _openai_extract(text, CurrentState, SYSTEM), "parser": _provenance(True)}


def _extract(text: str, schema: type, demo: dict[str, Any], kind: str) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {"data": schema.model_validate(demo).model_dump(mode="json"), "parser": _provenance(False, "Set OPENAI_API_KEY to enable OpenAI structured extraction.")}
    instructions = f"Extract only explicitly stated {kind} facts into the supplied JSON schema. Do not invent missing information. This is data extraction, not medical advice."
    return {"data": _openai_extract(text, schema, instructions), "parser": _provenance(True)}


def parse_profile_text(text: str, patient_id: str = "patient_001"):
    return _extract(text, MedicalProfile, {"patient_id": patient_id, "profile_version": 1, "demographics": {}, "diabetes": {"type": "T1D"}, "medications": [], "conditions": [], "baseline_metrics": {}, "source_metadata": {"demo": True}}, "medical profile")


def parse_policy_text(text: str, patient_id: str = "patient_001"):
    return _extract(text, DoctorPolicy, {"patient_id": patient_id, "policy_version": 1, "risk_priorities": {}, "clinical_facts": [], "hard_rules": [], "provenance": {"demo": True}}, "clinician policy")


def explain_structured_forecast(forecast: dict[str, Any]) -> dict[str, Any]:
    # A deterministic explanation prevents an LLM from inventing causal or clinical claims.
    p = forecast.get("proposed", forecast)
    warning = (p.get("warnings") or ["This is a research prototype."])[0] if isinstance(p, dict) else "This is a research prototype."
    return {"summary": "The model compared the current-state baseline with the proposed structured action.", "factors": ["recent glucose trend", "recorded insulin and carbohydrate context", "proposed action details", "available personal episode history"], "warning": warning, "generated_by": "deterministic_template"}
