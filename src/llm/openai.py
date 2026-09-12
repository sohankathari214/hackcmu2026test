"""LLM boundary: language becomes validated data; the LLM never forecasts glucose."""
from __future__ import annotations

import json
import os
import re
import time
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
        # The simulator must remain responsive even if the external provider is
        # slow. A timeout leads to the labelled local extraction path below.
        client = OpenAI(api_key=key, timeout=15.0, max_retries=0)
        last_error: Exception | None = None
        for attempt in range(1):
            try:
                response = client.responses.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                    instructions=instructions,
                    input=text,
                    max_output_tokens=1_800,
                    text={"format": {"type": "json_schema", "name": schema.__name__.lower(), "schema": _schema(schema), "strict": False}},
                )
                if not response.output_text:
                    raise RuntimeError("OpenAI returned no structured output.")
                payload = json.loads(response.output_text)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    raise
                time.sleep(0.35 * (attempt + 1))
        else:  # pragma: no cover - defensive; loop either breaks or raises
            raise last_error or RuntimeError("OpenAI extraction failed.")
        # The state timestamp is the parser's receipt time, not an invented
        # clinical observation. It keeps a partial user report usable by the
        # downstream time-based feature engine.
        if schema is CurrentState:
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            payload.setdefault("recent_readings", [])
            payload.setdefault("insulin", {})
            payload.setdefault("food", {})
            payload.setdefault("activity", {})
            payload.setdefault("sleep", {})
            payload.setdefault("context", {})
            payload.setdefault("proposed_action", {"action_type": "none"})
            payload.setdefault("parser_metadata", {})
        return schema.model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise RuntimeError(f"OpenAI structured extraction failed; no fallback was used: {exc}") from exc


def _demo_parse(text: str) -> dict[str, Any]:
    lower = text.lower()
    glucose = re.search(r"(?:at|glucose(?: is)?)\s*(\d{2,3})", lower)
    minutes = re.search(r"(\d+)\s*(?:minute|min)", lower)
    duration = int(minutes.group(1)) if minutes else None
    action = "exercise" if any(x in lower for x in ("run", "workout", "exercise", "gym")) else "caffeine" if any(x in lower for x in ("coffee", "caffeine")) else "alcohol" if any(x in lower for x in ("beer", "wine", "drink")) else "none"
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "glucose": {"current_mg_dl": int(glucose.group(1)) if glucose else 145}, "insulin": {}, "food": {}, "activity": {}, "sleep": {}, "context": {}, "proposed_action": {"action_type": action}, "parser_metadata": {"estimated_fields": ["glucose.current_mg_dl"] if not glucose else []}}
    if action == "exercise":
        result["proposed_action"]["exercise"] = {"category": "cardio", "duration_minutes": duration, "intensity": "moderate", "start_delay_minutes": 0}
    if action == "caffeine":
        result["proposed_action"]["dose_mg"] = None
    return result


def _enrich_state(state: dict[str, Any], text: str) -> dict[str, Any]:
    """Fill only POC defaults needed by the feature engine, with explicit provenance."""
    lower = text.lower()
    metadata = state.setdefault("parser_metadata", {})
    estimated = list(metadata.get("estimated_fields", []))
    glucose = state.setdefault("glucose", {})
    if glucose.get("current_mg_dl") is None and glucose.get("mg_dl") is not None:
        glucose["current_mg_dl"] = glucose["mg_dl"]
    if glucose.get("current_mg_dl") is None:
        glucose["current_mg_dl"] = 145.0; estimated.append("glucose.current_mg_dl")
    insulin = state.setdefault("insulin", {})
    food = state.setdefault("food", {})
    units = re.search(r"(\d+(?:\.\d+)?)\s*(?:u|units?)\b", lower)
    carbs = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s*)?(?:carb|carbs|carbohydrate)", lower)
    ago = re.search(r"(\d+)\s*(minutes?|mins?|hours?|hrs?)\s*ago", lower)
    minutes_ago = int(ago.group(1)) * (60 if ago and ago.group(2).startswith(("hour", "hr")) else 1) if ago else 0
    if units and not insulin.get("recent_doses"):
        insulin["recent_doses"] = [{"units": float(units.group(1)), "minutes_ago": minutes_ago}]
    if carbs and not food.get("recent_meals"):
        food["recent_meals"] = [{"carbs_g": float(carbs.group(1)), "minutes_ago": minutes_ago}]
    action = state.setdefault("proposed_action", {"action_type": "none"})
    action_type = str(action.get("action_type", "none"))
    if action_type == "exercise":
        exercise = action.setdefault("exercise", {})
        duration = re.search(r"(\d+)\s*(?:minute|min)\s*(?:run|walk|workout|exercise)?", lower)
        if exercise.get("duration_minutes") is None:
            exercise["duration_minutes"] = int(duration.group(1)) if duration else 30; estimated.append("proposed_action.exercise.duration_minutes") if not duration else None
        if not exercise.get("intensity"):
            exercise["intensity"] = "high" if "high intensity" in lower else "low" if "low intensity" in lower else "moderate"; estimated.append("proposed_action.exercise.intensity") if "intensity" not in lower else None
        exercise.setdefault("category", "cardio"); exercise.setdefault("start_delay_minutes", 0)
    elif action_type == "meal" and not food.get("recent_meals"):
        food["recent_meals"] = [{"carbs_g": 45.0, "minutes_ago": 0}]; estimated.append("food.recent_meals[0].carbs_g")
    metadata["estimated_fields"] = sorted(set(estimated))
    metadata["event_types"] = [action_type] + (["insulin"] if insulin.get("recent_doses") else []) + (["meal"] if food.get("recent_meals") else [])
    metadata["poc_imputation"] = bool(metadata["estimated_fields"])
    return state


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
        return {"state": CurrentState.model_validate(_enrich_state(parsed, text)).model_dump(mode="json"), "parser": _provenance(False, "Set OPENAI_API_KEY to enable OpenAI structured extraction.")}
    try:
        parsed = _enrich_state(_openai_extract(text, CurrentState, SYSTEM), text)
        return {"state": CurrentState.model_validate(parsed).model_dump(mode="json"), "parser": _provenance(True)}
    except RuntimeError as exc:
        parsed = _enrich_state(_demo_parse(text), text)
        return {"state": CurrentState.model_validate(parsed).model_dump(mode="json"), "parser": _provenance(False, f"OpenAI was unavailable; used resilient local extraction: {exc}")}


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


def user_feedback(state: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
    """Best-effort LLM context explanation; never used to control the forecast."""
    fallback = {"message": "Review the projected trajectory and the input fields before acting. Record what actually happens afterward to improve future calibration.", "questions": ["Are the glucose, timing, insulin, carbohydrate, and activity details accurate?"], "safety_note": "This is decision support, not medical advice or dosing guidance.", "generated_by": "deterministic_fallback"}
    if not os.getenv("OPENAI_API_KEY"):
        return fallback
    prompt = {"state": state, "forecast": {"baseline": forecast.get("baseline", {}).get("population_forecast", {}), "proposed": forecast.get("proposed", {}).get("population_forecast", {}), "difference": forecast.get("difference", {})}}
    instructions = "Provide a concise, non-prescriptive explanation of the supplied glucose forecast. Do not recommend insulin, food, exercise, or treatment. Return JSON with message (string), questions (array of up to two data-quality questions), and safety_note (string)."
    try:
        from openai import OpenAI
        response = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=15.0, max_retries=0).responses.create(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"), instructions=instructions, input=json.dumps(prompt), max_output_tokens=600, text={"format": {"type": "json_object"}})
        answer = json.loads(response.output_text)
        return {"message": str(answer.get("message", fallback["message"])), "questions": [str(x) for x in answer.get("questions", [])[:2]], "safety_note": str(answer.get("safety_note", fallback["safety_note"])), "generated_by": "openai"}
    except Exception:
        return fallback
