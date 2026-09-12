"""LLM boundary: language becomes validated data; LLM output never forecasts glucose."""
from __future__ import annotations
import os,re
from datetime import datetime, timezone
from typing import Any
from src.schemas.current_state import CurrentState
from src.schemas.medical_profile import MedicalProfile
from src.schemas.doctor_policy import DoctorPolicy

SYSTEM = """Extract only explicitly stated information into CurrentState JSON. Never invent values. Unknown values must be omitted or null. Do not give medical advice, insulin recommendations, safety decisions, or forecasts. Set proposed_action only when the user clearly states one."""

def _demo_parse(text:str) -> dict[str,Any]:
    lower=text.lower(); glucose=re.search(r"(?:at|glucose(?: is)?)\s*(\d{2,3})",lower); minutes=re.search(r"(\d+)\s*(?:minute|min)",lower); duration=int(minutes.group(1)) if minutes else None
    action="exercise" if any(x in lower for x in ("run","workout","exercise","gym")) else "caffeine" if any(x in lower for x in ("coffee","caffeine")) else "alcohol" if any(x in lower for x in ("beer","wine","drink")) else "none"
    result={"timestamp":datetime.now(timezone.utc).isoformat(),"glucose":{"current_mg_dl":int(glucose.group(1)) if glucose else None},"insulin":{},"food":{},"activity":{},"sleep":{},"context":{},"proposed_action":{"action_type":action}}
    if action=="exercise": result["proposed_action"]["exercise"]={"category":"cardio","duration_minutes":duration,"intensity":"moderate","start_delay_minutes":0}
    if action=="caffeine": result["proposed_action"]["dose_mg"] = None
    return result

def parse_state_text(text:str) -> dict[str,Any]:
    if not text or not text.strip(): raise ValueError("text is required")
    key=os.getenv("GEMINI_API_KEY")
    if not key:
        parsed=_demo_parse(text)
        return {"state":CurrentState.model_validate(parsed).model_dump(mode="json"),"parser":{"provider":"deterministic_demo","gemini_used":False,"warning":"Set GEMINI_API_KEY to enable Gemini structured extraction."}}
    try:
        from google import genai
        from google.genai import types
        client=genai.Client(api_key=key)
        response=client.models.generate_content(model=os.getenv("GEMINI_MODEL","gemini-2.5-flash"),contents=f"{SYSTEM}\n\nUser text:\n{text}",config=types.GenerateContentConfig(response_mime_type="application/json",response_schema=CurrentState))
        state=CurrentState.model_validate_json(response.text)
        return {"state":state.model_dump(mode="json"),"parser":{"provider":"gemini","gemini_used":True}}
    except Exception as exc:
        raise RuntimeError(f"Gemini parsing failed; no fallback was used: {exc}") from exc

def _extract(text: str, schema, demo: dict, kind: str) -> dict[str, Any]:
    key=os.getenv("GEMINI_API_KEY")
    if not key:
        return {"data":schema.model_validate(demo).model_dump(mode="json"),"parser":{"provider":"deterministic_demo","gemini_used":False,"warning":"Set GEMINI_API_KEY to enable Gemini structured extraction."}}
    try:
        from google import genai
        from google.genai import types
        client=genai.Client(api_key=key)
        prompt=f"Extract only explicitly stated {kind} facts into the supplied JSON schema. Do not invent missing information. This is data extraction, not medical advice.\n\n{text}"
        response=client.models.generate_content(model=os.getenv("GEMINI_MODEL","gemini-2.5-flash"),contents=prompt,config=types.GenerateContentConfig(response_mime_type="application/json",response_schema=schema))
        return {"data":schema.model_validate_json(response.text).model_dump(mode="json"),"parser":{"provider":"gemini","gemini_used":True}}
    except Exception as exc: raise RuntimeError(f"Gemini {kind} parsing failed; no fallback was used: {exc}") from exc

def parse_profile_text(text:str, patient_id="patient_001"):
    return _extract(text,MedicalProfile,{"patient_id":patient_id,"profile_version":1,"demographics":{},"diabetes":{"type":"T1D"},"medications":[],"conditions":[],"baseline_metrics":{},"source_metadata":{"demo":True}},"medical profile")

def parse_policy_text(text:str, patient_id="patient_001"):
    return _extract(text,DoctorPolicy,{"patient_id":patient_id,"policy_version":1,"risk_priorities":{},"clinical_facts":[],"hard_rules":[],"provenance":{"demo":True}},"clinician policy")

def explain_structured_forecast(forecast:dict[str,Any]) -> dict[str,Any]:
    # Deterministic explanation prevents an LLM from inventing causal or clinical claims.
    p=forecast.get("proposed",forecast); warning=(p.get("warnings") or ["This is a research prototype."])[0] if isinstance(p,dict) else "This is a research prototype."
    return {"summary":"The model compared the current-state baseline with the proposed structured action.","factors":["recent glucose trend","recorded insulin and carbohydrate context","proposed action details","available personal episode history"],"warning":warning,"generated_by":"deterministic_template"}
