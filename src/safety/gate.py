from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def evaluate_safety(state: Any, action: dict[str,Any], policy: Any=None) -> dict[str,Any]:
    cfg=yaml.safe_load((Path(__file__).parents[2]/"configs/params.yaml").read_text())["safety"]; warnings=[]; block=[]
    glucose=state.glucose.get("current_mg_dl",state.glucose.get("mg_dl")); ts=state.timestamp
    if glucose is None: block.append("A current glucose reading is required.")
    if action.get("action_type")=="exercise" and glucose is not None and glucose<float(cfg["min_glucose_for_exercise_mg_dl"]): block.append("Exercise scenario blocked by POC safety threshold; confirm with a clinician.")
    if float((state.context or {}).get("ketones_mmol_l",0))>=float(cfg["ketone_escalation_mmol_l"]): block.append("Ketone red flag: contact a clinician; optimization is suppressed.")
    for rule in (getattr(policy,"hard_rules",[]) if policy else []):
        if rule.get("enabled",True) and rule.get("action_type")==action.get("action_type") and rule.get("deny",False): block.append(rule.get("message","Blocked by clinician-authored policy."))
    if (state.context or {}).get("illness"): warnings.append("Illness context increases uncertainty; this is not medical advice.")
    return {"outcome":"escalate" if block else "proceed","allowed":not block,"block_reasons":block,"warnings":warnings,"provenance":{"safety_rules":"configs/params.yaml and optional clinician policy","clinical_validation":False}}
