from __future__ import annotations
from typing import Any
from src.physiology import estimate_state,simulate
from src.safety import evaluate_safety

def evaluate_permission(current_state:Any, action:dict[str,Any], policy:Any=None) -> dict[str,Any]:
    gate=evaluate_safety(current_state,action,policy)
    baseline=simulate(estimate_state(current_state),[],horizon_minutes=480)
    if not gate["allowed"]: return {"verdict":"denied","safety":gate,"baseline":baseline,"proposed":None,"alternatives":[]}
    proposed=simulate(estimate_state(current_state),[action],horizon_minutes=480)
    risk=proposed["risk"]; alternatives=[]
    for variant in ({**action,"start_delay_minutes":60},{**action,"duration_minutes":max(15,int(action.get("duration_minutes",30)/2))}):
        sim=simulate(estimate_state(current_state),[variant],horizon_minutes=480); alternatives.append({"action":variant,"risk":sim["risk"]})
    verdict="approved" if risk["hypoglycemia_probability"]<=.05 else "approved_with_condition"
    return {"verdict":verdict,"safety":gate,"baseline":baseline,"proposed":proposed,"alternatives":alternatives,"explanation":{"top_factors":["current glucose","active insulin estimate","planned action"],"clinical_validation":False}}
