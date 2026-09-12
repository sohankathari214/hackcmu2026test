from __future__ import annotations
from typing import Any
from .permission import evaluate_permission
def plan_day(current_state:Any, actions:list[dict[str,Any]], policy:Any=None):
    cards=[{"action":a,"result":evaluate_permission(current_state,a,policy)} for a in actions]
    return {"timeline":cards,"disclaimer":"Scenario planning POC; not clinical guidance or insulin dosing."}
