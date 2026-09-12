from __future__ import annotations
from typing import Any
from src.physiology import estimate_state,simulate
from src.physiology.long_term import project_long_term
def evaluate_lifestyle_change(state:Any, events:list[dict[str,Any]]):
    base=simulate(estimate_state(state),[],horizon_minutes=1440); proposed=simulate(estimate_state(state),events,horizon_minutes=1440)
    return {"baseline":project_long_term(base["glucose_mg_dl"]["p50"]),"proposed":project_long_term(proposed["glucose_mg_dl"]["p50"]),"disclaimer":"Illustrative POC exposure comparison; not medical risk prediction."}
