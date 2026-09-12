from __future__ import annotations
import json
from src.physiology import estimate_state, simulate
from src.schemas.current_state import CurrentState
from src.safety import evaluate_safety
def test_simulation_is_reproducible_and_returns_uncertainty():
    s=CurrentState.model_validate(json.load(open("examples/current_state.json"))); state=estimate_state(s); a=simulate(state,[{"action_type":"exercise","duration_minutes":30}],horizon_minutes=30,samples=5,seed=7); b=simulate(state,[{"action_type":"exercise","duration_minutes":30}],horizon_minutes=30,samples=5,seed=7)
    assert a["glucose_mg_dl"]==b["glucose_mg_dl"] and len(a["glucose_mg_dl"]["p10"])==7
def test_safety_blocks_low_glucose_exercise():
    payload=json.load(open("examples/current_state.json"));payload["glucose"]["current_mg_dl"]=80; s=CurrentState.model_validate(payload); assert not evaluate_safety(s,{"action_type":"exercise"})["allowed"]
