from __future__ import annotations
from src.inference.forecast import forecast_scenario
def compare_scenarios(profile,state,actions=None,population_models=None,patient_models=None,episodes=None):
    base=state.model_copy(deep=True); base.proposed_action={"action_type":"none"}; prop=state.model_copy(deep=True); prop.proposed_action=(actions or [{"action_type":"none"}])[0]
    b=forecast_scenario(profile,base,population_models,patient_models,episodes); p=forecast_scenario(profile,prop,population_models,patient_models,episodes)
    return {"baseline":b,"proposed":p,"difference":{f"delta{h}":p["population_forecast"][f"g{h}"]-b["population_forecast"][f"g{h}"] for h in (30,60,90,120)}}
