from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

@dataclass
class PhysiologicalState:
    glucose_mg_dl: float
    gut_carbs_g: float = 0.0
    active_insulin_u: float = 0.0
    blood_alcohol_units: float = 0.0
    caffeine_mg: float = 0.0
    exercise_load: float = 0.0
    hydration: float = 1.0
    stress_level: float = 0.0
    illness: bool = False
    sleep_debt_hours: float = 0.0
    ketones_mmol_l: float = 0.0
    covariance: float = 15.0
    def model_dump(self): return asdict(self)

def estimate_state(current_state: Any) -> PhysiologicalState:
    """A deterministic observable-state estimator for the POC; not a clinical filter."""
    glucose = current_state.glucose.get("current_mg_dl", current_state.glucose.get("mg_dl", 110.0))
    insulin = current_state.insulin or {}; food = current_state.food or {}; context = current_state.context or {}; sleep = current_state.sleep or {}; activity = current_state.activity or {}
    active = sum(float(d.get("units", 0)) * np.exp(-float(d.get("minutes_ago", 0))/240) for d in insulin.get("recent_doses", []))
    carbs = sum(float(m.get("carbs_g", 0)) * np.exp(-float(m.get("minutes_ago", 0))/75) for m in food.get("recent_meals", []))
    caffeine = sum(float(x.get("dose_mg", x.get("caffeine_mg", 0))) * np.exp(-float(x.get("minutes_ago", 0))/(5*60/np.log(2))) for x in context.get("caffeine_events", []))
    alcohol = sum(float(x.get("drink_count", 0)) * np.exp(-float(x.get("minutes_ago", 0))/180) for x in context.get("alcohol_events", []))
    hydration = {"low": .75, "adequate": 1., "normal": 1., "high": 1.05}.get(str(context.get("hydration", "normal")).lower(), 1.)
    return PhysiologicalState(float(glucose), carbs, active, alcohol, caffeine, float(activity.get("recent_activity_minutes", 0))/60, hydration, float(context.get("stress_level",0)), bool(context.get("illness",False)), max(0., 8-float(sleep.get("last_sleep_hours",8))), float(context.get("ketones_mmol_l",0)))
