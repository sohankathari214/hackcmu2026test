from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataQualityResult:
    level: str
    components: dict


def data_quality(feature_vector: pd.Series, state) -> dict:
    components = {
        "current_glucose_present": bool(pd.notna(feature_vector.get("glucose_current"))),
        "recent_glucose_history": bool(feature_vector.get("glucose_lag_120", 0) is not None),
        "recent_insulin_available": bool(pd.notna(feature_vector.get("bolus_units_30"))),
        "recent_meal_available": bool(pd.notna(feature_vector.get("carbs_30"))),
        "action_complete": bool(getattr(state.proposed_action, "action_type", None) is not None),
        "personal_history_quantity": "low",
    }

    if components["current_glucose_present"] and components["recent_glucose_history"]:
        level = "high"
    elif components["current_glucose_present"]:
        level = "medium"
    else:
        level = "low"

    return {"level": level, "components": components}
