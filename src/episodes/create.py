from __future__ import annotations

from datetime import datetime
from typing import Any


def create_episode(patient_id: str, state: dict[str, Any], planned_action: dict[str, Any], prediction: dict[str, Any]):
    return {
        "episode_id": f"episode_{patient_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "patient_id": patient_id,
        "start_timestamp": state.get("timestamp"),
        "pre_action_state": state,
        "planned_action": planned_action,
        "actual_action": None,
        "prediction": prediction,
        "observed": {},
        "residuals": {},
        "unexpected_events": [],
        "quality": {
            "complete": False,
            "confounded": False,
            "usable_for_personalization": False,
            "exclusion_reasons": [],
        },
    }
