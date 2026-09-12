from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.current_state import CurrentState
from src.schemas.medical_profile import MedicalProfile
from src.inference.forecast import forecast_scenario


if __name__ == "__main__":
    profile = MedicalProfile.model_validate(json.loads(Path("examples/medical_profile.json").read_text()))
    state = CurrentState.model_validate(json.loads(Path("examples/current_state.json").read_text()))
    res = forecast_scenario(profile, state)
    print(json.dumps(res, indent=2))
