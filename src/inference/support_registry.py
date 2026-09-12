from __future__ import annotations

from typing import Any


class ActionSupportRegistry:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def check(self, action_type: str) -> tuple[bool, str | None]:
        if action_type == "exercise":
            if self.config.get("exercise", {}).get("supported", False):
                return True, None
            return False, "insufficient_training_data"
        if action_type in {"meal", "caffeine", "alcohol"}:
            support = self.config.get(action_type, {}).get("supported", False)
            if support:
                return True, None
            return False, "insufficient_training_data"
        return True, None
