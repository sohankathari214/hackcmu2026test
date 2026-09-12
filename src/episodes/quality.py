from __future__ import annotations

from typing import Any


def quality_check(episode: dict[str, Any]):
    quality = episode.get("quality", {})
    quality["complete"] = quality.get("complete", False)
    quality["confounded"] = quality.get("confounded", False)
    quality["usable_for_personalization"] = quality.get("usable_for_personalization", False)
    quality["exclusion_reasons"] = quality.get("exclusion_reasons", [])
    return quality
