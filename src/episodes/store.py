from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class EpisodeStore:
    def __init__(self,path:str|Path="artifacts/episodes.jsonl"): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def append(self,episode:dict[str,Any]):
        with self.path.open("a") as f: f.write(json.dumps(episode,default=str)+"\n")
        return episode
    def patient(self,patient_id:str):
        if not self.path.exists(): return []
        return [x for x in (json.loads(line) for line in self.path.read_text().splitlines() if line.strip()) if x.get("patient_id")==patient_id]
    def replace(self,episode:dict[str,Any]):
        rows=[] if not self.path.exists() else [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        rows=[episode if x.get("episode_id")==episode.get("episode_id") else x for x in rows]
        self.path.write_text("".join(json.dumps(x,default=str)+"\n" for x in rows)); return episode
