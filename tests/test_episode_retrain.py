import pandas as pd
from src.episodes.retrain import retrain_from_episodes
from src.features.build_state import FEATURE_NAMES
def test_retrain_from_finalized_feature_snapshots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); episodes=[]
    for i in range(10): episodes.append({"quality":{"usable_for_personalization":True},"feature_vector":{x:float(i) for x in FEATURE_NAMES},"residuals":{f"r{h}":float(i) for h in (30,60,90,120)}})
    assert retrain_from_episodes("p",episodes)["status"]=="retrained"
