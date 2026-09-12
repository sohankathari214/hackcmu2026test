from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from .patient_traits import generate_traits
from .action_sampler import sample_action
from .response_curves import effect
from .provenance import PROVENANCE
from src.features.build_state import FEATURE_NAMES
def generate_hybrid_episodes(real, seed=42, sample_fraction=.1, simulator_version="poc-v1"):
    traits=generate_traits(real.patient_id.unique(),seed); rows=[]
    for _,r in real.dropna(subset=["y30","y60","y90","y120"]).iterrows():
        n=int(hashlib.sha256(f"{seed}:{r.patient_id}:{r.timestamp}".encode()).hexdigest()[:16],16); rng=np.random.default_rng(n)
        if rng.random()>sample_fraction: continue
        action=sample_action(rng); features=r.reindex(FEATURE_NAMES).to_dict(); delta=effect(action,traits[r.patient_id],features,rng)
        for k in ["action_type_none","action_type_exercise","action_type_meal","action_type_caffeine","action_type_alcohol","exercise_cardio","exercise_resistance","exercise_mixed","hydration_low","hydration_normal","hydration_high"]: features[k]=0.0
        features[f"action_type_{action['action_type']}"]=1.0
        ctx=action["context"]; features["stress_level"]=ctx["stress_level"]; features["illness"]=float(ctx["illness"]); features[f"hydration_{ctx['hydration']}"]=1.; features["sleep_hours"]=ctx["sleep_hours"]
        if action["action_type"]=="exercise":
            ex=action["exercise"]; features[f"exercise_{ex['category']}"]=1.; features["exercise_duration_minutes"]=ex["duration_minutes"]; features["exercise_intensity_numeric"]={"low":1,"moderate":2,"high":3}[ex["intensity"]]; features["exercise_start_delay_minutes"]=ex["start_delay_minutes"]
        elif action["action_type"]=="caffeine": features["caffeine_mg"]=action["caffeine_mg"]
        elif action["action_type"]=="alcohol": features["alcohol_grams"]=action["alcohol_grams"]; features["alcohol_drink_count"]=action["alcohol_drink_count"]
        out={**r.to_dict(),**features,**action,**{f"real_baseline_g{h}":float(r[f"y{h}"]) for h in (30,60,90,120)},**delta,**PROVENANCE,"data_source":"hybrid","simulator_version":simulator_version,"random_seed":seed}
        for h in (30,60,90,120): out[f"y{h}"]=float(np.clip(r[f"y{h}"]+delta[f"delta{h}"],40,400))
        out["episode_id"]=f"synthetic_{r.patient_id}_{pd.Timestamp(r.timestamp).strftime('%Y%m%d%H%M%S')}_{n%100000}"; out["base_state_id"]=f"{r.patient_id}:{pd.Timestamp(r.timestamp).isoformat()}"; rows.append(out)
    return pd.DataFrame(rows),traits
