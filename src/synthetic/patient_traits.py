from __future__ import annotations
import hashlib
import numpy as np
TRAITS=("exercise_sensitivity","carb_sensitivity","caffeine_sensitivity","alcohol_sensitivity","stress_sensitivity","sleep_sensitivity","illness_sensitivity","hydration_sensitivity")
def generate_traits(patient_ids, seed=42):
    out={}
    for pid in sorted(set(patient_ids)):
        n=int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:16],16); rng=np.random.default_rng(n)
        out[pid]={k:round(float(rng.uniform(.6,1.4)),6) for k in TRAITS}
    return out
