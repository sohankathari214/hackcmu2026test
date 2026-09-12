from __future__ import annotations
import numpy as np
def sample_action(rng):
    typ=rng.choice(["exercise","caffeine","alcohol","meal"],p=[.55,.2,.15,.1]); ctx={"stress_level":int(rng.integers(0,4)),"illness":bool(rng.random()<.08),"hydration":str(rng.choice(["low","normal","high"],p=[.2,.65,.15])),"sleep_hours":float(rng.uniform(5,9))}
    action={"action_type":typ,"context":ctx}
    if typ=="exercise": action["exercise"]={"category":str(rng.choice(["cardio","resistance","mixed"])),"duration_minutes":int(rng.integers(15,91)),"intensity":str(rng.choice(["low","moderate","high"])),"start_delay_minutes":0}
    if typ=="caffeine": action["caffeine_mg"]=int(rng.integers(50,301))
    if typ=="alcohol": action["alcohol_drink_count"]=int(rng.integers(1,4)); action["alcohol_grams"]=action["alcohol_drink_count"]*14
    if typ=="meal": action["carbs_g"]=int(rng.integers(10,61))
    return action
