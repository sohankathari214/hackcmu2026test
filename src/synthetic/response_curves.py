from __future__ import annotations
import numpy as np
H=(30,60,90,120)
def effect(action, traits, features, rng, noise=(1,1.5,2,2.5)):
    typ=action["action_type"]; ctx=action.get("context",{}); state=1+min(.35,float(features.get("recent_insulin_exposure",0) or 0)/20)+min(.2,float(features.get("carbs_120",0) or 0)/150)
    modifier=(1+.06*ctx.get("stress_level",0)*traits["stress_sensitivity"]+.08*bool(ctx.get("illness",False))*traits["illness_sensitivity"]+({"low":.04,"normal":0,"high":-.02}.get(ctx.get("hydration","normal"),0)*traits["hydration_sensitivity"]))
    if typ=="exercise":
        ex=action["exercise"]; curve={"cardio":(-8,-16,-20,-18),"resistance":(-2,-5,-8,-10),"mixed":(-5,-11,-15,-14)}[ex["category"]]; mag=ex["duration_minutes"]/45*{"low":.7,"moderate":1,"high":1.3}[ex["intensity"]]*traits["exercise_sensitivity"]*state
    elif typ=="caffeine": curve=(1,3,4,3); mag=action["caffeine_mg"]/150*traits["caffeine_sensitivity"]
    elif typ=="alcohol": curve=(0,-1,-4,-7); mag=action["alcohol_drink_count"]*traits["alcohol_sensitivity"]*(1+min(.2,float(features.get("recent_insulin_exposure",0) or 0)/20))
    else: curve=(5,10,12,9); mag=action.get("carbs_g",0)/30*traits["carb_sensitivity"]
    return {f"delta{h}":float(curve[j]*mag*modifier+rng.normal(0,noise[j])) for j,h in enumerate(H)}
