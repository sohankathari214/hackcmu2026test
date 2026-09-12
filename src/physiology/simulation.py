from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from .state import PhysiologicalState

def _params(params=None):
    if params: return params
    return yaml.safe_load((Path(__file__).parents[2]/"configs/params.yaml").read_text())["simulator"]

def simulate(state: PhysiologicalState, events: list[dict[str,Any]], parameters: dict[str,Any]|None=None, horizon_minutes: int=480, samples: int|None=None, seed: int=42) -> dict[str,Any]:
    """Five-minute transparent compartment POC with Monte-Carlo uncertainty.

    This deliberately does not optimize or recommend insulin; its effects are
    conservative visualization assumptions, not validated physiology.
    """
    p=_params(parameters); dt=int(p.get("step_minutes",5)); times=np.arange(0,horizon_minutes+dt,dt); n=samples or int(p.get("monte_carlo_samples",100)); rng=np.random.default_rng(seed); trajectories=[]
    for _ in range(n):
        glucose=state.glucose_mg_dl+rng.normal(0,state.covariance*.15); gut=state.gut_carbs_g; insulin=state.active_insulin_u; alcohol=state.blood_alcohol_units; caffeine=state.caffeine_mg; load=state.exercise_load; curve=[]
        for t in times:
            for e in events:
                if int(e.get("start_delay_minutes",0)) == int(t):
                    typ=e.get("action_type",e.get("type",""))
                    if typ=="meal": gut += float(e.get("carbs_g",0))
                    elif typ=="exercise": load += float(e.get("duration_minutes",30))/45*{"low":.7,"moderate":1,"high":1.3}.get(str(e.get("intensity","moderate")),1)
                    elif typ=="caffeine": caffeine += float(e.get("caffeine_mg",e.get("dose_mg",0)))
                    elif typ=="alcohol": alcohol += float(e.get("drink_count",0))
            carb_flux=gut/max(float(p["carb_absorption_minutes"]),1); insulin_flux=insulin/max(float(p["insulin_action_minutes"]),1)
            effect=(float(p["meal_gain_mg_dl_per_g"])*carb_flux-float(p["insulin_gain_mg_dl_per_u"])*insulin_flux-float(p["exercise_gain_mg_dl"])*load*.02+float(p["caffeine_gain_mg_dl"])*caffeine/1000-float(p["alcohol_delayed_gain_mg_dl"])*alcohol*(1-np.exp(-t/180))+float(p["stress_gain_mg_dl"])*state.stress_level+float(p["illness_gain_mg_dl"])*state.illness+float(p["hydration_gain_mg_dl"])*(1-state.hydration))
            glucose += dt*(float(p["target_glucose_mg_dl"])-glucose)*float(p["glucose_relaxation_per_min"])+dt*effect+rng.normal(0,1.2)
            glucose=float(np.clip(glucose,*p["glucose_clip_mg_dl"])); gut*=np.exp(-dt/float(p["carb_absorption_minutes"])); insulin*=np.exp(-dt/float(p["insulin_action_minutes"])); alcohol*=np.exp(-dt/180); caffeine*=np.exp(-dt/(5*60/np.log(2))); load*=np.exp(-dt/180); curve.append(glucose)
        trajectories.append(curve)
    arr=np.asarray(trajectories); lo,hi=np.quantile(arr,[.1,.9],axis=0); med=np.median(arr,axis=0)
    return {"times_minutes":times.tolist(),"glucose_mg_dl":{"p10":lo.tolist(),"p50":med.tolist(),"p90":hi.tolist()},"risk":{"hypoglycemia_probability":float((arr.min(axis=1)<70).mean()),"hyperglycemia_probability":float((arr.max(axis=1)>180).mean())},"state":state.model_dump(),"provenance":{"model":"transparent grey-box POC","clinical_validation":False,"simulator_version":p.get("version","greybox-poc-v1")}}
