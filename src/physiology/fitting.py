"""Small bounded least-squares parameter initializer for the grey-box POC."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def fit_patient_parameters(frame: pd.DataFrame) -> dict:
    """Fit only observable linear response coefficients; retain population defaults otherwise."""
    required={"glucose_mg_dl","carbs_g","bolus_units"}
    if not required.issubset(frame.columns) or len(frame)<24: return {"status":"insufficient_data","parameters":{},"clinical_validation":False}
    x=frame.sort_values("timestamp").copy(); g=x.glucose_mg_dl.to_numpy(float); dg=np.diff(g); carbs=x.carbs_g.fillna(0).to_numpy(float)[:-1]; insulin=x.bolus_units.fillna(0).to_numpy(float)[:-1]
    def residual(p): return (p[0]*carbs-p[1]*insulin-p[2]*(g[:-1]-110))-dg
    fit=least_squares(residual,x0=[1.,20.,.03],bounds=([0,0,0],[5,80,.3]))
    return {"status":"ok","parameters":{"carb_gain":float(fit.x[0]),"insulin_gain":float(fit.x[1]),"relaxation":float(fit.x[2])},"rmse":float(np.mean(fit.fun**2)**.5),"method":"bounded_least_squares","clinical_validation":False}
