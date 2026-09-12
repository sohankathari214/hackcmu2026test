from __future__ import annotations
import numpy as np
def monitor_drift(residuals, threshold_mg_dl=20., consecutive_days=3):
    vals=np.asarray(residuals,float); rolling=np.convolve(np.abs(vals),np.ones(min(len(vals),3))/min(len(vals),3),mode="valid") if len(vals) else np.array([])
    flagged=bool(len(rolling)>=consecutive_days and np.all(rolling[-consecutive_days:]>threshold_mg_dl))
    return {"drift_detected":flagged,"latest_rolling_absolute_residual":float(rolling[-1]) if len(rolling) else None,"recommendation":"contact a clinician / refit inputs" if flagged else "continue monitoring","clinical_validation":False}
