"""Population-level exposure projections for the POC; not individualized risk predictions."""
from __future__ import annotations
import numpy as np
def project_long_term(glucose_curve, days=90, years=10):
    g=np.asarray(glucose_curve,float); mean=float(np.mean(g)); a1c=(mean+46.7)/28.7
    exposure=float(np.mean(np.maximum(g-140,0))); variability=float(np.std(g)); risk=min(.95,max(.01,.08+.0015*exposure+.0007*variability))
    return {"horizon_days":days,"a1c_proxy":a1c,"population_level_ten_year_risk_proxy":{"retinopathy":risk,"kidney":risk*.75,"neuropathy":risk*.85},"provenance":{"clinical_validation":False,"meaning":"illustrative population-level exposure proxy, not an individual complication probability"}}
