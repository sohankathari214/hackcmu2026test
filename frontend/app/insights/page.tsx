"use client";
import {useEffect,useState} from "react";
import {Shell} from "../../components/Shell";
const api="/api/backend";
export default function Insights(){const [status,setStatus]=useState<any>();useEffect(()=>{fetch(`${api}/patients/patient_001/model-status`).then(r=>r.json()).then(setStatus)},[]);return <Shell title="Your model is learning you"><section className="grid"><div className="card"><h2>Personalization progress</h2><div className="metric">{status?.valid_episode_count??0}</div><p className="muted">usable episodes</p><p>Population model: active<br/>Personal residual model: {status?.active?"active":"learning"}<br/>Correction weight: {status?.alpha?.toFixed?.(2)??"0.00"}</p></div><div className="card"><h2>How to read this</h2><p className="muted">Observed outcomes are compared with the population forecast. A general residual model learns recurring differences without using hidden simulator traits.</p></div></section></Shell>}
