"use client";
import {useEffect,useState} from "react";
import {Shell} from "../../components/Shell";
const api=process.env.NEXT_PUBLIC_API_URL||"http://localhost:8000";
export default function History(){const [episodes,setEpisodes]=useState<any[]>([]);useEffect(()=>{fetch(`${api}/patients/patient_001/episodes`).then(r=>r.json()).then(x=>setEpisodes(x.episodes||[])).catch(()=>{})},[]);return <Shell title="History"><section className="card"><h2>Previous scenarios</h2>{episodes.length?episodes.map(e=><div className="row" key={e.episode_id}><span>{e.planned_action?.action_type||"Scenario"} · {e.start_timestamp}</span><span>+60 predicted {e.prediction?.g60??"—"} · actual {e.observed?.g60??"pending"}</span></div>):<p className="muted">No saved episodes yet. Confirm an action after reviewing a scenario to begin personal learning.</p>}</section></Shell>}
