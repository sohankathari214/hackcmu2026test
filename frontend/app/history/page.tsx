"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import "../dashboard.css";

const api = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const patient = "patient_001";

export default function History() {
  const [episodes, setEpisodes] = useState<any[]>([]);
  const [selected, setSelected] = useState("");
  const [glucose, setGlucose] = useState({ g30: "", g60: "", g90: "", g120: "" });
  const [message, setMessage] = useState("");
  const load = () => { void fetch(`${api}/patients/${patient}/episodes`).then(r => r.json()).then(x => setEpisodes(x.episodes || [])).catch(() => setMessage("Could not load local episode history.")); };
  useEffect(load, []);
  const pending = episodes.filter(e => !e.quality?.complete);

  async function finalize(e: React.FormEvent) {
    e.preventDefault(); setMessage("");
    const observed = Object.fromEntries(Object.entries(glucose).filter(([, value]) => value !== "").map(([key, value]) => [key, Number(value)]));
    try {
      await fetch(`${api}/episodes/finalize`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ episode_id: selected, actual_action: episodes.find(x => x.episode_id === selected)?.planned_action || {}, observed }) }).then(r => r.ok ? r.json() : r.json().then(Promise.reject));
      setMessage("Outcome saved. This episode is now eligible for personalization if it is complete and unconfounded."); setSelected(""); setGlucose({ g30: "", g60: "", g90: "", g120: "" }); load();
    } catch (err: any) { setMessage(err?.detail || "Could not save the observed outcome."); }
  }

  return <main className="history-page"><header className="topbar"><div className="brand">GlucoPilot <small>Research prototype</small></div><nav><Link href="/">Overview</Link><Link href="/">Scenarios</Link><Link href="/history">History</Link></nav><div className="avatar">M</div></header><section className="history-wrap"><p className="eyebrow">PERSONALIZATION LOOP</p><h1>Episode history</h1><p className="muted">Save what actually happened after an event. The system computes residuals against the original population forecast; after 10 usable episodes, you can train a personal residual model.</p><div className="history-grid"><article className="card"><h2>Pending outcomes</h2>{pending.length ? <form onSubmit={finalize}><label>Saved scenario<select required value={selected} onChange={e => setSelected(e.target.value)}><option value="">Choose an event</option>{pending.map(e => <option value={e.episode_id} key={e.episode_id}>{e.planned_action?.action_type || "Scenario"} · {new Date(e.start_timestamp).toLocaleString()}</option>)}</select></label><div className="outcomes">{[30, 60, 90, 120].map(h => <label key={h}>+{h} min mg/dL<input inputMode="decimal" value={glucose[`g${h}` as keyof typeof glucose]} onChange={e => setGlucose({ ...glucose, [`g${h}`]: e.target.value })} /></label>)}</div><button type="submit">Save observed outcome</button></form> : <p className="muted">No pending events. Run a scenario, then choose “I’m doing this” to save one.</p>}{message && <p className="history-message">{message}</p>}</article><article className="card"><p className="eyebrow">TRAINING THRESHOLD</p><h2>{episodes.filter(e => e.quality?.usable_for_personalization).length} / 10 usable episodes</h2><div className="history-progress"><i style={{ width: `${Math.min(100, episodes.filter(e => e.quality?.usable_for_personalization).length * 10)}%` }} /></div><p className="muted">Only complete, unconfounded observations with a stored feature snapshot are included in personal training.</p></article></div><article className="card history-list"><h2>All recorded scenarios</h2>{episodes.length ? episodes.slice().reverse().map(e => <div key={e.episode_id}><span><b>{e.planned_action?.action_type || "Scenario"}</b><small>{e.start_timestamp}</small></span><span>{e.quality?.complete ? `Observed +60: ${e.observed?.g60 ?? "—"}` : "Awaiting observation"}</span><span>{e.quality?.usable_for_personalization ? "Eligible for training" : "Not yet eligible"}</span></div>) : <p className="muted">No saved episodes yet.</p>}</article></section></main>;
}
