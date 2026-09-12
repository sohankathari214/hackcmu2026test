"use client";

import { useState } from "react";
import { Activity, AlertTriangle, ArrowDownRight, Clock3, FileText, FlaskConical, Gauge, Info, Play, ShieldCheck, Sparkles } from "lucide-react";
import "./dashboard.css";

const API = "/api/backend";
const profile = { patient_id: "patient_001", profile_version: 1, demographics: { age_years: 24, weight_kg: 64 }, diabetes: { type: "T1D", years_since_diagnosis: 11, insulin_delivery: "pump" }, medications: [], conditions: [], baseline_metrics: {}, source_metadata: { demo: true } };
const seed = "I’m at 145 and trending down. I ate 50 grams of carbs and took 3 units an hour ago. I want to go for a moderate 45-minute run now.";

function Chart({ values, tone = "coral" }: { values: number[]; tone?: string }) {
  const w = 620, h = 185, min = 60, max = 200;
  const points = values.map((v, i) => `${i * w / (values.length - 1)},${h - (v - min) / (max - min) * h}`).join(" ");
  return <svg viewBox={`0 0 ${w} ${h}`} className="chart" preserveAspectRatio="none">
    <line x1="0" x2={w} y1={h - (140 - min) / (max - min) * h} y2={h - (140 - min) / (max - min) * h} className="target" />
    <line x1="0" x2={w} y1={h - (70 - min) / (max - min) * h} y2={h - (70 - min) / (max - min) * h} className="low" />
    <polyline points={points} className={`line ${tone}`} />
    {values.map((v, i) => <circle key={i} cx={i * w / (values.length - 1)} cy={h - (v - min) / (max - min) * h} r="3" className={`dot ${tone}`} />)}
  </svg>;
}

export default function Home() {
  const [text, setText] = useState(seed);
  const [parsed, setParsed] = useState<any>();
  const [result, setResult] = useState<any>();
  const [saved, setSaved] = useState<any>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const baseline = result?.baseline?.population_forecast;
  const proposed = result?.proposed?.population_forecast;

  async function run() {
    setBusy(true); setError("");
    try {
      const p = await fetch(`${API}/parse-state`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }).then(r => r.ok ? r.json() : r.json().then(Promise.reject));
      const comparison = await fetch(`${API}/compare-scenarios`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profile, state: p.state }) }).then(r => r.ok ? r.json() : r.json().then(Promise.reject));
      setParsed(p); setResult(comparison); setSaved(undefined);
    } catch (e: any) { setError(e?.detail || "Could not connect to the GlucoPilot API."); }
    finally { setBusy(false); }
  }

  async function saveEpisode() {
    if (!parsed || !proposed) return;
    setBusy(true); setError("");
    try {
      const episode = await fetch(`${API}/episodes/create`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patient_id: profile.patient_id, medical_profile: profile, state: parsed.state, planned_action: parsed.state.proposed_action, prediction: proposed }) }).then(r => r.ok ? r.json() : r.json().then(Promise.reject));
      setSaved(episode);
    } catch (e: any) { setError(e?.detail || "Could not save this episode."); }
    finally { setBusy(false); }
  }

  const trajectory = proposed ? [145, proposed.g30, proposed.g60, proposed.g90, proposed.g120] : [145, 138, 118, 102, 96];
  return <main className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark"><Activity size={18} /></span>GlucoPilot <small>Research prototype</small></div><nav><a href="#overview">Overview</a><a href="#simulator">Scenarios</a><a href="/history">History</a><a href="/insights">Insights</a></nav><div className="avatar">M</div></header>
    <section className="page-wrap" id="overview">
      <div className="welcome"><div><p className="eyebrow">DEMO PATIENT · CURRENT CONTEXT</p><h1>Good morning, Maya.</h1><p className="muted">Here’s your current context and what your model estimates may happen next.</p></div><a href="/care-plan" className="outline">View care plan</a></div>
      <div className="safety"><ShieldCheck size={18} /><span><b>Decision support, not medical advice.</b> GlucoPilot does not recommend insulin doses or determine whether an activity is medically safe.</span><Info size={16} /></div>
      <div className="layout"><section>
        <div className="heading"><div><p className="eyebrow">CURRENT STATE</p><h2>What’s happening now</h2></div></div>
        <div className="state-grid">
          <article className="card glucose"><p><Gauge size={16} /> Current glucose</p><strong>145 <small>mg/dL</small></strong><span className="trend"><ArrowDownRight size={16} /> Falling slowly</span><div className="spark"><Chart values={[162, 158, 154, 150, 148, 145]} tone="teal" /></div><em>Demo state · last reading 6 min ago</em></article>
          <article className="card"><p><Clock3 size={16} /> Recent inputs</p><dl><div><dt>Insulin</dt><dd>3.0 units · 60 min ago</dd></div><div><dt>Carbohydrates</dt><dd>50 g · 60 min ago</dd></div><div><dt>Sleep</dt><dd>5.0 hours last night</dd></div></dl><button onClick={() => setText(seed)}>Restore demo context</button></article>
        </div>
        <div className="heading simhead" id="simulator"><div><p className="eyebrow">SCENARIO SIMULATOR</p><h2>What might happen?</h2></div><span className="ready">● Model ready</span></div>
        <article className="card simulator"><label>What are you considering?</label><textarea value={text} onChange={e => setText(e.target.value)} /><div className="runrow"><span><Sparkles size={15} /> OpenAI extracts structured input; models generate the forecast.</span><button onClick={run} disabled={busy}>{busy ? "Running simulation…" : "Run simulation"} <Play size={14} /></button></div>{error && <p className="error">{error}</p>}{parsed && <div className="parsed"><b>I understood</b><span>{parsed.state.proposed_action.action_type} · parser: {parsed.parser.provider}</span></div>}<div className="options"><div><i className="coral" /><b>Run now</b><span>{proposed ? `${proposed.g120?.toFixed(0)} mg/dL at +120` : "simulate to estimate"}</span></div><div><i className="teal" /><b>No action</b><span>{baseline ? `${baseline.g120?.toFixed(0)} mg/dL at +120` : "baseline"}</span></div></div>{result && <div className="episode-row">{saved ? <span>Episode saved. Add observed glucose in History after the event.</span> : <button onClick={saveEpisode} disabled={busy}>I’m doing this — save for learning</button>}</div>}</article>
        <div className="heading"><div><p className="eyebrow">MODEL EXPLANATION</p><h2>Why this estimate?</h2></div></div>
        <article className="card explain"><div><span className="coral-icon"><ArrowDownRight size={17} /></span><p><b>Proposed activity may lower the trajectory</b><small>The action response is learned from synthetic POC augmentation, not clinical evidence.</small></p></div><div><span className="blue-icon"><FlaskConical size={17} /></span><p><b>Recent insulin and carbohydrate context are included</b><small>The feature engine uses only data available at the scenario timestamp.</small></p></div><div><span className="amber-icon"><AlertTriangle size={17} /></span><p><b>Short sleep adds uncertainty</b><small>Prediction bands and data-quality signals remain available from the model output.</small></p></div></article>
      </section><aside>
        <div className="heading"><div><p className="eyebrow">PREDICTED TRAJECTORY</p><h2>Next 2 hours</h2></div></div>
        <article className="card trajectory"><div className="legend"><span><i className="coral" /> Proposed action</span><span><i className="teal" /> Baseline</span></div><Chart values={trajectory} /><div className="labels"><span>Now</span><span>30m</span><span>60m</span><span>90m</span><span>120m</span></div>{result ? <div className="callout"><b>{result.difference.delta60?.toFixed(0)} mg/dL estimated difference at 60 minutes.</b><span>{result.proposed.warnings?.[0] || "Population forecast with explicit provenance."}</span></div> : <div className="callout"><b>Run a scenario to see a model comparison.</b><span>Baseline and proposed action will be displayed here.</span></div>}</article>
        <article className="card personal"><div><p className="eyebrow">PERSONALIZATION</p><h2>Your model learns</h2></div><strong>{result?.proposed?.personalization?.active ? "Active" : "Learning"}</strong><p>Complete observed episodes train a general personal residual model after enough history.</p><div className="progress"><i style={{ width: result?.proposed?.personalization?.active ? "66%" : "28%" }} /></div><a href="/history"><FileText size={15} /> Review history</a></article>
      </aside></div>
    </section><footer>GlucoPilot · Research and engineering prototype · Your data remains local in this demo</footer>
  </main>;
}
