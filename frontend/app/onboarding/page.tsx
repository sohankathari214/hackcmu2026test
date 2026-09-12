import Link from "next/link";
import {Shell} from "../../components/Shell";
export default function Onboarding(){return <Shell title="Welcome to GlucoPilot"><section className="card"><h2>Understand what may happen before you act.</h2><p className="muted">Maya’s profile, care plan, and demo data are preloaded. The full product confirms structured profile and clinician-note extraction before use.</p><Link href="/simulate"><button>Use demo profile</button></Link></section></Shell>}
