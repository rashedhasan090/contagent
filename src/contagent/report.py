"""Plain-text and Markdown rendering of forecasts, traces and plans."""

from __future__ import annotations

from .engine import RunResult
from .interventions import Intervention, Ranked
from .montecarlo import Forecast
from .scenario import Scenario


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _bar(x: float, width: int = 20) -> str:
    filled = int(round(x * width))
    return "#" * filled + "." * (width - filled)


def _tti(f: Forecast) -> str:
    if f.tti_median is None:
        return "n/a (no impact observed)"
    return f"median round {f.tti_median:g}, p90 round {f.tti_p90:g}"


def render_forecast(s: Scenario, f: Forecast) -> str:
    lo, hi = f.p_impact_ci
    lines = [
        f"Scenario: {s.name}  ({len(s.agents)} agents, {len(s.edges)} channels, {f.runs} runs, backend={f.backend})",
    ]
    if s.question:
        lines.append(f"Question: {s.question}")
    lines += [
        "",
        f"  P(impact)                 {_pct(f.p_impact)}   95% CI [{_pct(lo).strip()}, {_pct(hi).strip()}]",
        f"  P(detected before impact) {_pct(f.p_detect_before_impact)}",
        f"  Mean blast radius         {f.mean_blast_radius:.2f} agents",
        f"  Time to impact            {_tti(f)}",
        "",
        "  Compromise rate by agent:",
    ]
    width = max(len(a) for a in f.agent_compromise) if f.agent_compromise else 4
    for aid, rate in sorted(f.agent_compromise.items(), key=lambda kv: -kv[1]):
        extra = f"  impact {_pct(f.agent_impact[aid]).strip()}" if aid in f.agent_impact else ""
        lines.append(f"    {aid:<{width}} {_bar(rate)} {_pct(rate)}{extra}")
    if f.top_paths:
        lines += ["", "  Most common kill chains (share of impactful runs):"]
        for path, share in f.top_paths:
            lines.append(f"    {_pct(share)}  {path}")
    return "\n".join(lines)


def render_comparison(base: Forecast, other: Forecast, label: str) -> str:
    d = other.p_impact - base.p_impact
    arrow = "down" if d < 0 else ("up" if d > 0 else "unchanged")
    return "\n".join(
        [
            f"What-if: {label}",
            f"  P(impact)          {_pct(base.p_impact)} -> {_pct(other.p_impact)}  ({arrow} {abs(100 * d):.1f} pts)",
            f"  Detected first     {_pct(base.p_detect_before_impact)} -> {_pct(other.p_detect_before_impact)}",
            f"  Mean blast radius  {base.mean_blast_radius:.2f} -> {other.mean_blast_radius:.2f}",
        ]
    )


def render_ranking(base: Forecast, ranked: list[Ranked], top: int = 10) -> str:
    lines = [f"Baseline P(impact) {_pct(base.p_impact)} over {base.runs} runs. Single-change hardening options:", ""]
    lines.append(f"  {'intervention':<34} {'P(impact)':>9} {'reduction':>10} {'cost':>5} {'pts/cost':>8}")
    for r in ranked[:top]:
        lines.append(
            f"  {str(r.intervention):<34} {_pct(r.p_impact):>9} {100 * r.delta:>8.1f}pt {r.intervention.cost:>5.1f} {100 * r.efficiency:>8.1f}"
        )
    return "\n".join(lines)


def render_plan(base: Forecast, steps: list[tuple[Intervention, Forecast]], budget: float) -> str:
    lines = [f"Greedy hardening plan (budget {budget:g}). Start: P(impact) {_pct(base.p_impact)}"]
    spent = 0.0
    for i, (iv, f) in enumerate(steps, 1):
        spent += iv.cost
        lines.append(f"  {i}. {str(iv):<34} cost {iv.cost:>3.1f} (total {spent:>4.1f})  ->  P(impact) {_pct(f.p_impact)}")
    if not steps:
        lines.append("  No affordable intervention reduced risk.")
    return "\n".join(lines)


def render_trace(s: Scenario, r: RunResult) -> str:
    icons = {
        "inject": ">>", "blocked": "XX", "flag": "!!", "comply": "**", "ignore": "..",
        "relay": "->", "impact": "$$", "gate_block": "##", "quarantine": "[]", "warn": "~~",
    }
    lines = [f"Episode trace for '{s.name}' (seed {r.seed})", ""]
    for e in r.events:
        if e.kind == "relay":
            continue  # relays are implied by the next round's judgement
        src = f" <- {e.src}" if e.src and e.kind not in ("impact", "quarantine") else ""
        lines.append(f"  r{e.round:02d} {icons.get(e.kind, '  ')} {e.kind:<10} {e.agent:<16}{src:<20} {e.detail}")
    lines.append("")
    if r.impact:
        rnd, agent, tool = r.impacts[0]
        lines.append(f"  Outcome: IMPACT in round {rnd}: {agent} ran {tool}. Chain: {' -> '.join(r.impact_chain())}")
    else:
        lines.append("  Outcome: no impact.")
    if r.first_alert_round is not None:
        lines.append(f"  First alert in round {r.first_alert_round}.")
    lines.append(f"  Compromised: {', '.join(sorted(r.compromised)) or 'none'}")
    return "\n".join(lines)


def mermaid(s: Scenario, f: Forecast | None = None) -> str:
    """Mermaid flowchart of the agent graph, optionally shaded by compromise rate."""
    lines = ["flowchart LR"]
    for a in s.agents:
        shape = ("[(", ")]") if a.kind == "store" else (("([", "])") if a.kind == "human" else ("[", "]"))
        label = a.id
        if f is not None:
            label += f"<br/>{100 * f.agent_compromise.get(a.id, 0):.0f}% compromised"
        if a.defenses:
            label += "<br/><i>" + ", ".join(a.defenses) + "</i>"
        lines.append(f'  {a.id}{shape[0]}"{label}"{shape[1]}')
    entries = sorted({inj.agent for inj in s.injections})
    lines.append('  external{{"untrusted input"}}')
    for e in entries:
        lines.append(f"  external -.-> {e}")
    for e in s.edges:
        lbl = f"|{e.label} t={e.trust:g}|" if e.label else f"|t={e.trust:g}|"
        lines.append(f"  {e.src} -->{lbl} {e.dst}")
    if f is not None:
        for a in s.agents:
            rate = f.agent_compromise.get(a.id, 0)
            if rate >= 0.5:
                lines.append(f"  style {a.id} fill:#f8b4b4")
            elif rate >= 0.2:
                lines.append(f"  style {a.id} fill:#fde68a")
    return "\n".join(lines)


def markdown_report(
    s: Scenario,
    f: Forecast,
    ranked: list[Ranked] | None = None,
    sample: RunResult | None = None,
) -> str:
    lo, hi = f.p_impact_ci
    out = [
        f"# contagent risk report: {s.name}",
        "",
        s.description.strip(),
        "",
    ]
    if s.question:
        out += [f"**Question:** {s.question}", ""]
    out += [
        "## Headline numbers",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Runs | {f.runs} (backend: {f.backend}) |",
        f"| P(impact) | {100 * f.p_impact:.1f}% (95% CI {100 * lo:.1f} to {100 * hi:.1f}%) |",
        f"| P(detected before impact) | {100 * f.p_detect_before_impact:.1f}% |",
        f"| Mean blast radius | {f.mean_blast_radius:.2f} agents |",
        f"| Time to impact | {_tti(f)} |",
        "",
        "## Agent exposure",
        "",
        "| Agent | Kind | Defenses | Compromised | Caused impact |",
        "|---|---|---|---|---|",
    ]
    for a in s.agents:
        out.append(
            f"| {a.id} | {a.kind} | {', '.join(a.defenses) or 'none'} | "
            f"{100 * f.agent_compromise.get(a.id, 0):.1f}% | {100 * f.agent_impact.get(a.id, 0):.1f}% |"
        )
    if f.top_paths:
        out += ["", "## Most common kill chains", ""]
        for path, share in f.top_paths:
            out.append(f"- `{path}` ({100 * share:.1f}% of impactful runs)")
    if ranked:
        out += ["", "## Recommended hardening (single changes)", "", "| Intervention | P(impact) after | Reduction | Cost |", "|---|---|---|---|"]
        for r in ranked[:8]:
            out.append(f"| `{r.intervention}` | {100 * r.p_impact:.1f}% | {100 * r.delta:.1f} pts | {r.intervention.cost:g} |")
    out += ["", "## Agent graph", "", "```mermaid", mermaid(s, f), "```"]
    if sample is not None:
        out += ["", "## Sample episode", "", "```text", render_trace(s, sample), "```"]
    out += ["", "_Generated by [contagent](https://github.com/rashedhasan090/contagent). Numbers are model estimates, not guarantees._", ""]
    return "\n".join(out)
