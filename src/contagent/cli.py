"""Command line interface: ``contagent <command> scenario.toml [options]``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .backends import make_backend
from .engine import Simulation
from .interventions import apply_intervention, parse_intervention, plan, rank_interventions
from .montecarlo import forecast
from .report import (
    markdown_report,
    mermaid,
    render_comparison,
    render_forecast,
    render_plan,
    render_ranking,
    render_trace,
)
from .scenario import ScenarioError, load_scenario

INTERVENTION_HELP = """intervention syntax:
  add:<defense>@<agent|*>      defenses: sanitizer, spotlighting, provenance, tool_gate, quarantine
  remove:<defense>@<agent|*>
  cut:<src>-><dst>
  trust:<src>-><dst>=<0..1>
  tools:<agent>-=<tool>
"""


def _backend(args):
    name = args.backend or os.environ.get("CONTAGENT_BACKEND", "mock")
    if name == "mock":
        return make_backend("mock")
    return make_backend(name, samples=args.samples)


def _common(p: argparse.ArgumentParser, runs: int = 500) -> None:
    p.add_argument("scenario", type=Path, help="scenario TOML file")
    p.add_argument("-n", "--runs", type=int, default=runs, help=f"Monte Carlo runs (default {runs})")
    p.add_argument("--seed", type=int, default=0, help="base random seed (default 0)")
    p.add_argument("--backend", choices=["mock", "openai"], default=None,
                   help="decision backend (default: $CONTAGENT_BACKEND or mock)")
    p.add_argument("--samples", type=int, default=3, help="LLM samples per unique situation (openai backend)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contagent",
        description="Forecast how prompt injections spread through multi-agent LLM systems.",
        epilog=INTERVENTION_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"contagent {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check a scenario file")
    p.add_argument("scenario", type=Path)

    p = sub.add_parser("run", help="Monte Carlo forecast, optionally with what-if changes",
                       epilog=INTERVENTION_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common(p)
    p.add_argument("-w", "--what-if", action="append", default=[], metavar="CHANGE",
                   help="apply an intervention and compare against the baseline (repeatable; applied together)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")

    p = sub.add_parser("trace", help="play one episode and print the event log")
    _common(p, runs=1)
    p.add_argument("--find", choices=["impact", "safe", "detected"], default=None,
                   help="search seeds for an episode with this outcome")

    p = sub.add_parser("rank", help="score single-change defenses by risk reduction")
    _common(p, runs=300)
    p.add_argument("--top", type=int, default=10)

    p = sub.add_parser("plan", help="greedy hardening plan under a cost budget")
    _common(p, runs=300)
    p.add_argument("--budget", type=float, default=4.0)

    p = sub.add_parser("report", help="write a Markdown risk report")
    _common(p, runs=500)
    p.add_argument("-o", "--output", type=Path, default=None, help="output file (default stdout)")

    p = sub.add_parser("graph", help="print a Mermaid diagram of the agent graph")
    _common(p, runs=300)
    p.add_argument("--plain", action="store_true", help="do not run a forecast for shading")
    return parser


def _find_episode(sim: Simulation, start: int, want: str | None, limit: int = 5000):
    for seed in range(start, start + limit):
        r = sim.run(seed)
        if want is None:
            return r
        if want == "impact" and r.impact:
            return r
        if want == "safe" and not r.impact and not r.compromised - {i.agent for i in sim.s.injections}:
            return r
        if want == "detected" and r.detected_before_impact and r.compromised:
            return r
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
    except (ScenarioError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "validate":
        print(f"ok: '{scenario.name}' has {len(scenario.agents)} agents, {len(scenario.edges)} channels, "
              f"{len(scenario.payloads)} payload(s), {len(scenario.injections)} injection(s)")
        hv = scenario.high_value_agents()
        print(f"agents holding goal tools: {', '.join(hv) if hv else 'none (no impact possible)'}")
        return 0

    try:
        backend = _backend(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.cmd == "run":
            base = forecast(scenario, args.runs, args.seed, backend)
            if not args.what_if:
                print(json.dumps(base.to_dict(), indent=2) if args.json else render_forecast(scenario, base))
                return 0
            changed = scenario
            for w in args.what_if:
                changed = apply_intervention(changed, parse_intervention(w))
            other = forecast(changed, args.runs, args.seed, backend)
            if args.json:
                print(json.dumps({"baseline": base.to_dict(), "what_if": args.what_if, "result": other.to_dict()}, indent=2))
            else:
                print(render_forecast(scenario, base))
                print()
                print(render_comparison(base, other, " + ".join(args.what_if)))
            return 0

        if args.cmd == "trace":
            sim = Simulation(scenario, backend, trace=True)
            r = _find_episode(sim, args.seed, args.find)
            if r is None:
                print(f"no episode with outcome '{args.find}' found", file=sys.stderr)
                return 1
            print(render_trace(scenario, r))
            return 0

        if args.cmd == "rank":
            base, ranked = rank_interventions(scenario, args.runs, args.seed, backend)
            print(render_ranking(base, ranked, args.top))
            return 0

        if args.cmd == "plan":
            base, steps = plan(scenario, args.budget, args.runs, args.seed, backend)
            print(render_plan(base, steps, args.budget))
            return 0

        if args.cmd == "report":
            base, ranked = rank_interventions(scenario, args.runs, args.seed, backend)
            sim = Simulation(scenario, backend, trace=True)
            sample = _find_episode(sim, args.seed, "impact", limit=500) or sim.run(args.seed)
            text = markdown_report(scenario, base, ranked, sample)
            if args.output:
                args.output.write_text(text, encoding="utf-8")
                print(f"wrote {args.output}")
            else:
                print(text)
            return 0

        if args.cmd == "graph":
            f = None if args.plain else forecast(scenario, args.runs, args.seed, backend)
            print(mermaid(scenario, f))
            return 0
    except (ScenarioError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
