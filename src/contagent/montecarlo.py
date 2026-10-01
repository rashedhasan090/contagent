"""Monte Carlo forecasting on top of the single-episode engine."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .backends import Backend, MockBackend
from .engine import RunResult, Simulation
from .scenario import Scenario


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _percentile(sorted_vals: list[int], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


@dataclass
class Forecast:
    scenario: str
    runs: int
    backend: str
    p_impact: float
    p_impact_ci: tuple[float, float]
    p_detect_before_impact: float
    mean_blast_radius: float
    tti_median: float | None
    tti_p90: float | None
    agent_compromise: dict[str, float]
    agent_impact: dict[str, float]
    edge_usage: dict[str, float]
    top_paths: list[tuple[str, float]]
    tool_impacts: dict[str, float]
    seeds: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "runs": self.runs,
            "backend": self.backend,
            "p_impact": round(self.p_impact, 4),
            "p_impact_ci95": [round(x, 4) for x in self.p_impact_ci],
            "p_detect_before_impact": round(self.p_detect_before_impact, 4),
            "mean_blast_radius": round(self.mean_blast_radius, 3),
            "time_to_impact": {"median": self.tti_median, "p90": self.tti_p90},
            "agent_compromise_rate": {k: round(v, 4) for k, v in self.agent_compromise.items()},
            "agent_impact_rate": {k: round(v, 4) for k, v in self.agent_impact.items()},
            "edge_infection_rate": {k: round(v, 4) for k, v in self.edge_usage.items()},
            "top_impact_paths": [{"path": p, "share": round(s, 4)} for p, s in self.top_paths],
            "tool_impact_rate": {k: round(v, 4) for k, v in self.tool_impacts.items()},
        }


def summarize(scenario: Scenario, results: list[RunResult], backend_name: str) -> Forecast:
    n = len(results)
    impacts = sum(r.impact for r in results)
    detected = sum(r.detected_before_impact for r in results)
    tti = sorted(r.time_to_impact for r in results if r.time_to_impact is not None)
    comp = Counter()
    imp_agents = Counter()
    edges = Counter()
    paths = Counter()
    tools = Counter()
    blast = 0
    for r in results:
        blast += len(r.compromised)
        comp.update(r.compromised)
        imp_agents.update({a for _, a, _ in r.impacts})
        tools.update({t for _, _, t in r.impacts})
        for dst, src in r.infected_by.items():
            edges[f"{src}->{dst}"] += 1
        if r.impact:
            paths[" -> ".join(r.impact_chain())] += 1
    agent_ids = [a.id for a in scenario.agents]
    return Forecast(
        scenario=scenario.name,
        runs=n,
        backend=backend_name,
        p_impact=impacts / n if n else 0.0,
        p_impact_ci=wilson_interval(impacts, n),
        p_detect_before_impact=detected / n if n else 0.0,
        mean_blast_radius=blast / n if n else 0.0,
        tti_median=_percentile(tti, 0.5),
        tti_p90=_percentile(tti, 0.9),
        agent_compromise={a: comp[a] / n for a in agent_ids},
        agent_impact={a: imp_agents[a] / n for a in agent_ids if imp_agents[a]},
        edge_usage={e: c / n for e, c in edges.most_common()},
        top_paths=[(p, c / impacts) for p, c in paths.most_common(5)] if impacts else [],
        tool_impacts={t: c / n for t, c in tools.most_common()},
        seeds=[r.seed for r in results],
    )


def simulate(scenario: Scenario, runs: int = 500, seed: int = 0, backend: Backend | None = None) -> list[RunResult]:
    backend = backend or MockBackend()
    sim = Simulation(scenario, backend)
    return [sim.run(seed + i) for i in range(runs)]


def forecast(scenario: Scenario, runs: int = 500, seed: int = 0, backend: Backend | None = None) -> Forecast:
    """Run ``runs`` episodes with seeds ``seed .. seed+runs-1`` and aggregate them.

    Using a fixed seed range gives common random numbers, so two forecasts of
    slightly different scenarios are directly comparable.
    """
    backend = backend or MockBackend()
    return summarize(scenario, simulate(scenario, runs, seed, backend), backend.name)
