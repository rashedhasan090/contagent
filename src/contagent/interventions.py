"""What-if interventions and an automatic defense planner.

Intervention syntax (also used on the CLI with ``--what-if``):

  add:<defense>@<agent|*>      give an agent (or every LLM agent) a defense
  remove:<defense>@<agent|*>   take a defense away
  cut:<src>-><dst>             delete a channel
  trust:<src>-><dst>=<value>   change how much <dst> trusts messages from <src>
  tools:<agent>-=<tool>        revoke a tool from an agent (least privilege)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .backends import Backend, MockBackend
from .montecarlo import Forecast, forecast
from .scenario import KNOWN_DEFENSES, Scenario, ScenarioError

DEFENSE_COST = {"sanitizer": 1.0, "spotlighting": 1.0, "provenance": 2.0, "tool_gate": 1.5, "quarantine": 2.0}
CUT_COST = 3.0
TOOL_REVOKE_COST = 2.5


@dataclass(frozen=True)
class Intervention:
    op: str
    target: str
    arg: str = ""
    value: float | None = None

    def __str__(self) -> str:
        if self.op in ("add", "remove"):
            return f"{self.op}:{self.arg}@{self.target}"
        if self.op == "cut":
            return f"cut:{self.target}"
        if self.op == "trust":
            return f"trust:{self.target}={self.value}"
        if self.op == "tools":
            return f"tools:{self.target}-={self.arg}"
        return f"{self.op}:{self.target}"

    @property
    def cost(self) -> float:
        if self.op == "add":
            return DEFENSE_COST.get(self.arg, 1.0)
        if self.op == "cut":
            return CUT_COST
        if self.op == "tools":
            return TOOL_REVOKE_COST
        return 1.0


_PATTERNS = (
    (re.compile(r"^(add|remove):(\w+)@([\w*.-]+)$"), "defense"),
    (re.compile(r"^cut:([\w.-]+)->([\w.-]+)$"), "cut"),
    (re.compile(r"^trust:([\w.-]+)->([\w.-]+)=([0-9.]+)$"), "trust"),
    (re.compile(r"^tools:([\w.-]+)-=([\w.-]+)$"), "tools"),
)


def parse_intervention(text: str) -> Intervention:
    text = text.strip()
    for pattern, kind in _PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        if kind == "defense":
            op, defense, target = m.groups()
            if defense not in KNOWN_DEFENSES:
                raise ScenarioError(f"unknown defense {defense!r} in {text!r}")
            return Intervention(op, target, defense)
        if kind == "cut":
            return Intervention("cut", f"{m.group(1)}->{m.group(2)}")
        if kind == "trust":
            return Intervention("trust", f"{m.group(1)}->{m.group(2)}", value=float(m.group(3)))
        if kind == "tools":
            return Intervention("tools", m.group(1), m.group(2))
    raise ScenarioError(f"cannot parse intervention {text!r}; see `contagent --help`")


def apply_intervention(scenario: Scenario, iv: Intervention | str) -> Scenario:
    """Return a modified copy of ``scenario``. The original is untouched."""
    if isinstance(iv, str):
        iv = parse_intervention(iv)
    s = scenario.copy()
    if iv.op in ("add", "remove"):
        targets = [a for a in s.agents if a.kind == "llm"] if iv.target == "*" else [s.agent(iv.target)]
        for a in targets:
            if iv.op == "add" and iv.arg not in a.defenses:
                a.defenses.append(iv.arg)
            elif iv.op == "remove" and iv.arg in a.defenses:
                a.defenses.remove(iv.arg)
    elif iv.op == "cut":
        src, dst = iv.target.split("->")
        before = len(s.edges)
        s.edges = [e for e in s.edges if not (e.src == src and e.dst == dst)]
        if len(s.edges) == before:
            raise ScenarioError(f"no edge {iv.target} to cut")
    elif iv.op == "trust":
        src, dst = iv.target.split("->")
        e = s.edge(src, dst)
        if e is None:
            raise ScenarioError(f"no edge {iv.target}")
        e.trust = float(iv.value)
    elif iv.op == "tools":
        a = s.agent(iv.target)
        if iv.arg not in a.tools:
            raise ScenarioError(f"agent {a.id} has no tool {iv.arg!r}")
        a.tools.remove(iv.arg)
    return s


def candidate_interventions(scenario: Scenario, base: Forecast, max_edges: int = 6) -> list[Intervention]:
    """Generate plausible single-step hardening moves for the planner."""
    out: list[Intervention] = []
    goal = {t for p in scenario.payloads for t in p.goal_tools}
    for a in scenario.agents:
        if a.kind != "llm":
            continue
        if base.agent_compromise.get(a.id, 0) == 0 and not goal.intersection(a.tools):
            continue
        for d in KNOWN_DEFENSES:
            if d == "tool_gate" and not goal.intersection(a.tools):
                continue
            if d not in a.defenses:
                out.append(Intervention("add", a.id, d))
        for t in sorted(goal.intersection(a.tools)):
            out.append(Intervention("tools", a.id, t))
    hot_edges = [e for e in base.edge_usage if not e.startswith("external->")][:max_edges]
    for e in hot_edges:
        src, dst = e.split("->")
        if scenario.edge(src, dst) is not None:
            out.append(Intervention("cut", e))
    return out


@dataclass
class Ranked:
    intervention: Intervention
    p_impact: float
    delta: float
    blast: float

    @property
    def efficiency(self) -> float:
        return self.delta / self.intervention.cost


def rank_interventions(
    scenario: Scenario,
    runs: int = 300,
    seed: int = 0,
    backend: Backend | None = None,
    candidates: list[Intervention] | None = None,
    base: Forecast | None = None,
) -> tuple[Forecast, list[Ranked]]:
    """Score every candidate with common random numbers and sort by risk reduction."""
    backend = backend or MockBackend()
    base = base or forecast(scenario, runs, seed, backend)
    candidates = candidates if candidates is not None else candidate_interventions(scenario, base)
    ranked = []
    for iv in candidates:
        f = forecast(apply_intervention(scenario, iv), runs, seed, backend)
        ranked.append(Ranked(iv, f.p_impact, base.p_impact - f.p_impact, f.mean_blast_radius))
    ranked.sort(key=lambda r: (-r.delta, r.intervention.cost, str(r.intervention)))
    return base, ranked


def plan(
    scenario: Scenario,
    budget: float = 4.0,
    runs: int = 300,
    seed: int = 0,
    backend: Backend | None = None,
) -> tuple[Forecast, list[tuple[Intervention, Forecast]]]:
    """Greedy hardening plan: repeatedly add the most cost-efficient move until the budget runs out."""
    backend = backend or MockBackend()
    base = forecast(scenario, runs, seed, backend)
    current, current_f = scenario, base
    steps: list[tuple[Intervention, Forecast]] = []
    spent = 0.0
    while True:
        cands = [c for c in candidate_interventions(current, current_f) if spent + c.cost <= budget]
        if not cands or current_f.p_impact == 0:
            break
        _, ranked = rank_interventions(current, runs, seed, backend, cands, base=current_f)
        best = max(ranked, key=lambda r: (r.efficiency, -r.intervention.cost))
        if best.delta <= 0:
            break
        current = apply_intervention(current, best.intervention)
        current_f = forecast(current, runs, seed, backend)
        spent += best.intervention.cost
        steps.append((best.intervention, current_f))
    return base, steps
