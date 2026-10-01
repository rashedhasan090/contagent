"""Scenario model and loader.

A scenario describes a multi-agent system as a directed graph of agents
(nodes) and message channels (edges), plus one or more adversarial payloads
that get injected at entry points. Scenarios are written in TOML so they can
be loaded with the standard library only.
"""

from __future__ import annotations

import copy
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_KINDS = ("llm", "store", "human")
KNOWN_DEFENSES = ("sanitizer", "spotlighting", "provenance", "tool_gate", "quarantine")

# Phrases that commonly show up in injection payloads. Used both to estimate
# how "pushy" a payload is and by the sanitizer defense to detect it.
INJECTION_MARKERS = (
    r"ignore (all |any )?(previous|prior|above) (instructions|messages)",
    r"disregard (your|the) (instructions|rules|system prompt)",
    r"you are now",
    r"system (override|prompt|message)",
    r"\bdo not (tell|inform|alert)\b",
    r"\burgent(ly)?\b",
    r"\bimmediately\b",
    r"\bsecret(ly)?\b",
    r"\bforward (this|the following)\b",
    r"\bexfiltrat",
    r"\bsend (all|the) .*(to|@)",
    r"\badmin(istrator)? (mode|access)\b",
)


class ScenarioError(ValueError):
    """Raised when a scenario file is malformed."""


def marker_hits(text: str) -> int:
    """Count how many known injection markers appear in ``text``."""
    lowered = text.lower()
    return sum(1 for pattern in INJECTION_MARKERS if re.search(pattern, lowered))


def estimate_potency(text: str) -> float:
    """Heuristic persuasiveness score in [0.35, 0.95] derived from payload text."""
    return min(0.95, 0.35 + 0.1 * marker_hits(text))


def estimate_detectability(text: str) -> float:
    """Heuristic chance in [0.15, 0.9] that a pattern-based filter spots the payload."""
    return min(0.9, 0.15 + 0.12 * marker_hits(text))


@dataclass
class Agent:
    id: str
    role: str = ""
    kind: str = "llm"
    persona: str = ""
    tools: list[str] = field(default_factory=list)
    susceptibility: float = 0.5
    detect_skill: float = 0.2
    defenses: list[str] = field(default_factory=list)

    def has_defense(self, name: str) -> bool:
        return name in self.defenses


@dataclass
class Edge:
    src: str
    dst: str
    trust: float = 0.7
    label: str = ""


@dataclass
class Payload:
    id: str
    text: str
    goal_tools: list[str]
    potency: float | None = None
    detectability: float | None = None
    mutation_rate: float = 0.2
    technique: str = "indirect prompt injection"

    def resolved_potency(self) -> float:
        return self.potency if self.potency is not None else estimate_potency(self.text)

    def resolved_detectability(self) -> float:
        if self.detectability is not None:
            return self.detectability
        return estimate_detectability(self.text)


@dataclass
class Injection:
    payload: str
    agent: str
    round: int = 0
    trust: float = 0.5
    channel: str = "external"


@dataclass
class Scenario:
    name: str
    description: str = ""
    max_rounds: int = 12
    agents: list[Agent] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    payloads: list[Payload] = field(default_factory=list)
    injections: list[Injection] = field(default_factory=list)
    defense_params: dict[str, dict[str, float]] = field(default_factory=dict)
    dynamics: dict[str, float] = field(default_factory=dict)
    question: str = ""

    # ------------------------------------------------------------------ lookup
    def agent(self, agent_id: str) -> Agent:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise KeyError(agent_id)

    def payload(self, payload_id: str) -> Payload:
        for p in self.payloads:
            if p.id == payload_id:
                return p
        raise KeyError(payload_id)

    def out_edges(self, agent_id: str) -> list[Edge]:
        return [e for e in self.edges if e.src == agent_id]

    def neighbors(self, agent_id: str) -> set[str]:
        out = {e.dst for e in self.edges if e.src == agent_id}
        inc = {e.src for e in self.edges if e.dst == agent_id}
        return out | inc

    def edge(self, src: str, dst: str) -> Edge | None:
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e
        return None

    def copy(self) -> "Scenario":
        return copy.deepcopy(self)

    def high_value_agents(self) -> list[str]:
        goal = {t for p in self.payloads for t in p.goal_tools}
        return [a.id for a in self.agents if goal.intersection(a.tools)]

    # -------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty if valid)."""
        problems: list[str] = []
        ids = [a.id for a in self.agents]
        if not ids:
            problems.append("scenario has no agents")
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            problems.append(f"duplicate agent ids: {sorted(dupes)}")
        idset = set(ids)
        for a in self.agents:
            if a.kind not in AGENT_KINDS:
                problems.append(f"agent {a.id}: unknown kind {a.kind!r} (use one of {AGENT_KINDS})")
            for name, val in (("susceptibility", a.susceptibility), ("detect_skill", a.detect_skill)):
                if not 0.0 <= val <= 1.0:
                    problems.append(f"agent {a.id}: {name} must be in [0, 1], got {val}")
            for d in a.defenses:
                if d not in KNOWN_DEFENSES:
                    problems.append(f"agent {a.id}: unknown defense {d!r} (known: {KNOWN_DEFENSES})")
        for e in self.edges:
            if e.src not in idset or e.dst not in idset:
                problems.append(f"edge {e.src}->{e.dst} references an unknown agent")
            if not 0.0 <= e.trust <= 1.0:
                problems.append(f"edge {e.src}->{e.dst}: trust must be in [0, 1]")
        pids = {p.id for p in self.payloads}
        if not self.payloads:
            problems.append("scenario has no payloads")
        for p in self.payloads:
            if not p.goal_tools:
                problems.append(f"payload {p.id}: goal_tools must not be empty")
        if not self.injections:
            problems.append("scenario has no injections")
        for inj in self.injections:
            if inj.payload not in pids:
                problems.append(f"injection references unknown payload {inj.payload!r}")
            if inj.agent not in idset:
                problems.append(f"injection references unknown agent {inj.agent!r}")
            if inj.round < 0 or inj.round >= self.max_rounds:
                problems.append(f"injection round {inj.round} outside [0, {self.max_rounds})")
        if self.max_rounds < 1:
            problems.append("max_rounds must be >= 1")
        return problems


# ---------------------------------------------------------------------- loading

def _build(cls, raw: dict[str, Any], where: str):
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ScenarioError(f"{where}: unknown field(s) {sorted(unknown)}")
    try:
        return cls(**raw)
    except TypeError as exc:  # missing required field
        raise ScenarioError(f"{where}: {exc}") from exc


def scenario_from_dict(data: dict[str, Any], validate: bool = True) -> Scenario:
    data = copy.deepcopy(data)
    meta = data.pop("scenario", {})
    if "name" not in meta:
        raise ScenarioError("[scenario] table needs a 'name'")
    agents = [_build(Agent, a, f"agent #{i}") for i, a in enumerate(data.pop("agents", []))]
    edges: list[Edge] = []
    for i, e in enumerate(data.pop("edges", [])):
        if "bidirectional" in e:
            bidi = e.pop("bidirectional")
        else:
            bidi = False
        edge = _build(Edge, e, f"edge #{i}")
        edges.append(edge)
        if bidi:
            edges.append(Edge(src=edge.dst, dst=edge.src, trust=edge.trust, label=edge.label))
    payloads = [_build(Payload, p, f"payload #{i}") for i, p in enumerate(data.pop("payloads", []))]
    injections = [_build(Injection, j, f"injection #{i}") for i, j in enumerate(data.pop("injections", []))]
    defense_params = data.pop("defense_params", {})
    dynamics = data.pop("dynamics", {})
    if data:
        raise ScenarioError(f"unknown top-level table(s): {sorted(data)}")
    scen = Scenario(
        name=meta["name"],
        description=meta.get("description", ""),
        max_rounds=int(meta.get("max_rounds", 12)),
        question=meta.get("question", ""),
        agents=agents,
        edges=edges,
        payloads=payloads,
        injections=injections,
        defense_params=defense_params,
        dynamics=dynamics,
    )
    if validate:
        problems = scen.validate()
        if problems:
            raise ScenarioError("invalid scenario:\n  - " + "\n  - ".join(problems))
    return scen


def load_scenario(path: str | Path, validate: bool = True) -> Scenario:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ScenarioError(f"{path}: invalid TOML: {exc}") from exc
    return scenario_from_dict(data, validate=validate)
