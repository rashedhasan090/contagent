"""Discrete-time contagion engine.

One call to :meth:`Simulation.run` plays out a single stochastic episode:

1. Payloads enter the system at their injection points.
2. Each delivery is screened (sanitizer), then judged by the backend, which
   returns the chance that the receiving agent flags the message or complies
   with it.
3. Compromised agents relay the payload to downstream agents, sometimes as a
   paraphrased variant that is harder to filter. Poisoned stores (memory,
   vector DB, shared docs) keep serving the payload to every reader.
4. Compromised agents that hold a goal tool try to use it. A tool gate may
   stop the call and raise an alarm. A successful call is an *impact*.
5. Flags warn neighbouring agents and can trigger quarantine of compromised
   agents that run under monitoring.

The interesting part is the interaction between steps. For example, without
provenance tracking a payload relayed by a trusted teammate inherits that
teammate's trust ("trust laundering"), and paraphrased variants slip past
pattern filters that would have caught the original.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from .backends import Backend, MockBackend, Situation
from .scenario import Scenario

DEFAULT_DEFENSE_PARAMS: dict[str, dict[str, float]] = {
    "sanitizer": {"strength": 0.8},
    "spotlighting": {"strength": 0.6},
    "tool_gate": {"strength": 0.75},
    "quarantine": {"rate": 0.6},
    "provenance": {},
}

DEFAULT_DYNAMICS: dict[str, float] = {
    "relay_prob": 0.85,       # chance a compromised agent forwards on each out-edge, first round
    "persist_prob": 0.3,      # chance per later round it keeps forwarding (memory persistence)
    "act_prob": 0.7,          # chance per round a compromised agent tries its goal tool
    "store_read_prob": 0.6,   # chance per round a reader pulls from a poisoned store
    "warn_factor": 0.5,       # susceptibility multiplier for agents warned by a neighbour's flag
    "mutation_potency_jitter": 0.15,
    "mutation_detect_drop": 0.25,
}


class State(str, Enum):
    CLEAN = "clean"
    COMPROMISED = "compromised"
    ALERTED = "alerted"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class Variant:
    payload_id: str
    generation: int
    potency: float
    detectability: float
    origin_trust: float


@dataclass
class Delivery:
    dst: str
    variant: Variant
    sender: str
    channel: str
    trust: float
    hop: int


@dataclass
class Event:
    round: int
    kind: str  # inject, blocked, flag, comply, ignore, relay, impact, gate_block, quarantine, warn
    agent: str
    detail: str = ""
    src: str = ""

    def as_dict(self) -> dict:
        return {"round": self.round, "kind": self.kind, "agent": self.agent, "src": self.src, "detail": self.detail}


@dataclass
class RunResult:
    seed: int
    states: dict[str, State]
    infected_by: dict[str, str]         # agent -> sender that compromised it
    compromised_round: dict[str, int]
    impacts: list[tuple[int, str, str]]  # (round, agent, tool)
    first_alert_round: int | None
    events: list[Event] = field(default_factory=list)

    @property
    def compromised(self) -> set[str]:
        return set(self.compromised_round)

    @property
    def impact(self) -> bool:
        return bool(self.impacts)

    @property
    def time_to_impact(self) -> int | None:
        return self.impacts[0][0] if self.impacts else None

    @property
    def detected_before_impact(self) -> bool:
        if self.first_alert_round is None:
            return False
        if not self.impacts:
            return True
        return self.first_alert_round <= self.impacts[0][0]

    def chain_to(self, agent: str) -> list[str]:
        """Infection chain from the entry point to ``agent`` (inclusive)."""
        chain = [agent]
        seen = {agent}
        cur = agent
        while cur in self.infected_by:
            nxt = self.infected_by[cur]
            if nxt in seen or nxt == "external":
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        return list(reversed(chain))

    def impact_chain(self) -> list[str]:
        return self.chain_to(self.impacts[0][1]) if self.impacts else []


class Simulation:
    def __init__(self, scenario: Scenario, backend: Backend | None = None, trace: bool = False):
        self.s = scenario
        self.backend = backend or MockBackend()
        self.trace = trace
        self.dparams = {k: dict(v) for k, v in DEFAULT_DEFENSE_PARAMS.items()}
        for name, vals in scenario.defense_params.items():
            self.dparams.setdefault(name, {}).update(vals)
        self.dyn = dict(DEFAULT_DYNAMICS)
        self.dyn.update(scenario.dynamics)
        if isinstance(self.backend, MockBackend):
            self.backend.spotlight_strength = self.dparams["spotlighting"]["strength"]

    # ------------------------------------------------------------------ helpers
    def _log(self, events: list[Event], *args, **kw) -> None:
        if self.trace:
            events.append(Event(*args, **kw))

    def _mutate(self, v: Variant, rng: random.Random) -> Variant:
        p = self.s.payload(v.payload_id)
        if rng.random() >= p.mutation_rate:
            return v
        jitter = self.dyn["mutation_potency_jitter"]
        potency = min(0.98, max(0.05, v.potency * (1.0 + rng.uniform(-jitter, jitter / 2))))
        detect = max(0.02, v.detectability * (1.0 - self.dyn["mutation_detect_drop"]))
        return Variant(v.payload_id, v.generation + 1, potency, detect, v.origin_trust)

    # --------------------------------------------------------------------- run
    def run(self, seed: int = 0) -> RunResult:
        rng = random.Random(seed)
        s = self.s
        states = {a.id: State.CLEAN for a in s.agents}
        warned: set[str] = set()
        infected_by: dict[str, str] = {}
        comp_round: dict[str, int] = {}
        carried: dict[str, Variant] = {}
        hop_of: dict[str, int] = {}
        impacts: list[tuple[int, str, str]] = []
        first_alert: int | None = None
        events: list[Event] = []
        pending: list[Delivery] = []
        goal_tools = {p.id: set(p.goal_tools) for p in s.payloads}
        alarm_raised = False

        def raise_alert(r: int, agent_id: str, why: str) -> None:
            nonlocal first_alert, alarm_raised
            alarm_raised = True
            if first_alert is None:
                first_alert = r
            for n in sorted(s.neighbors(agent_id)):
                if states[n] == State.CLEAN and n not in warned:
                    warned.add(n)
                    self._log(events, r, "warn", n, f"warned by {agent_id}", src=agent_id)

        for r in range(s.max_rounds):
            # 1. new external injections this round
            for inj in s.injections:
                if inj.round == r:
                    p = s.payload(inj.payload)
                    v = Variant(p.id, 0, p.resolved_potency(), p.resolved_detectability(), inj.trust)
                    pending.append(Delivery(inj.agent, v, "external", inj.channel, inj.trust, 0))
                    self._log(events, r, "inject", inj.agent, f"{p.id} via {inj.channel}", src="external")

            # 2. process deliveries (shuffled so ordering does not bias outcomes)
            current, pending = pending, []
            rng.shuffle(current)
            for d in current:
                agent = s.agent(d.dst)
                if states[agent.id] != State.CLEAN:
                    continue
                v = d.variant
                if agent.has_defense("sanitizer"):
                    if rng.random() < self.dparams["sanitizer"]["strength"] * v.detectability:
                        self._log(events, r, "blocked", agent.id, f"sanitizer caught {v.payload_id} gen{v.generation}", src=d.sender)
                        continue
                if agent.kind == "store":
                    states[agent.id] = State.COMPROMISED
                    infected_by[agent.id] = d.sender
                    comp_round[agent.id] = r
                    carried[agent.id] = v
                    hop_of[agent.id] = d.hop
                    self._log(events, r, "comply", agent.id, f"store poisoned with {v.payload_id} gen{v.generation}", src=d.sender)
                    continue
                trust = d.trust
                if agent.has_defense("provenance"):
                    trust = min(trust, v.origin_trust)
                susceptibility = agent.susceptibility
                if agent.id in warned:
                    susceptibility *= self.dyn["warn_factor"]
                view = agent if susceptibility == agent.susceptibility else _with_susceptibility(agent, susceptibility)
                j = self.backend.judge(
                    Situation(
                        agent=view, payload=s.payload(v.payload_id), sender=d.sender, channel=d.channel,
                        trust=trust, hop=d.hop, generation=v.generation, potency=v.potency,
                        detectability=v.detectability, spotlight=agent.has_defense("spotlighting"),
                        warned=agent.id in warned,
                    )
                )
                roll = rng.random()
                if roll < j.p_flag:
                    states[agent.id] = State.ALERTED
                    self._log(events, r, "flag", agent.id, f"flagged {v.payload_id} gen{v.generation} (p_flag={j.p_flag:.2f})", src=d.sender)
                    raise_alert(r, agent.id, "flag")
                elif rng.random() < j.p_comply:
                    states[agent.id] = State.COMPROMISED
                    infected_by[agent.id] = d.sender
                    comp_round[agent.id] = r
                    carried[agent.id] = v
                    hop_of[agent.id] = d.hop
                    self._log(events, r, "comply", agent.id, f"followed {v.payload_id} gen{v.generation} (p={j.p_comply:.2f}, trust={trust:.2f})", src=d.sender)
                else:
                    self._log(events, r, "ignore", agent.id, f"ignored {v.payload_id} gen{v.generation} (p={j.p_comply:.2f})", src=d.sender)

            # 3. compromised agents act and spread
            for agent in s.agents:
                aid = agent.id
                if states[aid] != State.COMPROMISED:
                    continue
                if agent.has_defense("quarantine") and alarm_raised and rng.random() < self.dparams["quarantine"]["rate"]:
                    states[aid] = State.QUARANTINED
                    self._log(events, r, "quarantine", aid, "isolated by monitor after alert")
                    continue
                v = carried[aid]
                fresh = comp_round[aid] == r
                # goal tool use
                hits = goal_tools[v.payload_id].intersection(agent.tools)
                if hits and agent.kind != "store" and rng.random() < self.dyn["act_prob"]:
                    tool = sorted(hits)[0]
                    if agent.has_defense("tool_gate") and rng.random() < self.dparams["tool_gate"]["strength"]:
                        self._log(events, r, "gate_block", aid, f"tool gate stopped {tool}")
                        states[aid] = State.ALERTED
                        raise_alert(r, aid, "tool_gate")
                        continue
                    impacts.append((r, aid, tool))
                    self._log(events, r, "impact", aid, f"executed {tool} for {v.payload_id}")
                # propagation
                if agent.kind == "store":
                    prob = self.dyn["store_read_prob"]
                else:
                    prob = self.dyn["relay_prob"] if fresh else self.dyn["persist_prob"]
                for e in s.out_edges(aid):
                    if states[e.dst] != State.CLEAN or rng.random() >= prob:
                        continue
                    nv = v if agent.kind == "store" else self._mutate(v, rng)
                    pending.append(Delivery(e.dst, nv, aid, e.label or f"{aid}->{e.dst}", e.trust, hop_of[aid] + 1))
                    self._log(events, r, "relay", e.dst, f"{nv.payload_id} gen{nv.generation} over {e.label or 'edge'}", src=aid)

            if not pending and not any(inj.round > r for inj in s.injections):
                if not any(st == State.COMPROMISED for st in states.values()):
                    break

        return RunResult(seed, states, infected_by, comp_round, impacts, first_alert, events)


def _with_susceptibility(agent, value: float):
    from dataclasses import replace

    return replace(agent, susceptibility=value)
