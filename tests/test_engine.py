from contagent import Simulation, forecast, scenario_from_dict
from contagent.backends import Judgement
from contagent.engine import State

from conftest import chain_dict


class FixedBackend:
    name = "fixed"

    def __init__(self, comply=1.0, flag=0.0):
        self.j = Judgement(comply, flag)

    def judge(self, s):
        return self.j


def test_determinism(support_desk):
    a = forecast(support_desk, runs=200, seed=3)
    b = forecast(support_desk, runs=200, seed=3)
    assert a.to_dict() == b.to_dict()


def test_always_comply_reaches_impact(chain):
    sim = Simulation(chain, FixedBackend(comply=1.0), trace=True)
    r = sim.run(0)
    assert r.impact
    assert r.impact_chain() == ["a", "b", "c"]
    assert r.compromised == {"a", "b", "c"}


def test_never_comply_means_no_impact(chain):
    f = forecast(chain, runs=100, backend=FixedBackend(comply=0.0))
    assert f.p_impact == 0.0
    assert f.mean_blast_radius == 0.0


def test_flag_alerts_and_warns(chain):
    sim = Simulation(chain, FixedBackend(comply=0.0, flag=1.0), trace=True)
    r = sim.run(0)
    assert r.states["a"] == State.ALERTED
    assert r.first_alert_round == 0
    assert any(e.kind == "warn" and e.agent == "b" for e in r.events)
    assert r.detected_before_impact


def test_no_goal_tool_means_no_impact():
    s = scenario_from_dict(chain_dict(tools_c=()))
    f = forecast(s, runs=200)
    assert f.p_impact == 0.0


def test_store_always_poisoned_and_relays():
    d = chain_dict()
    d["agents"][1] = {"id": "b", "kind": "store"}
    s = scenario_from_dict(d)
    r = Simulation(s, FixedBackend(comply=1.0)).run(0)
    assert "b" in r.compromised
    assert r.impact


def test_tool_gate_reduces_impact():
    open_ = scenario_from_dict(chain_dict())
    d = chain_dict()
    d["agents"][2]["defenses"] = ["tool_gate"]
    gated = scenario_from_dict(d)
    f0 = forecast(open_, runs=400, backend=FixedBackend(comply=1.0))
    f1 = forecast(gated, runs=400, backend=FixedBackend(comply=1.0))
    assert f0.p_impact > 0.9  # relays are probabilistic, so a few chains stall
    assert f1.p_impact < 0.6


def test_provenance_blocks_trust_laundering():
    """Low-trust external input relayed over a high-trust channel should not regain trust."""
    base = chain_dict(susceptibility=0.9, trust=1.0)
    base["injections"][0]["trust"] = 0.0
    laundered = scenario_from_dict(base)
    for a in base["agents"]:
        a["defenses"] = ["provenance"]
    tracked = scenario_from_dict(base)
    f0 = forecast(laundered, runs=800)
    f1 = forecast(tracked, runs=800)
    assert f1.p_impact < f0.p_impact


def test_mutation_lowers_detectability(chain):
    chain.payloads[0].mutation_rate = 1.0
    sim = Simulation(chain, FixedBackend(comply=1.0), trace=True)
    r = sim.run(1)
    gens = [e.detail for e in r.events if e.kind == "comply"]
    assert any("gen2" in g for g in gens)


def test_forecast_shapes(support_desk):
    f = forecast(support_desk, runs=300)
    assert 0 <= f.p_impact <= 1
    lo, hi = f.p_impact_ci
    assert lo <= f.p_impact <= hi
    assert set(f.agent_compromise) == {a.id for a in support_desk.agents}
    if f.top_paths:
        assert abs(sum(s for _, s in f.top_paths) - 1.0) < 1e-9 or len(f.top_paths) == 5
    d = f.to_dict()
    assert d["runs"] == 300 and "edge_infection_rate" in d
