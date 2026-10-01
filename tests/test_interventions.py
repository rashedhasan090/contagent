import pytest

from contagent import apply_intervention, forecast, parse_intervention, rank_interventions
from contagent.interventions import plan
from contagent.scenario import ScenarioError


def test_parse_roundtrip():
    for text in ["add:tool_gate@ops", "remove:sanitizer@*", "cut:triage->ops", "trust:kb->ops=0.2", "tools:ops-=send_email"]:
        assert str(parse_intervention(text)) == text


@pytest.mark.parametrize("bad", ["add:magic@ops", "explode:ops", "cut:a-b", "trust:a->b=x"])
def test_parse_rejects(bad):
    with pytest.raises(ScenarioError):
        parse_intervention(bad)


def test_apply_does_not_mutate_original(support_desk):
    before = list(support_desk.agent("ops").defenses)
    changed = apply_intervention(support_desk, "add:tool_gate@ops")
    assert "tool_gate" in changed.agent("ops").defenses
    assert support_desk.agent("ops").defenses == before


def test_wildcard_only_hits_llm_agents(support_desk):
    changed = apply_intervention(support_desk, "add:spotlighting@*")
    assert "spotlighting" not in changed.agent("kb").defenses
    assert "spotlighting" not in changed.agent("lead").defenses
    assert "spotlighting" in changed.agent("writer").defenses


def test_cut_and_tools(support_desk):
    s = apply_intervention(support_desk, "cut:triage->ops")
    assert s.edge("triage", "ops") is None
    s = apply_intervention(support_desk, "tools:ops-=export_crm")
    assert "export_crm" not in s.agent("ops").tools
    with pytest.raises(ScenarioError):
        apply_intervention(support_desk, "cut:lead->triage")


def test_removing_all_goal_tools_eliminates_impact(support_desk):
    s = apply_intervention(support_desk, "tools:ops-=export_crm")
    s = apply_intervention(s, "tools:ops-=send_email")
    assert forecast(s, runs=200).p_impact == 0.0


def test_rank_is_sorted_and_helpful(support_desk):
    base, ranked = rank_interventions(support_desk, runs=200)
    assert ranked, "planner should find candidates"
    deltas = [r.delta for r in ranked]
    assert deltas == sorted(deltas, reverse=True)
    assert ranked[0].delta > 0


def test_plan_respects_budget(support_desk):
    base, steps = plan(support_desk, budget=3.0, runs=150)
    assert sum(iv.cost for iv, _ in steps) <= 3.0
    if steps:
        assert steps[-1][1].p_impact <= base.p_impact
