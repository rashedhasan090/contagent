import json

import pytest

from contagent import Simulation
from contagent.backends import MockBackend, OpenAIBackend, Situation, build_messages, make_backend, parse_action


def situation(agent, payload, **kw):
    base = dict(
        agent=agent, payload=payload, sender="external", channel="email", trust=0.3, hop=0,
        generation=0, potency=payload.resolved_potency(), detectability=payload.resolved_detectability(), spotlight=False,
    )
    base.update(kw)
    return Situation(**base)


def test_mock_monotone_in_trust(support_desk):
    m = MockBackend()
    a, p = support_desk.agent("ops"), support_desk.payloads[0]
    lo = m.judge(situation(a, p, trust=0.1)).p_comply
    hi = m.judge(situation(a, p, trust=0.9)).p_comply
    assert hi > lo


def test_mock_spotlight_weaker_on_relays(support_desk):
    m = MockBackend()
    a, p = support_desk.agent("ops"), support_desk.payloads[0]
    plain = m.judge(situation(a, p)).p_comply
    direct = m.judge(situation(a, p, spotlight=True, hop=0)).p_comply
    relayed = m.judge(situation(a, p, spotlight=True, hop=2)).p_comply
    assert direct < relayed < plain


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"action": "comply", "reason": "x"}', "comply"),
        ('Sure! ```json\n{"action":"FLAG","reason":"odd"}\n```', "flag"),
        ('blah "action": "ignore" blah', "ignore"),
        ("I refuse.", "ignore"),
    ],
)
def test_parse_action(text, expected):
    assert parse_action(text) == expected


def test_build_messages_spotlight(support_desk):
    a, p = support_desk.agent("ops"), support_desk.payloads[0]
    msgs = build_messages(situation(a, p, spotlight=True))
    assert "<<DATA>>" in msgs[1]["content"]
    assert "^" in msgs[1]["content"]
    assert "export_crm" in msgs[0]["content"]


def fake_transport(action_cycle):
    calls = []

    def transport(url, headers, body):
        calls.append((url, headers, body))
        action = action_cycle[(len(calls) - 1) % len(action_cycle)]
        return {"choices": [{"message": {"content": json.dumps({"action": action, "reason": "test"})}}]}

    transport.calls = calls
    return transport


def test_openai_backend_with_fake_transport(support_desk):
    t = fake_transport(["comply", "comply", "flag"])
    b = OpenAIBackend(model="test-model", base_url="http://llm.local/v1", api_key="k", samples=3, transport=t)
    a, p = support_desk.agent("ops"), support_desk.payloads[0]
    j = b.judge(situation(a, p))
    assert len(t.calls) == 3
    assert t.calls[0][0] == "http://llm.local/v1/chat/completions"
    assert t.calls[0][2]["model"] == "test-model"
    assert j.p_flag == pytest.approx(2 / 6)
    assert j.p_comply == pytest.approx(3 / 4)
    b.judge(situation(a, p))
    assert len(t.calls) == 3, "identical situation must hit the cache"


def test_openai_backend_drives_simulation(support_desk):
    t = fake_transport(["ignore"])
    b = OpenAIBackend(api_key="k", samples=2, transport=t)
    r = Simulation(support_desk, b).run(0)
    assert len(t.calls) >= 2
    assert b.calls == len(t.calls)
    assert r.seed == 0


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("CONTAGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        make_backend("openai")


def test_env_config(monkeypatch):
    monkeypatch.setenv("CONTAGENT_LLM_API_KEY", "abc")
    monkeypatch.setenv("CONTAGENT_LLM_BASE_URL", "http://x/v1/")
    monkeypatch.setenv("CONTAGENT_LLM_MODEL", "m")
    b = make_backend("openai")
    assert (b.base_url, b.model, b.api_key) == ("http://x/v1", "m", "abc")


def test_warned_changes_prompt_and_cache_key(support_desk):
    t = fake_transport(["ignore"])
    b = OpenAIBackend(api_key="k", samples=1, transport=t)
    a, p = support_desk.agent("ops"), support_desk.payloads[0]
    b.judge(situation(a, p))
    b.judge(situation(a, p, warned=True))
    assert len(t.calls) == 2
    assert "suspicious message" in t.calls[1][2]["messages"][0]["content"]
