import json

from contagent.cli import main

from conftest import EXAMPLES

SD = str(EXAMPLES / "support_desk.toml")


def test_validate(capsys):
    assert main(["validate", SD]) == 0
    assert "ok:" in capsys.readouterr().out


def test_run_text_and_json(capsys):
    assert main(["run", SD, "-n", "100"]) == 0
    assert "P(impact)" in capsys.readouterr().out
    assert main(["run", SD, "-n", "100", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runs"] == 100


def test_what_if(capsys):
    assert main(["run", SD, "-n", "100", "-w", "add:tool_gate@ops", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["result"]["p_impact"] <= data["baseline"]["p_impact"]


def test_trace_find_impact(capsys):
    assert main(["trace", SD, "--find", "impact"]) == 0
    assert "IMPACT" in capsys.readouterr().out


def test_rank_plan_graph(capsys):
    assert main(["rank", SD, "-n", "80", "--top", "3"]) == 0
    assert "intervention" in capsys.readouterr().out
    assert main(["plan", SD, "-n", "80", "--budget", "2"]) == 0
    assert "Greedy" in capsys.readouterr().out
    assert main(["graph", SD, "--plain"]) == 0
    assert capsys.readouterr().out.startswith("flowchart LR")


def test_report_file(tmp_path, capsys):
    out = tmp_path / "r.md"
    assert main(["report", SD, "-n", "80", "-o", str(out)]) == 0
    text = out.read_text()
    assert text.startswith("# contagent risk report")
    assert "```mermaid" in text


def test_bad_inputs(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[scenario]\nname='x'\n")
    assert main(["validate", str(bad)]) == 2
    assert main(["run", SD, "-n", "10", "-w", "nonsense"]) == 2
    assert main(["validate", str(tmp_path / "missing.toml")]) == 2
