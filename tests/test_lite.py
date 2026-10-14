"""eda-lite: prompt rendering, the cells contract, grading, and one run end to end."""

import json
import shutil

import pytest
from conftest import FIXTURE_PROJECT, cells_response

from eda_lite import prompts, run
from eda_lite.config import ConfigError, load_project
from eda_lite.grade import CHECKS, METRICS, RUN_INFO, TRACE, grade, load_answers, matches
from eda_lite.model import ModelError, parse_cells
from eda_lite.patterns import PATTERNS
from eda_lite.report import compare


@pytest.fixture
def project(tmp_path):
    shutil.copytree(FIXTURE_PROJECT, tmp_path / "projects" / "tiny_project")
    return load_project("tiny_project", root=tmp_path / "projects")


def test_load_project_is_strict(project, tmp_path):
    with pytest.raises(ConfigError, match="known: tiny_project"):
        load_project("nope", root=tmp_path / "projects")
    shutil.copytree(project.dir, tmp_path / "projects" / "renamed")
    with pytest.raises(ConfigError, match="must match its folder name"):
        load_project("renamed", root=tmp_path / "projects")


def test_prompt_fills_project_fields_and_leaves_other_braces(project):
    text = prompts.render("zero_shot", project)
    assert project.config.research_question in text
    assert "## Required claims\n- `n_rows`: Number of rows" in text
    assert 'df = pd.read_csv("data/tiny_dataset.csv")' in text
    assert '{"cells": [{"type"' in text  # the JSON example is not a placeholder
    assert "expected" not in text.lower()  # nothing from the answer key


def test_fill_does_not_rescan_inserted_text():
    assert prompts.fill("{a} {b}", a="{b}", b="x") == "{b} x"


def test_parse_cells_rejects_off_contract_answers():
    assert parse_cells(cells_response())[0]["type"] == "markdown"
    for text in ("not json", '{"cells": []}', '{"cells": [{"type": "sql", "content": ""}]}'):
        with pytest.raises(ModelError) as error:
            parse_cells(text)
        assert error.value.kind == "malformed"


def test_matches():
    assert matches(33076, 33076.0)
    assert not matches(33076, 33077)  # whole numbers: exact
    assert matches(0.049492, 0.0495)  # others: within 0.1 %
    assert not matches(0.049492, 0.05)
    assert not matches(1, True)
    assert matches("2024-09", " 2024-09 ")


def test_load_answers_must_cover_exactly_the_claims(project):
    assert load_answers(project)["mean_a"] == 10.0
    project.answers_path.write_text("n_rows: 20\nmean_a: 10.0\nmean_b: 28.0\nextra: 1\n")
    with pytest.raises(ConfigError) as error:
        load_answers(project)
    assert "no answer for 'n_outliers'" in str(error.value)
    assert "'extra' is not a claim" in str(error.value)


def _write_findings(run_dir, findings: dict):
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "findings.json").write_text(json.dumps(findings))


def test_grade_pass_fail_missing(project, tmp_path):
    _write_findings(tmp_path / "a", {"n_rows": 20, "mean_a": 10.0, "mean_b": 28.0, "n_outliers": 1})
    assert grade(tmp_path / "a", project)["score"] == 1
    assert (tmp_path / "a" / CHECKS).exists()

    _write_findings(tmp_path / "b", {"n_rows": 20, "mean_a": 28.0})
    report = grade(tmp_path / "b", project)
    statuses = {claim["claim_id"]: claim["status"] for claim in report["claims"]}
    assert statuses == {
        "n_rows": "pass",
        "mean_a": "fail",
        "mean_b": "missing",
        "n_outliers": "missing",
    }
    assert (report["passed"], report["total"], report["score"]) == (1, 4, 0.25)


def _invocation(project, tmp_path, monkeypatch, answer: str, pattern="zero_shot"):
    monkeypatch.setattr(run, "chat", _fake_chat(answer))
    return run.Invocation(project, pattern, PATTERNS["zero_shot"], tmp_path / "runs")


def _no_docker(monkeypatch):
    monkeypatch.setattr(run, "ensure_image", lambda: "sha256:test")


def test_invocation_layout(project, tmp_path, monkeypatch):
    _no_docker(monkeypatch)
    invocation = _invocation(project, tmp_path, monkeypatch, "not json")
    first, second = invocation.run(1), invocation.run(2)
    invocation.finish()

    assert invocation.dir.name.endswith("_tiny_project_zero_shot")
    assert [first.dir.name, second.dir.name] == ["seed_1", "seed_2"]
    info = json.loads((invocation.dir / "invocation.json").read_text())
    assert info["seeds"]["2"] == {"status": "succeeded", "score": 0.0}
    assert {"dataset_sha", "answers_sha", "image_id", "prompt_sha", "model"} <= set(info)
    assert {p.name for p in invocation.dir.iterdir()} == {
        "invocation.json",
        "config.yaml",
        "seed_1",
        "seed_2",
    }
    run_info = json.loads(first.path(RUN_INFO).read_text())
    assert set(run_info) == {"run_id", "seed", "started_at", "status", "error", "ended_at", "score"}
    top = {p.name for p in first.dir.iterdir()}
    assert top == {"logs", "evaluation", "summary.md"}  # no notebook ran, so no result
    logs = {p.name for p in (first.dir / "logs").iterdir()}
    assert logs == {
        "run.json",
        "trace.jsonl",
        "prompt_001.md",
        "call_001.json",
        "response_001.txt",
    }
    assert {p.name for p in (first.dir / "evaluation").iterdir()} == {"checks.json", "metrics.json"}


def test_malformed_answer_is_logged_and_graded(project, tmp_path, monkeypatch):
    _no_docker(monkeypatch)
    result = _invocation(project, tmp_path, monkeypatch, "this is not json").run(1)
    assert result.path("logs/response_001.txt").read_text().startswith("Parsing failed")
    assert json.loads(result.path(RUN_INFO).read_text())["score"] == 0.0
    assert json.loads(result.path(METRICS).read_text())["first_error"] == "model_malformed"
    assert "Missing (4)" in result.path("summary.md").read_text()


@pytest.mark.docker
def test_zero_shot_end_to_end(project, tmp_path, monkeypatch):
    result = _invocation(project, tmp_path, monkeypatch, cells_response()).run(1)
    checks = json.loads(result.path(CHECKS).read_text())
    assert checks["score"] == 1, checks
    assert result.path("result.ipynb").exists()
    assert result.path("logs/response_001.ipynb").exists()
    assert result.path("artifacts/boxplot.png").exists()
    events = [json.loads(line)["event"] for line in result.path(TRACE).read_text().splitlines()]
    assert events.count("cell") == 3  # setup + two code cells


def _fake_chat(text: str):
    def chat(messages, config, *, seed, schema, timeout_s):
        request = {"model": config.name, "messages": messages, "options": {"seed": seed}}
        return request, {"message": {"content": text}, "prompt_eval_count": 1, "eval_count": 2}

    return chat


def test_edited_prompt_gets_its_own_row_in_compare(project, tmp_path, monkeypatch):
    _no_docker(monkeypatch)
    templates = tmp_path / "prompts"
    shutil.copytree(prompts.PROMPTS_DIR, templates)
    monkeypatch.setattr(prompts, "PROMPTS_DIR", templates)
    template = templates / "zero_shot.md"

    old = _invocation(project, tmp_path, monkeypatch, "not json")
    old.run(1)
    old.finish()
    again = _invocation(project, tmp_path, monkeypatch, "not json")  # same prompt: pooled
    again.run(2)
    again.finish()
    assert again.info["prompt_sha"] == old.info["prompt_sha"]

    template.write_text("Answer {research_question} as cells.")
    edited = _invocation(project, tmp_path, monkeypatch, "not json")
    assert "_tiny_project_zero_shot" in edited.dir.name  # (+ _N when started in the same second)
    assert edited.info["prompt_sha"] != old.info["prompt_sha"]
    run_dir = edited.run(1).dir
    assert (run_dir / "logs/prompt_001.md").read_text().startswith("Answer How do the two")
    edited.finish()

    (templates / "setup_cell.py").write_text("# changed setup\n")  # part of the prompt too
    assert prompts.prompt_sha("zero_shot") != edited.info["prompt_sha"]
    (templates / "other.md").write_text(template.read_text())
    other_pattern = _invocation(project, tmp_path, monkeypatch, "not json", pattern="other")
    other_pattern.run(1)
    other_pattern.finish()
    project.answers_path.write_text(project.answers_path.read_text() + "# edited\n")
    other_key = _invocation(project, tmp_path, monkeypatch, "not json")
    other_key.run(1)
    other_key.finish()

    project.answers_path.write_text(project.answers_path.read_text().replace("# edited\n", ""))
    other_model = _invocation(
        project.__class__(
            project.dir,
            project.config.model_copy(
                update={"model": project.config.model.model_copy(update={"name": "other"})}
            ),
        ),
        tmp_path,
        monkeypatch,
        "not json",
    )
    other_model.run(1)
    other_model.finish()

    table = compare(edited.dir, tmp_path / "runs")
    calls_part, rest = table.removeprefix("Calls\n").split("\nPrompts\n")
    prompts_part, overview = rest.split("\nOverview\n")

    def rows(part: str) -> list[str]:
        return [line for line in part.splitlines() if line.startswith("| ")][1:]  # no header

    call_rows = rows(calls_part)
    assert [row.split(" | ")[1] for row in call_rows] == [
        old.dir.name,
        again.dir.name,
        edited.dir.name,
    ]
    assert call_rows[2].startswith("| → |")
    assert "First errors" not in calls_part

    prompt_rows = rows(prompts_part)
    assert len(prompt_rows) == 2  # other pattern, answer key and model are left out
    new = next(row for row in prompt_rows if edited.info["prompt_sha"][:7] in row)
    pooled = next(row for row in prompt_rows if old.info["prompt_sha"][:7] in row)
    assert new.startswith("| → |")
    assert "| 0/2 |" in pooled  # perfect score: none of the two pooled seeds

    assert f"Call      {edited.dir.name} (prompt {edited.info['prompt_sha'][:7]})" in overview
    assert f"Model     {project.config.model.name}" in overview


def test_trace_prints_every_seed_or_one(tmp_path, monkeypatch, capsys):
    from eda_lite import cli

    for name in ("20260101T000000Z_p_zero_shot", "20260101T000001Z_p_zero_shot"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "invocation.json").write_text(json.dumps({"model": "m"}))
        for seed in (1, 2):
            logs = tmp_path / name / f"seed_{seed}" / "logs"
            logs.mkdir(parents=True)
            info = {"run_id": f"{name}/seed_{seed}", "seed": seed, "started_at": f"t{seed}"}
            (logs / "run.json").write_text(json.dumps(info))
            event = {"ts": "2026-01-01T00:00:00+00:00", "event": "run_finished", "status": "ok"}
            (logs / "trace.jsonl").write_text(json.dumps(event) + "\n")
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path)

    def trace(*args: str) -> str:
        monkeypatch.setattr("sys.argv", ["eda-lite", "trace", *args])
        cli.main()
        return capsys.readouterr().out

    out = trace("000000Z")
    assert "000000Z_p_zero_shot/seed_1" in out and "000000Z_p_zero_shot/seed_2" in out
    assert "000001Z" not in out
    out = trace("--seed", "2")  # latest invocation
    assert "000001Z_p_zero_shot/seed_2" in out and "seed_1" not in out
    with pytest.raises(SystemExit):
        trace("000000Z", "--seed", "9")
    assert "seeds: 1, 2" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        trace("_p_zero_shot")  # ambiguous
    assert "must match one invocation" in capsys.readouterr().err
