"""Automated grading against the project's answer key, `evaluation_spec/answers.yaml`, plus
cost and reliability metrics from the trace. A clean execution never counts as a correct one.

The answer key is flat: `claim_id: expected value`, one per claim in project.yaml. One rule
compares a reported value with it: text and whole numbers must match exactly, other numbers
within 0.1% (relative).
"""

import json
import math
from datetime import datetime
from pathlib import Path

import yaml

from eda_lite.config import ConfigError, Project

FINDINGS = "artifacts/findings.json"  # {claim_id: value}, written by report() in the setup cell
REL_TOL = 1e-3  # for numbers that are not whole
INVOCATION = "invocation.json"  # in the invocation folder: what all its seeds share
# Where a run keeps its record and its grade (paths inside a run folder).
LOGS, EVALUATION = "logs", "evaluation"
TRACE, RUN_INFO = f"{LOGS}/trace.jsonl", f"{LOGS}/run.json"
CHECKS, METRICS = f"{EVALUATION}/checks.json", f"{EVALUATION}/metrics.json"


def load_answers(project: Project) -> dict:
    """The answer key; it must give a value for exactly the claims in project.yaml."""
    path = project.answers_path
    answers = yaml.safe_load(path.read_text())
    if not isinstance(answers, dict):
        raise ConfigError(f"{path} must map each claim_id to its expected value")
    claims = project.config.claims
    problems = [f"no answer for '{c}'" for c in claims if answers.get(c) is None]
    problems += [f"'{c}' is not a claim in project.yaml" for c in answers if c not in claims]
    if problems:
        raise ConfigError(f"invalid {path}:\n  " + "\n  ".join(problems))
    return answers


def load_findings(path: Path) -> dict:
    """claim_id -> reported value; empty if the file is missing or not a JSON object."""
    try:
        findings = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return findings if isinstance(findings, dict) else {}


def matches(expected, actual) -> bool:
    if _is_number(expected) and _is_number(actual):
        if float(expected).is_integer():
            return actual == expected
        return math.isclose(actual, expected, rel_tol=REL_TOL)
    return str(expected).strip() == str(actual).strip()


def grade(run_dir: Path, project: Project) -> dict:
    """Grade every claim as pass, fail or missing. Writes `evaluation/checks.json`."""
    answers, findings = load_answers(project), load_findings(run_dir / FINDINGS)
    claims = []
    for claim_id in project.config.claims:
        expected = answers[claim_id]
        claim = {"claim_id": claim_id, "status": "missing", "expected": expected}
        if claim_id in findings:
            value = findings[claim_id]
            claim |= {"status": "pass" if matches(expected, value) else "fail", "value": value}
        claims.append(claim)
    passed = sum(claim["status"] == "pass" for claim in claims)
    report = {
        "score": round(passed / len(claims), 3),
        "passed": passed,
        "total": len(claims),
        "claims": claims,
    }
    (run_dir / EVALUATION).mkdir(exist_ok=True)
    (run_dir / CHECKS).write_text(json.dumps(report, indent=2) + "\n")
    return report


def read_trace(run_dir: Path) -> list[dict]:
    path = run_dir / TRACE
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.exists()
        else []
    )


def metrics(events: list[dict]) -> dict:
    """Cost and reliability of one run, computed from its trace alone."""
    calls = [e for e in events if e["event"] == "model_call"]
    answered = [e for e in calls if not e.get("error")]
    cells = [e for e in events if e["event"] == "cell"]
    first_error, detail = _first_error(events)
    wall = 0.0
    if events:
        first, last = (datetime.fromisoformat(events[i]["ts"]) for i in (0, -1))
        wall = round((last - first).total_seconds(), 3)
    return {
        "model_calls": len(calls),
        "model_errors": len(calls) - len(answered),
        "input_tokens": sum(e.get("input_tokens") or 0 for e in answered),
        "output_tokens": sum(e.get("output_tokens") or 0 for e in answered),
        "model_latency_s": round(sum(e["latency_s"] for e in calls), 3),
        "executions": sum(e["event"] == "execution" for e in events),
        "cells_run": len(cells),
        "cell_success_rate": (
            round(sum(e["status"] == "ok" for e in cells) / len(cells), 3) if cells else None
        ),
        "first_error": first_error,
        "first_error_detail": detail,
        "wall_clock_s": wall,
    }


def _first_error(events: list[dict]) -> tuple[str | None, str | None]:
    """The first thing that went wrong: the most useful single label for a failed run."""
    for e in events:
        if e["event"] == "model_call" and e.get("error"):
            return f"model_{e['error']['kind']}", e["error"]["message"]
        if e["event"] == "cell" and e["status"] == "error":
            return f"cell_error:{e['error']['name']}", f"{e['cell_id']}: {e['error']['message']}"
        if e["event"] == "cell" and e["status"] == "timeout":
            return "cell_timeout", e["cell_id"]
        if e["event"] == "execution" and e["status"] == "killed":
            return "execution_killed", f"container killed after {e['duration_s']:.0f}s"
        if e["event"] in ("budget_exhausted", "run_failed"):
            return e["event"], e.get("budget") or e.get("error")
    return None, None


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
