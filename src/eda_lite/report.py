"""What a person reads: a run's summary.md, its trace as a timeline, a prompt comparison.

Evaluator-side: these show expected values, so no pattern may put them in front of a model.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import yaml

from eda_lite.grade import CHECKS, INVOCATION, METRICS, RUN_INFO, TRACE, read_trace


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# -- summary.md ------------------------------------------------------------------------


def headline(checks: dict) -> str:
    counts = Counter(claim["status"] for claim in checks["claims"])
    parts = [f"{checks['passed']}/{checks['total']} claims correct"]
    parts += [
        f"{counts[s]} {w}" for s, w in (("fail", "wrong"), ("missing", "missing")) if counts[s]
    ]
    return " · ".join(parts)


def write_summary(run_dir: Path) -> None:
    info, checks, m = (_read(run_dir / name) for name in (RUN_INFO, CHECKS, METRICS))
    shared = _read(run_dir.parent / INVOCATION)
    facts = [
        shared["pattern"],
        shared["model"],
        f"seed {info['seed']}",
        f"{m['wall_clock_s']:.0f} s",
        f"{m['input_tokens']} in / {m['output_tokens']} out tokens",
    ]
    lines = [f"# {info['run_id']}", "", f"**{headline(checks)}**", "", " · ".join(facts)]
    if info.get("error"):
        lines += ["", f"Run ended early: {info['error']}"]

    if m.get("first_error"):
        lines += ["", "## What went wrong", "", f"`{m['first_error']}` — {m['first_error_detail']}"]
        failed = next((e for e in read_trace(run_dir) if e["event"] == "cell" and e["error"]), None)
        source = _cell_source(run_dir, failed["code"]) if failed else None
        if source:
            lines += ["", f"The failing cell (`{failed['code']}`):", "", "```python", source, "```"]

    by_status: dict[str, list] = {}
    for claim in checks["claims"]:
        by_status.setdefault(claim["status"], []).append(claim)
    lines += ["", "## Claims", ""]
    for status, word in (("pass", "Correct"), ("missing", "Missing")):
        if status in by_status:
            names = ", ".join(c["claim_id"] for c in by_status[status])
            lines += [f"{word} ({len(by_status[status])}): {names}", ""]
    if "fail" in by_status:
        lines += ["| Wrong claim | Reported | Expected |", "|---|---|---|"]
        lines += [
            f"| {c['claim_id']} | {c['value']} | {c['expected']} |" for c in by_status["fail"]
        ]

    lines += [
        "",
        "## Files",
        "",
        "- `result.ipynb` notebook with outputs, `artifacts/` the files it wrote",
        "- `evaluation/` the grade (`checks.json`) and cost (`metrics.json`)",
        "- `logs/prompt_NNN.md`, `call_NNN.json`, `response_NNN.*` per model call",
        f"- `{TRACE}` (`eda-lite trace {info['run_id']}`)",
    ]
    lines += [
        f"- ![{p.name}]({p.relative_to(run_dir)})"
        for p in sorted((run_dir / "artifacts").glob("*.png"))
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _cell_source(run_dir: Path, code_ref: str) -> str | None:
    path, _, cell_id = code_ref.partition("#")
    if not (run_dir / path).exists():
        return None
    cells = json.loads((run_dir / path).read_text())["cells"]
    return next(("".join(c["source"]).strip() for c in cells if c.get("id") == cell_id), None)


# -- trace timeline --------------------------------------------------------------------


def render_trace(run_dir: Path) -> str:
    info, events = _read(run_dir / RUN_INFO), read_trace(run_dir)
    model = _read(run_dir.parent / INVOCATION).get("model")
    lines = [" · ".join(str(x) for x in (info.get("run_id"), model, info.get("status"))), ""]
    if not events:
        return lines[0] + "\n(no events)\n"
    start = datetime.fromisoformat(events[0]["ts"])
    for e in events:
        offset = (datetime.fromisoformat(e["ts"]) - start).total_seconds()
        ref = e.get("call") or e.get("code") or e.get("stderr_file") or ""
        lines.append(
            f"{offset:7.1f}s  {e.get('step', '-'):<10} {_describe(e)[:64]:<64}  {ref}".rstrip()
        )
    return "\n".join(lines) + "\n"


def _describe(e: dict) -> str:
    kind = e["event"]
    if kind == "model_call":
        if e.get("error"):
            return f"MODEL {e['error']['kind'].upper()}: {e['error']['message']}"
        return (
            f"model call {e['latency_s']:.1f}s  {e['input_tokens']} in / {e['output_tokens']} out"
        )
    if kind == "cell":
        if e["status"] == "ok":
            return f"{e['cell_id']} ok  {e['duration_s']:.2f}s  [{', '.join(e['outputs']) or '-'}]"
        detail = f"{e['error']['name']}: {e['error']['message']}" if e["error"] else ""
        return f"{e['cell_id']} {e['status'].upper()}  {detail}"
    if kind == "execution":
        return f"execution {e['status']} ({e['duration_s']:.1f}s, {len(e['artifacts'])} files)"
    if kind == "grade":
        return f"grade: {e['passed']}/{e['total']} claims correct"
    hidden = ("ts", "event", "step")
    return " ".join(
        [kind.replace("_", " "), *(f"{k}={v}" for k, v in e.items() if k not in hidden)]
    )


# -- comparison: one prompt against the others ---------------------------------------


def compare(target: Path, runs_root: Path) -> str:
    """The prompt of invocation `target` against every other prompt tried with the same
    pattern, an identical config.yaml (project, model settings, execution limits), the same
    dataset and the same answer key: a table of the invocations, a table of the prompts
    (invocations with the same prompt text pool their seeds), then an overview of what they
    share. A failed seed counts as 0."""
    info = _read(target / INVOCATION) | {"config_sha": _config_sha(target)}
    same = ("pattern", "config_sha", "dataset_sha", "answers_sha")
    calls: dict[str, tuple[str, list[dict]]] = {}  # invocation -> (prompt_sha, seed rows)
    for path in sorted(runs_root.glob(f"*/{INVOCATION}")):
        other = _read(path) | {"config_sha": _config_sha(path.parent)}
        if any(other.get(key) != info.get(key) for key in same) or "ended_at" not in other:
            continue
        seeds = sorted(
            (
                {
                    "seed": _read(d / RUN_INFO).get("seed"),
                    "score": _read(d / CHECKS).get("score", 0),
                }
                | _read(d / METRICS)
                for d in path.parent.glob("seed_*")
            ),
            key=lambda row: row["seed"],
        )
        calls[path.parent.name] = (other["prompt_sha"], seeds)

    by_prompt: dict[str, list[dict]] = {}
    for sha, seeds in calls.values():
        by_prompt.setdefault(sha, []).extend(seeds)
    per_call = [
        [
            "→" if name == target.name else "",
            name,
            sha[:7],
            ", ".join(str(row["seed"]) for row in seeds),
            *_stats(seeds),
        ]
        for name, (sha, seeds) in calls.items()
    ]
    prompts = [
        ["→" if sha == info["prompt_sha"] else "", sha[:7], *_stats(seeds), _first_errors(seeds)]
        for sha, seeds in sorted(by_prompt.items(), key=lambda kv: -_mean_score(kv[1]))
    ]
    return "\n".join(
        [
            "Calls",
            *_table(["", "Call", "Prompt", "Seeds", *STAT_COLUMNS], per_call),
            "Prompts",
            *_table(["", "Prompt", *STAT_COLUMNS, "First errors"], prompts),
            "Overview",
            f"Project   {info['project_id']}",
            f"Pattern   {info['pattern']}",
            f"Model     {info['model']}",
            f"Thinking  {_thinking(target)}",
            f"Config    {info['config_sha'][:7]} (identical config.yaml in every call)",
            f"Dataset   {info['dataset_sha'][:7]} · answers {info['answers_sha'][:7]}",
            f"Call      {target.name} (prompt {info['prompt_sha'][:7]})",
            "Means are over seeds (a call's seeds, or all seeds with that prompt).",
            "Score = share of claims correct; a failed seed counts as 0. Perfect score = seeds",
            "with every claim correct, out of all seeds.",
        ]
    )


# The columns both comparison tables share, computed by _stats over a list of seed rows.
STAT_COLUMNS = [
    "Score (mean ± sd)",
    "Score (min – max)",
    "Perfect score",
    "Input tokens (mean)",
    "Output tokens (mean)",
    "Seconds (mean)",
]


def _config_sha(invocation: Path) -> str | None:
    """The hash of an invocation's config.yaml: only calls with identical files are compared."""
    path = invocation / "config.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def _thinking(invocation: Path) -> str:
    """Whether the model thought, from config.yaml (identical for every compared call)."""
    model = yaml.safe_load((invocation / "config.yaml").read_text())["model"]
    if not model.get("think"):
        return "off"
    budget = model.get("thinking_budget")
    return f"on (budget {budget} tokens)" if budget else "on (no budget)"


def _mean_score(seeds: list[dict]) -> float:
    return mean(row["score"] for row in seeds)


def _stats(seeds: list[dict]) -> list:
    scores = [row["score"] for row in seeds]
    return [
        f"{mean(scores):.2f} ± {pstdev(scores):.2f}",
        f"{min(scores):.2f} – {max(scores):.2f}",
        f"{sum(score == 1 for score in scores)}/{len(scores)}",
        round(mean(row.get("input_tokens", 0) for row in seeds)),
        round(mean(row.get("output_tokens", 0) for row in seeds)),
        round(mean(row.get("wall_clock_s", 0) for row in seeds)),
    ]


def _first_errors(seeds: list[dict]) -> str:
    errors = Counter(row["first_error"] for row in seeds if row.get("first_error"))
    return ", ".join(f"{k} x{v}" for k, v in errors.most_common()) or "-"


def _table(columns: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    return [*lines, *("| " + " | ".join(str(c) for c in row) + " |" for row in rows), ""]
