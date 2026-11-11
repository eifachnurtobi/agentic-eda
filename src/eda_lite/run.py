import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import nbformat
import yaml

from eda_lite import prompts
from eda_lite.config import ROOT, Project
from eda_lite.grade import INVOCATION, LOGS, METRICS, RUN_INFO, TRACE, grade, metrics, read_trace
from eda_lite.model import CELLS_SCHEMA, ModelError, chat, parse_cells
from eda_lite.report import write_summary
from eda_lite.sandbox import Execution, build_notebook, ensure_image, run_notebook

RUNS_ROOT = ROOT / "runs"


class BudgetExhausted(Exception):
    pass


@dataclass
class Answer:
    call: str
    text: str | None = None
    cells: list[dict] | None = None
    notebook: str | None = None
    error: ModelError | None = None


class Invocation:
    def __init__(
        self,
        project: Project,
        pattern_name: str,
        pattern: "Pattern",
        runs_root: Path = RUNS_ROOT,
    ):
        self.project, self.config = project, project.config
        self.pattern_name, self.pattern = pattern_name, pattern
        prompt_sha = prompts.prompt_sha(pattern_name)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.dir = _new_dir(Path(runs_root) / f"{stamp}_{self.config.project_id}_{pattern_name}")
        self.runs: list[Run] = []
        (self.dir / "config.yaml").write_text(
            yaml.safe_dump(self.config.model_dump(), sort_keys=False)
        )
        self.info = {
            "invocation_id": self.dir.name,
            "command": sys.argv,
            "project_id": self.config.project_id,
            "pattern": pattern_name,
            "prompt_sha": prompt_sha,
            "model": self.config.model.name,
            "started_at": _now(),
            "git": _git_version(),
            "dataset_sha": prompts.sha(project.dataset_path),
            "answers_sha": prompts.sha(project.answers_path),
            "image_id": ensure_image(),
        }
        self._write_info()

    def _write_info(self, **fields) -> None:
        self.info |= fields
        (self.dir / INVOCATION).write_text(json.dumps(self.info, indent=2) + "\n")

    def run(self, seed: int) -> "Run":

        run = run_once(self, seed)
        self.runs.append(run)
        return run

    def finish(self) -> None:

        seeds = {
            str(run.seed): {k: run.info.get(k) for k in ("status", "score")} for run in self.runs
        }
        self._write_info(ended_at=_now(), seeds=seeds)


class Run:
    def __init__(self, invocation: Invocation, seed: int):
        self.project, self.config = invocation.project, invocation.config
        self.seed = seed
        self.default_prompt = invocation.pattern_name
        self.dir = invocation.dir / f"seed_{seed}"
        self.dir.mkdir()
        self.id = f"{invocation.dir.name}/{self.dir.name}"
        self.calls = self.executions = 0
        self.deadline = time.monotonic() + self.config.execution.total_run_budget_s
        self.info = {"run_id": self.id, "seed": seed, "started_at": _now(), "status": "running"}
        self._write_info()

    def path(self, name: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, name: str, data) -> str:

        text = data if isinstance(data, str) else json.dumps(data, indent=2, default=str) + "\n"
        self.path(name).write_text(text)
        return name

    def log(self, event: str, **fields) -> None:
        line = {"ts": _now(), "event": event, **fields}
        with open(self.path(TRACE), "a") as trace:
            trace.write(json.dumps(line, default=str) + "\n")

    def _write_info(self, **fields) -> None:
        self.info |= fields
        self.save(RUN_INFO, self.info)

    def remaining_s(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExhausted("total_run_budget_s")
        return remaining

    def prompt(self, name: str | None = None, **extra) -> str:

        return prompts.render(name or self.default_prompt, self.project, **extra)

    def ask(
        self,
        prompt: str | list[dict],
        *,
        step: str = "generate",
        cells: bool = True,
    ) -> Answer:

        if self.calls >= self.config.budgets.max_model_calls:
            raise BudgetExhausted("max_model_calls")
        self.calls += 1
        answer = Answer(call=f"call_{self.calls:03d}")
        number = answer.call[5:]
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        event = {
            "step": step,
            "call": f"{LOGS}/{answer.call}.json",
            "prompt": self.save(f"{LOGS}/prompt_{number}.md", _readable(messages)),
        }

        remaining = self.remaining_s()
        timeout_s = min(self.config.model.request_timeout_s, remaining)
        record: dict = {"timeout_s": timeout_s}
        started = time.monotonic()
        try:
            record["request"], record["response"] = chat(
                messages,
                self.config.model,
                seed=self.seed,
                schema=CELLS_SCHEMA if cells else None,
                timeout_s=timeout_s,
            )
            body = record["response"]
            answer.text = body["message"]["content"]
            event |= {
                "input_tokens": body.get("prompt_eval_count"),
                "output_tokens": body.get("eval_count"),
            }
            if cells:
                answer.cells = parse_cells(answer.text)
                notebook = build_notebook(answer.cells, prompts.setup_code(self.project))
                event["response"] = answer.notebook = self.save(
                    f"{LOGS}/response_{number}.ipynb", nbformat.writes(notebook)
                )
            else:
                event["response"] = self.save(f"{LOGS}/response_{number}.md", answer.text)
        except ModelError as error:
            answer.error = error
            record["error"] = event["error"] = {"kind": error.kind, "message": str(error)}
            if error.payload is not None:
                record["request"], record["response"] = error.payload, error.partial
            if answer.text is not None:
                event["response"] = self.save(
                    f"{LOGS}/response_{number}.txt", f"Parsing failed: {error}\n\n{answer.text}\n"
                )
        body = record.get("response") or {}
        thinking = body.get("message", {}).get("thinking")
        if thinking:
            event["thinking"] = self.save(f"{LOGS}/thinking_{number}.md", thinking)
        if body.get("continuation"):
            event["continuation"] = self.save(
                f"{LOGS}/continuation_{number}.md", body["continuation"]["prompt"]
            )
        event |= {
            "thinking_chars": len(thinking or ""),
            "thinking_truncated": body.get("thinking_truncated", False),
            "done_reason": body.get("done_reason"),
        }
        self.save(event["call"], record)
        self.log("model_call", latency_s=round(time.monotonic() - started, 3), **event)
        if answer.error and answer.error.kind == "timeout" and timeout_s == remaining:
            raise BudgetExhausted("total_run_budget_s")
        return answer

    def execute(self, answer: Answer, *, step: str = "execute") -> Execution:

        self.executions += 1
        number = f"{self.executions:03d}"
        artifacts = self.path("artifacts")
        shutil.rmtree(artifacts, ignore_errors=True)
        result = run_notebook(
            self.path(answer.notebook), self.project, artifacts, self.remaining_s()
        )
        if result.executed is not None:
            nbformat.write(result.executed, self.path("result.ipynb"))
        else:
            shutil.copyfile(self.path(answer.notebook), self.path("result.ipynb"))

        for cell in result.cells:
            self.log(
                "cell",
                step=step,
                cell_id=cell["cell_id"],
                status=cell["status"],
                duration_s=cell["duration_s"],
                outputs=cell["outputs"],
                error=cell["error"],
                code=f"{answer.notebook}#{cell['cell_id']}",
            )
        streams = {
            f"{name}_file": self.save(f"{LOGS}/exec_{number}.{name}.txt", text)
            for name, text in (("stdout", result.stdout), ("stderr", result.stderr))
            if text.strip()
        }
        self.log(
            "execution",
            step=step,
            notebook=answer.notebook,
            status=result.status,
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            artifacts=sorted(p.name for p in artifacts.rglob("*") if p.is_file()),
            **streams,
        )
        return result


Pattern = Callable[[Run], None]


def run_once(invocation: Invocation, seed: int) -> Run:

    run = Run(invocation, seed)
    run.log(
        "run_started",
        pattern=invocation.pattern_name,
        seed=seed,
        model=run.config.model.name,
    )
    status, error = "failed", None
    try:
        invocation.pattern(run)
        status = "succeeded"
    except BudgetExhausted as exhausted:
        run.log("budget_exhausted", budget=str(exhausted))
        error = f"budget exhausted: {exhausted}"
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        run.log("run_failed", error=error)
        if not isinstance(failure, Exception):
            raise
    finally:
        checks = grade(run.dir, run.project)
        run.log("grade", score=checks["score"], passed=checks["passed"], total=checks["total"])
        run.log("run_finished", status=status)
        run.save(METRICS, metrics(read_trace(run.dir)))
        run._write_info(
            status=status,
            error=error,
            ended_at=_now(),
            score=checks["score"],
        )
        write_summary(run.dir)
    return run


def _new_dir(path: Path) -> Path:

    candidate, number = path, 1
    while True:
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            number += 1
            candidate = path.with_name(f"{path.name}_{number}")


def _readable(messages: list[dict]) -> str:

    if len(messages) == 1:
        return messages[0]["content"]
    return "\n\n".join(f"<!-- {m['role']} -->\n{m['content']}" for m in messages)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _git_version() -> dict:
    def git(*args):
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "--short", "HEAD")
    return {"sha": sha, "dirty": bool(git("status", "--porcelain", ".")) if sha else None}
