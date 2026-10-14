"""Generated code only ever runs here: a locked-down container declared in `compose.yaml`
(no network, read-only root, no capabilities, CPU/memory/process limits, your user).

The container sees the dataset file, `docs/` and the notebook read-only, and writes only to
`artifacts/`. The trusted executor (`docker/nb_executor.py`) runs the cells one by one and
writes a journal line per cell, so a killed container still leaves an accurate record.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from eda_lite.config import ROOT, Project

COMPOSE = ["docker", "compose", "--file", str(ROOT / "compose.yaml"), "--progress", "quiet"]
EXECUTOR = ROOT / "docker" / "nb_executor.py"
WORKDIR = "/work"
STARTUP_MARGIN_S = 60  # container and kernel start-up on top of the per-cell timeouts
STATUS_BY_EXIT_CODE = {0: "ok", 1: "error", 2: "timeout"}  # nb_executor.py's exit codes


class DockerUnavailable(Exception):
    pass


def build_notebook(cells: list[dict], setup: str) -> nbformat.NotebookNode:
    """The engine's setup cell (id `setup`), then the model's cells (`cell_0`, `cell_1`, ...)."""
    notebook = new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
        }
    )
    notebook.cells.append(new_code_cell(setup, id="setup"))
    for index, cell in enumerate(cells):
        make = new_code_cell if cell["type"] == "code" else new_markdown_cell
        notebook.cells.append(make(cell["content"], id=f"cell_{index}"))
    nbformat.validate(notebook)
    return notebook


def ensure_image() -> str:
    """Build the sandbox image if it is missing; return its ID."""
    if shutil.which("docker") is None:
        raise DockerUnavailable("docker CLI not found on PATH")
    image = _docker(*COMPOSE, "config", "--images").stdout.strip()
    inspect = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    result = _docker(*inspect, check=False)
    if result.returncode != 0 and "No such image" in result.stderr:
        _docker(*COMPOSE, "build")
        result = _docker(*inspect, check=False)
    if result.returncode != 0:
        raise DockerUnavailable(f"docker is not usable: {result.stderr.strip()}")
    return result.stdout.strip()


def _docker(*command: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise DockerUnavailable(f"`{' '.join(command[-2:])}` failed: {result.stderr.strip()}")
    return result


@dataclass
class Execution:
    status: str  # ok | error | timeout (a cell hit code_timeout_s) | killed (whole container)
    exit_code: int | None
    duration_s: float
    stdout: str
    stderr: str
    cells: list[dict] = field(default_factory=list)  # journal: cell_id, status, error, outputs
    executed: nbformat.NotebookNode | None = None  # with outputs; None if no cell finished


def run_notebook(
    notebook_path: Path, project: Project, artifacts: Path, max_duration_s: float
) -> Execution:
    """Execute a notebook in a fresh container and kernel. A failing cell does not stop the
    cells after it; a cell timeout does. Files the notebook writes land in `artifacts`."""
    execution = project.config.execution
    notebook = json.loads(notebook_path.read_text())
    code_cells = sum(cell["cell_type"] == "code" for cell in notebook["cells"])
    timeout_s = min(code_cells * execution.code_timeout_s + STARTUP_MARGIN_S, max_duration_s)
    artifacts.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=artifacts.parent, prefix=".exec_") as exchange:
        mounts = [
            (project.dataset_path, f"{WORKDIR}/data/{project.config.dataset}", "ro"),
            (project.docs_dir, f"{WORKDIR}/docs", "ro"),
            (notebook_path, f"{WORKDIR}/input.ipynb", "ro"),
            (EXECUTOR, "/opt/eda/nb_executor.py", "ro"),
            (artifacts, f"{WORKDIR}/artifacts", "rw"),
            (Path(exchange), f"{WORKDIR}/exec", "rw"),
        ]
        command = [
            *("python", "/opt/eda/nb_executor.py", f"{WORKDIR}/input.ipynb", f"{WORKDIR}/exec"),
            *("--cell-timeout", str(execution.code_timeout_s)),
        ]
        exit_code, stdout, stderr, duration_s = _run_container(
            command, mounts, execution.cpus, execution.memory, timeout_s
        )
        journal = Path(exchange) / "journal.jsonl"
        executed = Path(exchange) / "executed.ipynb"
        cells = (
            [json.loads(line) for line in journal.read_text().splitlines() if line]
            if (journal.exists())
            else []
        )
        result = nbformat.read(executed, as_version=4) if executed.exists() else None

    status = "killed" if exit_code is None else STATUS_BY_EXIT_CODE.get(exit_code, "error")
    return Execution(status, exit_code, duration_s, stdout, stderr, cells, result)


def _run_container(command, mounts, cpus, memory, timeout_s):
    """Run one sandbox container to completion, or kill it at the deadline (exit_code None)."""
    name = f"eda-lite-{uuid.uuid4().hex[:12]}"
    args = [*COMPOSE, "run", "--rm", "--no-TTY", "--no-deps", "--name", name]
    args += ["--user", f"{os.getuid()}:{os.getgid()}"]
    for host, container, mode in mounts:
        args += ["--volume", f"{Path(host).resolve()}:{container}{':ro' if mode == 'ro' else ''}"]
    started = time.monotonic()
    process = subprocess.Popen(
        [*args, "sandbox", *command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "EDA_CPUS": str(cpus), "EDA_MEMORY": memory},  # read by compose.yaml
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        # Killing the CLI would leave the container running; kill the container itself.
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        stdout, stderr = process.communicate()
        exit_code = None
    return exit_code, stdout, stderr, round(time.monotonic() - started, 3)
