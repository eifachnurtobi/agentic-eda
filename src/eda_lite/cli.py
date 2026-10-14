"""`eda-lite run | prompt | trace | compare`; see `eda-lite --help`.

Paths are fixed: projects come from `projects/<project_id>/`, prompts from
`src/eda_lite/prompts/<name>.md`, runs go to `runs/`.
"""

import argparse
import json
from pathlib import Path

from eda_lite import prompts
from eda_lite.config import ConfigError, Project, load_project
from eda_lite.grade import CHECKS, INVOCATION, METRICS, RUN_INFO, load_answers
from eda_lite.patterns import PATTERNS
from eda_lite.report import compare, headline, render_trace
from eda_lite.run import RUNS_ROOT, Invocation
from eda_lite.sandbox import DockerUnavailable

PROJECT_HELP = "project ID, a folder in projects/"
RUN_HELP = "`latest`, or part of an invocation's folder name in runs/"


def main() -> int:
    parser = argparse.ArgumentParser(prog="eda-lite")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a pattern once per seed, then compare its prompt")
    prompt = commands.add_parser("prompt", help="print the filled prompt, without calling a model")
    for sub in (run, prompt):
        sub.add_argument("project", help=PROJECT_HELP)
        sub.add_argument(
            "--pattern", choices=sorted(PATTERNS), required=True, help="uses prompts/<pattern>.md"
        )
    run.add_argument("--seeds", type=int, nargs="+", help="default: seeds in project.yaml")

    trace = commands.add_parser("trace", help="an invocation's trace, every seed or one")
    trace.add_argument("run", nargs="?", default="latest", help=RUN_HELP)
    trace.add_argument("--seed", type=int, help="only this seed")

    comp = commands.add_parser(
        "compare",
        help="compare an invocation's prompt with every other prompt tried on the same "
        "project, pattern, dataset and answer key",
    )
    comp.add_argument("run", nargs="?", default="latest", help=RUN_HELP)

    args = parser.parse_args()
    try:
        return COMMANDS[args.command](args)
    except (ConfigError, FileNotFoundError, DockerUnavailable) as error:
        parser.exit(2, f"error: {error}\n")


def _project(project_id: str) -> Project:
    project = load_project(project_id)
    load_answers(project)  # no run starts without a valid answer key
    return project


def _run(args) -> int:
    project = _project(args.project)
    seeds = list(dict.fromkeys(args.seeds or project.config.seeds))
    invocation = Invocation(project, args.pattern, PATTERNS[args.pattern])
    print(f"{invocation.dir.name} · {project.config.model.name}")
    for seed in seeds:
        print(f"  seed {seed}: running ...", end="\r", flush=True)
        run = invocation.run(seed)
        metrics = json.loads(run.path(METRICS).read_text())
        checks = json.loads(run.path(CHECKS).read_text())
        parts = [
            headline(checks),
            metrics["first_error"],
            f"{metrics['wall_clock_s']:.0f} s",
        ]
        print(f"  seed {seed}: {' · '.join(p for p in parts if p)}")
    invocation.finish()
    print("\n" + compare(invocation.dir, RUNS_ROOT))
    return 0


def _prompt(args) -> int:
    print(prompts.render(args.pattern, _project(args.project)), end="")
    return 0


def _trace(args) -> int:
    invocation = _find_invocation(args.run)
    infos = {
        path.parent.parent: json.loads(path.read_text())
        for path in invocation.glob(f"seed_*/{RUN_INFO}")
    }
    ordered = sorted(infos, key=lambda seed_dir: infos[seed_dir]["started_at"])
    chosen = [d for d in ordered if args.seed is None or infos[d]["seed"] == args.seed]
    if not chosen:
        known = ", ".join(str(infos[d]["seed"]) for d in ordered) or "none"
        raise ConfigError(f"{invocation.name} has no seed {args.seed}; seeds: {known}")
    print("\n".join(render_trace(seed_dir) for seed_dir in chosen), end="")
    return 0


def _compare(args) -> int:
    print(compare(_find_invocation(args.run), RUNS_ROOT))
    return 0


def _find_invocation(reference: str) -> Path:
    """The newest invocation for `latest`, else the one whose folder name contains `reference`."""
    invocations = sorted(path.parent for path in RUNS_ROOT.glob(f"*/{INVOCATION}"))
    if reference == "latest":
        matches = invocations[-1:]
    else:
        matches = [path for path in invocations if reference in path.name]
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise ConfigError(f"'{reference}' must match one invocation in {RUNS_ROOT}; found: {found}")
    return matches[0]


COMMANDS = {"run": _run, "prompt": _prompt, "trace": _trace, "compare": _compare}

if __name__ == "__main__":
    raise SystemExit(main())
