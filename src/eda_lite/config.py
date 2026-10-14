"""Project specification: schema, loading and validation of `project.yaml`."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[2]  # the eda-lite directory
PROJECTS_ROOT = ROOT / "projects"


class ConfigError(Exception):
    """The project specification is missing, malformed or incomplete."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    name: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"  # the Ollama server
    temperature: float = Field(0.2, ge=0)
    request_timeout_s: float = Field(30, gt=0)  # per model call; independent of execution limits
    think: bool = False
    num_ctx: int = Field(8192, gt=0)


class ExecutionConfig(_Strict):
    cpus: float = Field(2, gt=0)
    memory: str = "2g"
    code_timeout_s: float = Field(60, gt=0)  # per notebook cell or standalone script
    total_run_budget_s: float = Field(900, gt=0)  # whole run: model calls plus executions


class BudgetConfig(_Strict):
    """Hard limits per run. Reaching one ends the run cleanly."""

    max_model_calls: int = Field(10, ge=1)


class ProjectConfig(_Strict):
    """Contents of a project's `project.yaml`."""

    project_id: str
    research_question: str = Field(min_length=1)
    # What the notebook must report with `report(claim_id, value)`: claim_id -> definition.
    # Never an expected value; those live in evaluation_spec/answers.yaml.
    claims: dict[str, str] = Field(min_length=1)
    dataset: str  # file name of the CSV inside the project's data/ directory
    seeds: list[int] = [1, 2, 3]  # one run per seed
    model: ModelConfig = ModelConfig()
    execution: ExecutionConfig = ExecutionConfig()
    budgets: BudgetConfig = BudgetConfig()


@dataclass(frozen=True)
class Project:
    """A project directory with its validated configuration. The layout is fixed:

    projects/<project_id>/
      project.yaml       the public task (this schema)
      data/<dataset>     the CSV, mounted read-only into the sandbox
      docs/*.md          documentation, pasted into the prompt
      evaluation_spec/   EVALUATOR-ONLY: answers.yaml, never mounted, never in a prompt
    """

    dir: Path
    config: ProjectConfig

    @property
    def dataset_path(self) -> Path:
        return self.dir / "data" / self.config.dataset

    @property
    def docs_dir(self) -> Path:
        return self.dir / "docs"

    @property
    def answers_path(self) -> Path:
        return self.dir / "evaluation_spec" / "answers.yaml"


def load_yaml(path: Path, model: type[BaseModel]):
    """Read a YAML file into `model`; every problem is reported with its field path."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        problems = [
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in error.errors()
        ]
        raise ConfigError(f"invalid {path}:\n  " + "\n  ".join(problems)) from error


def load_project(project_id: str, root: Path = PROJECTS_ROOT) -> Project:
    """The project in `<root>/<project_id>/`, with every input checked to exist."""
    project_dir = (root / project_id).resolve()
    if Path(project_id).name != project_id or project_id.startswith("."):
        raise ConfigError(f"'{project_id}' is a path; give a project ID, a folder in {root}")
    if not (project_dir / "project.yaml").is_file():
        known = sorted(path.parent.name for path in root.glob("*/project.yaml"))
        raise ConfigError(f"no project '{project_id}' in {root}; known: {', '.join(known)}")
    project = Project(project_dir, load_yaml(project_dir / "project.yaml", ProjectConfig))
    if project.config.project_id != project_id:
        raise ConfigError(
            f"{project_dir / 'project.yaml'} says project_id: {project.config.project_id}; "
            f"it must match its folder name, {project_id}"
        )
    for path in (project.dataset_path, project.docs_dir, project.answers_path):
        if not path.exists():
            raise ConfigError(f"project input missing: {path}")
    return project
