from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT / "projects"


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    name: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(0.2, ge=0)
    request_timeout_s: float = Field(30, gt=0)
    think: bool = False

    thinking_budget: int | None = Field(None, gt=0)
    num_ctx: int = Field(8192, gt=0)


class ExecutionConfig(_Strict):
    cpus: float = Field(2, gt=0)
    memory: str = "2g"
    code_timeout_s: float = Field(60, gt=0)
    total_run_budget_s: float = Field(900, gt=0)


class BudgetConfig(_Strict):
    max_model_calls: int = Field(10, ge=1)


class ProjectConfig(_Strict):
    project_id: str
    research_question: str = Field(min_length=1)

    claims: dict[str, str] = Field(min_length=1)
    dataset: str
    seeds: list[int] = [1, 2, 3]
    model: ModelConfig = ModelConfig()
    execution: ExecutionConfig = ExecutionConfig()
    budgets: BudgetConfig = BudgetConfig()


@dataclass(frozen=True)
class Project:
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
