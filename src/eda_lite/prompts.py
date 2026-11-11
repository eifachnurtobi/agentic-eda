import hashlib
import re
from pathlib import Path

from eda_lite.config import Project

PROMPTS_DIR = Path(__file__).parent / "prompts"
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def find_prompt(name: str) -> Path:

    path = PROMPTS_DIR / f"{name.removesuffix('.md')}.md"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("*.md")))
        raise FileNotFoundError(f"no prompt '{name}' in {PROMPTS_DIR}; known: {known}")
    return path


def fill(template: str, **fields) -> str:

    return PLACEHOLDER.sub(lambda m: str(fields.get(m[1], m[0])), template)


def setup_code(project: Project) -> str:

    template = (PROMPTS_DIR / "setup_cell.py").read_text()
    return fill(template, dataset=f"data/{project.config.dataset}").strip()


def project_fields(project: Project) -> dict[str, str]:

    config = project.config
    claims = "\n".join(f"- `{claim}`: {text}" for claim, text in config.claims.items())
    docs = "\n\n".join(path.read_text().strip() for path in sorted(project.docs_dir.rglob("*.md")))
    return {
        "research_question": config.research_question,
        "claims": claims,
        "docs": docs,
        "setup": setup_code(project),
    }


def render(name: str, project: Project, **extra) -> str:
    return fill(find_prompt(name).read_text(), **project_fields(project), **extra)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def prompt_sha(name: str) -> str:

    digest = hashlib.sha256(find_prompt(name).read_bytes())
    digest.update((PROMPTS_DIR / "setup_cell.py").read_bytes())
    return digest.hexdigest()[:12]
