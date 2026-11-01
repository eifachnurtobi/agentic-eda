"""Patterns: plain functions of a Run. Add one here and register it in PATTERNS.

Building blocks (each one saved and traced by the Run):
    run.prompt(name=None, **extra)   fill a template from prompts/ (default: the pattern's)
    run.ask(prompt_or_messages, step=..., cells=True)  -> Answer(text, cells, notebook, error)
    run.execute(answer, step=...)    -> Execution(status, cells, stdout, stderr, executed)

The run is graded on whatever `artifacts/` holds when the pattern returns. A pattern's
prompt is the template of the same name (`zero_shot` uses `prompts/zero_shot.md`); other
templates can be filled for further steps, e.g. `run.prompt("fix", ...)`.
"""

from eda_lite.run import Pattern, Run


def zero_shot(run: Run) -> None:
    """Zero-shot: one generation call, one execution, nothing fed back."""
    answer = run.ask(run.prompt())
    if answer.notebook:
        run.execute(answer)


# Basic reflection step
# Man könnte das notebook auch zuerst laufen lassen, und
# das ausgeführte notebook zurück ans llm geben (wäre besser)
def reflection(run: Run) -> None:
    # Mit dem prompt namen den prompt holen
    prompt = run.prompt("zero_shot")
    # Den prompt laufen lassen und antwort erhalten
    answer = run.ask(prompt)
    
    # Die generierte antwort in den reflection prompt hineingeben
    reflection_prompt = run.prompt("reflection", answer=answer.text)
    # Den reflection promt laufen lassen und antwort erhalten
    reflection_answer = run.ask(reflection_prompt)
    
    # Notebook laufen lassen
    if reflection_answer.notebook:
        run.execute(reflection_answer)


PATTERNS: dict[str, Pattern] = {
        "zero_shot": zero_shot,
        "reflection": reflection,
    }
