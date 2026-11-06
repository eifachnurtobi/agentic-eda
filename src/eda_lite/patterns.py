from eda_lite.run import Pattern, Run


def zero_shot(run: Run) -> None:
    answer = run.ask(run.prompt(), step="generate")
    if answer.notebook:
        run.execute(answer, step="execute")


def reflection(run: Run) -> None:
    draft = run.ask(run.prompt("zero_shot"), step="generate")
    if not draft.text:
        return

    reflection_prompt = run.prompt("reflection", answer=draft.text)
    revised = run.ask(reflection_prompt, step="reflect")
    if revised.notebook:
        run.execute(revised, step="execute")


PATTERNS: dict[str, Pattern] = {
    "zero_shot": zero_shot,
    "reflection": reflection,
}
