# eda-lite

The lean version of eda-agent for fast prototyping. It uses the same projects, the same
sandbox setup and the same answer format. It still runs several seeds and grades every run
automatically, and it logs everything. There is less machinery around that, and the prompts
are plain files.

It is a project of its own: its own environment, its own sandbox image (`eda-lite:latest`,
from `compose.yaml` and `docker/` here), its own projects and its own runs. All paths are
fixed, and the CLI takes no paths:

| What | Where |
|---|---|
| projects | `projects/<project_id>/` (`stix_flares`, `stix_flares_simple`); the CLI takes the ID |
| prompts | `src/eda_lite/prompts/<pattern>.md`: each pattern uses its own template |
| runs | `runs/` |

```bash
cd eda-lite
uv sync                                                   # Python 3.12; Docker and Ollama running
uv run eda-lite prompt stix_flares_simple --pattern zero_shot     # the filled prompt, no model call
uv run eda-lite run stix_flares_simple --pattern zero_shot        # one run per seed in project.yaml
uv run eda-lite run stix_flares_simple --pattern zero_shot --seeds 1 2
uv run eda-lite trace [call] [--seed N]                   # timelines of a call's seeds (default: latest)
uv run eda-lite compare [call]                            # latest call's prompt vs. the others
uv run pytest                                             # unit tests; `-m docker` for end to end
```

## Where to change things

| To change | Edit |
|---|---|
| what the model is asked | `src/eda_lite/prompts/<pattern>.md`. Placeholders: `{research_question}` `{claims}` `{docs}` `{setup}` |
| the notebook's first cell (`df`, `report()`, `report_plot()`) | `src/eda_lite/prompts/setup_cell.py` |
| try a different prompt | edit the template: `compare` tells old and new apart by its hash. To keep both runnable side by side, add a pattern and a template of the same name |
| the agent's behaviour (reflection, retries, ...) | add a function to `src/eda_lite/patterns.py` |
| model, timeouts, seeds | `projects/<id>/project.yaml` (the only place; `--seeds` picks a subset for one call) |
| how long the model thinks | `model.thinking_budget` in project.yaml (with `think: true`): after that many reasoning tokens the reasoning is cut, closed with `</think>`, and the model answers from there (budget forcing). It uses Qwen's chat template, so it is for qwen-family models. Ollama itself has no thinking budget for qwen3. |
| the claims and their answers | `projects/<id>/project.yaml` (`claims:`) and `projects/<id>/evaluation_spec/answers.yaml` (see below) |

A pattern is a plain function of a `Run`, registered in `PATTERNS`. Its prompt is the template
with the same name (`zero_shot` → `prompts/zero_shot.md`); further steps can fill other
templates. For example:

```python
def reflect(run: Run) -> None:
    answer = run.ask(run.prompt())
    if not answer.notebook:
        return
    result = run.execute(answer)
    if result.status != "ok":
        critique = run.prompt("fix", errors=..., notebook=...)  # prompts/fix.md
        answer = run.ask(critique, step="revise")
        if answer.notebook:
            run.execute(answer, step="revise")  # the last execution is graded
```

`run.ask` and `run.execute` save and trace everything themselves. With `cells=False`,
`run.ask` returns free text, for example a critique. For a system prompt or a conversation,
pass `run.ask` a list of messages instead of a string.

## A run folder

Each `eda-lite run` call gets one folder, with one subfolder per seed:

```text
runs/<time>_<project>_<pattern>/
  invocation.json        what all seeds share: command, project, pattern, prompt hash,
                         model, git, dataset + answer-key hashes, sandbox image; score per seed
  config.yaml            project.yaml with every default filled in
  seed_<N>/
    summary.md           score, first error with the failing cell's code, wrong claims, plots
    result.ipynb         the latest executed notebook, with outputs
    artifacts/           the files it wrote: findings.json, plots
    evaluation/
      checks.json        the grade: score, and every claim as pass / fail / missing
      metrics.json       tokens, time, cell success rate, first error
    logs/
      run.json           this seed: id, seed, times, status, error, score
      trace.jsonl        one line per model call, cell, execution, grade
      prompt_NNN.md      the exact text of model call NNN
      thinking_NNN.md    its reasoning, when the model thinks (kept even if the call timed out)
      continuation_NNN.md  if the reasoning was cut at `thinking_budget`: the exact text the
                         model continued from (prompt, cut reasoning, injected `</think>`)
      call_NNN.json      its full request and response (or error)
      response_NNN.*     the answer: .ipynb before it ran, .md as free text, .txt if unparsable
      exec_NNN.*.txt     what execution NNN printed outside the notebook, if anything
```

**Comparing prompts.** `eda-lite compare [call]` (default: the latest; otherwise part of a
folder name) prints, to the console only, how that call's prompt does against every other
prompt tried with the same pattern, an identical `config.yaml` (project, claims, model
settings such as `think`, temperature, `num_ctx` and `thinking_budget`, execution limits),
the same dataset and the same answer key. Calls that differ in any of these are left out, so
only the prompt varies between rows, and nothing needs re-grading. The output has three
parts:

- **Calls:** one row per call, oldest first: its prompt, its seeds, and stats over its seeds.
- **Prompts:** one row per prompt text, best mean score first, with stats over every seed run
  with that prompt and its first errors. The prompt is named by its hash, which covers the
  pattern's template and the setup cell, so an edit in place shows up as a new row.
- **Overview:** what all rows share (project, pattern, model, thinking, config hash, dataset
  and answer key), and the
  call you asked about with its prompt.

The stats are score (mean ± sd, and min – max), perfect score (`k/n`: k of the n seeds got
every claim right), and the mean input tokens, output tokens and seconds per seed. The call
you asked about is marked `→`. `eda-lite run` prints the same comparison when it finishes.

`eda-lite trace [call]` shows the timeline of every seed of a call (default: the latest;
otherwise part of its folder name, as for `compare`); `--seed N` shows only that seed.

After changing `docker/Dockerfile` or `docker/requirements.lock`, rebuild with
`docker compose build` (from this folder).

## A project and its grading

```text
projects/<project_id>/
  project.yaml               research_question, dataset, claims (+ optional seeds, model, execution)
  data/<dataset>             the CSV, read-only in the sandbox
  docs/*.md                  pasted into the prompt
  evaluation_spec/           EVALUATOR-ONLY: never in a prompt, never in the sandbox
    answers.yaml             claim_id: expected value, one line per claim
    reference_analysis.py    optional: recomputes the answers with pandas and checks the file
```

```yaml
# project.yaml                                   # evaluation_spec/answers.yaml
claims:                                          n_rows: 33076
  n_rows: Number of rows in the file.            most_common_estimated_class: C1
  most_common_estimated_class: The most ...      mean_solo_distance_au: 0.659633
  mean_solo_distance_au: Mean of the column ...
```

The notebook reports each claim with `report("<claim_id>", value)` (defined in the setup
cell), which writes `artifacts/findings.json` as `{claim_id: value}`. Plots go through
`report_plot("<name>")`, which saves `artifacts/<name>.png` and shows the plot inline. Each claim is then
**pass**, **fail** or **missing**, by one rule: text and whole numbers must match exactly,
and other numbers must be within 0.1% (relative). The score is the share of claims that pass.
`answers.yaml` must name exactly the claims in `project.yaml`, or no run starts. To check it
against the data, run `uv run python projects/<id>/evaluation_spec/reference_analysis.py`.

## What eda-lite leaves out, compared with eda-agent

Fresh re-execution, preflight, the validated event vocabulary, input hashes for docs and
config, comparison runs and paired differences, and `model.structured_output` (the schema is
always sent), per-claim tolerances, grading gates and the check-by-eye list. The run budgets
from `project.yaml`
(`total_run_budget_s`, `max_model_calls`) are still enforced, in a few lines inside `Run`.
