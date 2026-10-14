Perform initial exploratory data analysis on this dataset and produce a report with code, tables, and visualizations. Think only for a few short sentences before coming up with a syntacically correct, complete and comprehensive answer.

## Research question
{research_question}

## Required claims
{claims}

## Dataset documentation
{docs}

## How to write the notebook
Respond with one JSON object: {"cells": [{"type": "code" | "markdown", "content": "..."}]}.
The cells form a Jupyter notebook and run top to bottom in one Python kernel. This setup
cell has already run before your first cell, so do not repeat it:

```python
{setup}
```

Rules:
- A code cell holds only Python code. Explanations go into markdown cells or `#` comments.
  Never add uncommented text or fenced code block syntax into code cells.
- One analysis per code cell. Keep cells short. Use report() not print().
- CSV columns arrive as text or numbers. Convert a column before you use it and assign the
  result back to the same column, e.g. `df["start"] = pd.to_datetime(df["start"])`.
- Call `report("<claim_id>", value)` for every claim under "Required claims" as soon as
  its value is computed. Compute every value with code; never type a value in.
- End every plot with `report_plot("<short_name>")`; it saves the plot to `artifacts/` and
  shows it. Do not call `plt.savefig` or `plt.show` yourself.
- Available libraries: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn. There is no
  internet access, and the dataset is read-only.
- You will not see any output, so markdown cells describe what is computed and why; they
  never state numbers.
