import json
from pathlib import Path

FIXTURE_PROJECT = Path(__file__).parent / "fixtures" / "tiny_project"


def notebook_cells() -> list[dict]:
    """A correct notebook for the tiny project; it relies on the engine's setup cell."""
    return [
        {"type": "markdown", "content": "# Tiny EDA"},
        {
            "type": "code",
            "content": "report('n_rows', len(df))\n"
            "report('mean_a', df.loc[df.group == 'A', 'value'].mean())\n"
            "report('mean_b', df.loc[df.group == 'B', 'value'].mean())\n"
            "z = (df.value - df.value.mean()) / df.value.std()\n"
            "report('n_outliers', (z.abs() > 3).sum())\n",
        },
        {
            "type": "code",
            "content": "df.boxplot(column='value', by='group')\nreport_plot('boxplot')",
        },
    ]


def cells_response() -> str:
    return json.dumps({"cells": notebook_cells()})
