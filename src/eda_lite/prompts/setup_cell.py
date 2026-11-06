
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("{dataset}")
_findings = {}


def report(claim_id, value):

    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (bool, int, float, str)):
        value = str(value)
    _findings[claim_id] = value
    with open("artifacts/findings.json", "w") as file:
        json.dump(_findings, file, indent=1)
    print(f"{claim_id} = {value}")


def report_plot(name, fig=None):

    fig = fig or plt.gcf()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)) or "plot"
    fig.savefig(f"artifacts/{safe}.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"plot {safe} -> artifacts/{safe}.png")
