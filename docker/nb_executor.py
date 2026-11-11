import argparse
import json
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

EXIT_CODES = {"ok": 0, "error": 1, "timeout": 2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--cell-timeout", type=float, required=True)
    parser.add_argument("--cwd", default="/work")
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.cell_timeout,
        kernel_name="python3",
        resources={"metadata": {"path": args.cwd}},
    )

    status = "ok"
    with open(args.out_dir / "journal.jsonl", "a") as journal, client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            entry = {"cell_id": cell.id, "status": "ok", "error": None}
            started = time.monotonic()
            try:
                client.execute_cell(cell, index)
            except CellTimeoutError:
                entry["status"] = "timeout"
            except CellExecutionError as error:
                entry["status"] = "error"
                entry["error"] = {"name": error.ename, "message": error.evalue}
            entry["duration_s"] = round(time.monotonic() - started, 3)
            entry["outputs"] = [output.output_type for output in cell.get("outputs", [])]

            journal.write(json.dumps(entry) + "\n")
            journal.flush()
            nbformat.write(notebook, args.out_dir / "executed.ipynb")
            if entry["status"] == "timeout":
                status = "timeout"
                break
            if entry["status"] == "error":
                status = "error"
    return EXIT_CODES[status]


if __name__ == "__main__":
    raise SystemExit(main())
