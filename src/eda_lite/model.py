"""The model: one Ollama chat request, and the cells contract a notebook answer must meet."""

import json

import requests

from eda_lite.config import ModelConfig

# The answer format for notebook generation: {"cells": [{"type", "content"}, ...]}.
CELLS_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["code", "markdown"]},
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        }
    },
    "required": ["cells"],
}


class ModelError(Exception):
    """No usable answer: timeout, HTTP failure, or (kind='malformed') an answer off contract."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # timeout | request | malformed


def chat(
    messages: list[dict],
    config: ModelConfig,
    *,
    seed: int,
    schema: dict | None,
    timeout_s: float,
) -> tuple[dict, dict]:
    """Send one chat request; returns (request payload, response body), both kept in full."""
    # Ollama takes the seed, but does not promise identical output across machines.
    options = {"temperature": config.temperature, "num_ctx": config.num_ctx, "seed": seed}
    payload = {
        "model": config.name,
        "messages": messages,
        "stream": False,
        "think": config.think,
        "options": options,
    }
    if schema is not None:
        payload["format"] = schema  # Ollama constrains the output to it
    try:
        reply = requests.post(f"{config.base_url}/api/chat", json=payload, timeout=timeout_s)
        reply.raise_for_status()
    except requests.Timeout:
        raise ModelError("timeout", f"no response within {timeout_s:.0f}s") from None
    except requests.RequestException as error:
        raise ModelError("request", f"Ollama request failed: {error}") from error
    return payload, reply.json()


def parse_cells(text: str) -> list[dict]:
    """The cell records in a notebook answer; raises ModelError('malformed') with the reason."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ModelError("malformed", f"response is not valid JSON: {error}") from None
    cells = parsed.get("cells") if isinstance(parsed, dict) else None
    if not isinstance(cells, list) or not cells:
        raise ModelError("malformed", "response has no non-empty 'cells' array")
    for index, cell in enumerate(cells):
        if not (
            isinstance(cell, dict)
            and cell.get("type") in ("code", "markdown")
            and isinstance(cell.get("content"), str)
        ):
            raise ModelError("malformed", f"cell {index} is not a valid cell record")
    return [{"type": cell["type"], "content": cell["content"]} for cell in cells]
