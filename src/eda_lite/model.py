import json
import time

import requests

from eda_lite.config import ModelConfig

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
    def __init__(self, kind: str, message: str, payload: dict | None = None, partial=None):
        super().__init__(message)
        self.kind = kind
        self.payload = payload
        self.partial = partial


BUDGET_NOTE = (
    "Considering the limited time, I have to give the solution based on the thinking directly now."
)


OLLAMA_STATS = (
    "prompt_eval_count",
    "eval_count",
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)


class _StreamFailure(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def chat(
    messages: list[dict],
    config: ModelConfig,
    *,
    seed: int,
    schema: dict | None,
    timeout_s: float,
) -> tuple[dict, dict]:

    options = {"temperature": config.temperature, "num_ctx": config.num_ctx, "seed": seed}
    payload = {
        "model": config.name,
        "messages": messages,
        "stream": True,
        "think": config.think,
        "options": options,
    }
    if schema is not None:
        payload["format"] = schema
    budget = config.thinking_budget if config.think else None
    deadline = time.monotonic() + timeout_s
    thinking, content, final, continuation = [], [], None, None
    phases: list[dict] = []

    def phase(endpoint: str, started: float, last: dict | None = None, **fields) -> None:
        stats = {key: last[key] for key in OLLAMA_STATS if key in (last or {})}
        seconds = round(time.monotonic() - started, 3)
        phases.append({"endpoint": endpoint, "seconds": seconds, **stats, **fields})

    def body(last: dict | None) -> dict:

        message = {"role": "assistant", "content": "".join(content)}
        if thinking:
            message["thinking"] = "".join(thinking)
        reply = {**(last or {"done": False}), "message": message, "phases": phases}
        reply.pop("response", None)
        reply.pop("context", None)
        if continuation is not None:
            reply |= {"thinking_truncated": True, "continuation": continuation}
            if last is not None:
                reply["eval_count"] = (last.get("eval_count") or 0) + len(thinking)
        return reply

    endpoint, started = "chat", time.monotonic()
    try:
        url = f"{config.base_url}/api/chat"
        for chunk in _chunks(url, payload, deadline, timeout_s):
            message = chunk.get("message", {})
            if message.get("thinking"):
                thinking.append(message["thinking"])
            if message.get("content"):
                content.append(message["content"])
            if chunk.get("done"):
                final = chunk
            elif budget and not content and len(thinking) >= budget:
                break
        if final is not None:
            phase("chat", started, final)
        else:
            phase("chat", started, thinking_tokens=len(thinking), stopped_by="thinking_budget")
            endpoint, started = "generate", time.monotonic()
            continuation = {
                "model": config.name,
                "prompt": _raw_prompt(messages, "".join(thinking)),
                "raw": True,
                "stream": True,
                "options": options,
            }
            if schema is not None:
                continuation["format"] = schema
            url = f"{config.base_url}/api/generate"
            for chunk in _chunks(url, continuation, deadline, timeout_s):
                if chunk.get("response"):
                    content.append(chunk["response"])
                if chunk.get("done"):
                    final = chunk
            phase("generate", started, final)
    except _StreamFailure as failure:
        phase(endpoint, started, stopped_by=failure.kind)
        raise ModelError(failure.kind, str(failure), payload, body(None)) from None
    return payload, body(final)


def _chunks(url: str, payload: dict, deadline: float, timeout_s: float):

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as reply:
            reply.raise_for_status()
            for line in reply.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    message = str(chunk["error"])
                    if "model" in message.lower() and "not found" in message.lower():
                        model = payload["model"]
                        message = f"Ollama model '{model}' is missing; run `ollama pull {model}`"
                    raise _StreamFailure("request", message)
                yield chunk
                if chunk.get("done"):
                    return
                if time.monotonic() > deadline:
                    raise _StreamFailure("timeout", f"no complete response within {timeout_s:.0f}s")
    except requests.Timeout:
        raise _StreamFailure("timeout", f"no response within {timeout_s:.0f}s") from None
    except requests.ConnectionError:
        server = url.split("/api/", 1)[0]
        raise _StreamFailure("request", f"Ollama is not running or reachable at {server}") from None
    except requests.RequestException as error:
        raise _StreamFailure("request", f"Ollama request failed: {error}") from error
    raise _StreamFailure("request", "the response stream ended before the answer was complete")


def _raw_prompt(messages: list[dict], thinking: str) -> str:

    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    reasoning = thinking.strip()
    return f"{turns}<|im_start|>assistant\n<think>\n{reasoning}\n\n{BUDGET_NOTE}\n</think>\n\n"


def parse_cells(text: str) -> list[dict]:

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
