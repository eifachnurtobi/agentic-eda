"""The model: one Ollama chat request, and the cells contract a notebook answer must meet."""

import json
import time

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
    """No usable answer: timeout, HTTP failure, or (kind='malformed') an answer off contract.
    Carries what was sent and what arrived before the failure, so both can be logged."""

    def __init__(self, kind: str, message: str, payload: dict | None = None, partial=None):
        super().__init__(message)
        self.kind = kind  # timeout | request | malformed
        self.payload = payload  # the request as sent
        self.partial = partial  # the response body received so far (thinking, content)


# Budget forcing (Qwen's recipe for a thinking budget): once the reasoning reaches the
# budget, it is cut, closed with this sentence and `</think>`, and the model answers from there.
BUDGET_NOTE = (
    "Considering the limited time, I have to give the solution based on the thinking directly now."
)


# Counts and durations Ollama reports in a request's final chunk (durations in nanoseconds).
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
    """Send one chat request; returns (request payload, response body), both kept in full.

    The answer is streamed, so a call that runs out of time still yields everything the
    model produced until then, its reasoning included. `timeout_s` bounds the whole call,
    including a continuation after the thinking budget (see BUDGET_NOTE)."""
    # Ollama takes the seed, but does not promise identical output across machines.
    options = {"temperature": config.temperature, "num_ctx": config.num_ctx, "seed": seed}
    payload = {
        "model": config.name,
        "messages": messages,
        "stream": True,
        "think": config.think,
        "options": options,
    }
    if schema is not None:
        payload["format"] = schema  # Ollama constrains the output to it
    budget = config.thinking_budget if config.think else None
    deadline = time.monotonic() + timeout_s
    thinking, content, final, continuation = [], [], None, None
    phases: list[dict] = []  # one entry per request: its time and Ollama's counts

    def phase(endpoint: str, started: float, last: dict | None = None, **fields) -> None:
        stats = {key: last[key] for key in OLLAMA_STATS if key in (last or {})}
        seconds = round(time.monotonic() - started, 3)
        phases.append({"endpoint": endpoint, "seconds": seconds, **stats, **fields})

    def body(last: dict | None) -> dict:
        """The same shape as a non-streamed chat reply: the last chunk, with the whole message."""
        message = {"role": "assistant", "content": "".join(content)}
        if thinking:
            message["thinking"] = "".join(thinking)
        reply = {**(last or {"done": False}), "message": message, "phases": phases}
        reply.pop("response", None)  # /api/generate's chunk field; the text is in `message`
        reply.pop("context", None)  # /api/generate's token ids: large and of no use here
        if continuation is not None:
            reply |= {"thinking_truncated": True, "continuation": continuation}
            if last is not None:  # the cut reasoning was generated too: about a token per chunk
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
                break  # closes the stream, which stops the generation
        if final is not None:
            phase("chat", started, final)
        else:  # the budget is used up: close the reasoning, let the model answer
            # Ollama reports counts only in the final chunk, which a cut stream never sends.
            phase("chat", started, thinking_tokens=len(thinking), stopped_by="thinking_budget")
            endpoint, started = "generate", time.monotonic()
            continuation = {
                "model": config.name,
                "prompt": _raw_prompt(messages, "".join(thinking)),
                "raw": True,  # sent as is, so the closed reasoning is not re-templated
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
    """The JSON chunks of one streamed Ollama request, up to the one marked done."""
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as reply:
            reply.raise_for_status()
            for line in reply.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise _StreamFailure("request", f"Ollama error: {chunk['error']}")
                yield chunk
                if chunk.get("done"):
                    return
                if time.monotonic() > deadline:
                    raise _StreamFailure("timeout", f"no complete response within {timeout_s:.0f}s")
    except requests.Timeout:
        raise _StreamFailure("timeout", f"no response within {timeout_s:.0f}s") from None
    except requests.RequestException as error:
        raise _StreamFailure("request", f"Ollama request failed: {error}") from error
    raise _StreamFailure("request", "the response stream ended before the answer was complete")


def _raw_prompt(messages: list[dict], thinking: str) -> str:
    """The conversation in Qwen's chat template (ChatML), then the assistant's turn with its
    reasoning cut off and closed, so the model continues with the answer itself."""
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    reasoning = thinking.strip()
    return f"{turns}<|im_start|>assistant\n{reasoning}\n\n{BUDGET_NOTE}\n</think>\n\n"


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
