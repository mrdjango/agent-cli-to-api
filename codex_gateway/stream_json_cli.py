from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from .openai_compat import normalize_message_content


@dataclass(frozen=True)
class StreamJsonResult:
    text: str
    usage: dict[str, int] | None


class TextAssembler:
    """
    Some CLIs emit partial deltas and later emit a full final message.
    This helper turns mixed streams into clean deltas (and a final assembled text).
    """

    def __init__(self) -> None:
        self.text = ""

    def feed(self, incoming: str) -> str:
        incoming = incoming or ""
        if not incoming:
            return ""
        if incoming == self.text:
            return ""
        if incoming.startswith(self.text):
            delta = incoming[len(self.text) :]
            self.text = incoming
            return delta
        # Fallback: treat as delta chunk.
        self.text += incoming
        return incoming


async def iter_stream_json_events(
    *,
    cmd: list[str],
    env: dict[str, str] | None,
    timeout_seconds: int,
    stream_limit: int,
    event_callback: Callable[[dict], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
    cwd: str | None = None,
) -> AsyncIterator[dict]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=stream_limit,
        env=env or os.environ.copy(),
        cwd=cwd,
    )

    stderr_buf: bytearray = bytearray()
    last_hint: str | None = None

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        text_buf = ""
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                if stderr_callback and text_buf.strip():
                    for line in text_buf.splitlines():
                        line = line.strip()
                        if line:
                            stderr_callback(line)
                return
            stderr_buf.extend(chunk)
            if len(stderr_buf) > 64_000:
                del stderr_buf[:-64_000]
            if stderr_callback:
                text_buf += chunk.decode(errors="ignore")
                if "\n" in text_buf:
                    lines = text_buf.splitlines(keepends=False)
                    if not text_buf.endswith("\n"):
                        text_buf = lines.pop() if lines else ""
                    else:
                        text_buf = ""
                    for line in lines:
                        line = line.strip()
                        if line:
                            stderr_callback(line)

    drain_task = asyncio.create_task(_drain_stderr())
    try:
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout not available")

        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_seconds)
            except (asyncio.TimeoutError, TimeoutError):
                proc.kill()
                await proc.wait()
                raise
            except ValueError as e:
                proc.kill()
                await proc.wait()
                msg = bytes(stderr_buf).decode(errors="ignore").strip()
                hint = (
                    f"subprocess output line exceeded asyncio stream limit ({stream_limit} bytes). "
                    "Increase CODEX_SUBPROCESS_STREAM_LIMIT."
                )
                raise RuntimeError(f"{hint}\n{msg}".strip()) from e

            if not line:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw.decode(errors="ignore"))
            except Exception:
                # Some CLIs print non-JSON lines even in stream-json mode.
                continue
            if evt.get("type") == "result" and isinstance(evt.get("result"), str) and evt.get("result"):
                last_hint = str(evt.get("result")).strip() or last_hint
            if evt.get("type") == "error" and isinstance(evt.get("message"), str) and evt.get("message"):
                last_hint = str(evt.get("message")).strip() or last_hint
            if event_callback:
                event_callback(evt)
            yield evt

        rc = await proc.wait()
        await drain_task
        if rc != 0:
            msg = bytes(stderr_buf).decode(errors="ignore").strip()
            raise RuntimeError(msg or last_hint or f"subprocess failed: {rc}")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not drain_task.done():
            drain_task.cancel()


def extract_text_from_content(content: object) -> str:
    return normalize_message_content(content)


def extract_cursor_agent_delta(evt: dict, assembler: TextAssembler) -> str:
    if evt.get("type") != "assistant":
        return ""
    message = evt.get("message") or {}
    if not isinstance(message, dict):
        return ""
    incoming = extract_text_from_content(message.get("content"))
    return assembler.feed(incoming)


def extract_claude_delta(evt: dict, assembler: TextAssembler) -> str:
    if evt.get("type") != "assistant":
        return ""
    message = evt.get("message") or {}
    if not isinstance(message, dict):
        return ""
    incoming = extract_text_from_content(message.get("content"))
    return assembler.feed(incoming)


def extract_gemini_delta(evt: dict, assembler: TextAssembler) -> str:
    if evt.get("type") != "message":
        return ""
    if evt.get("role") != "assistant":
        return ""
    incoming = extract_text_from_content(evt.get("content"))
    return assembler.feed(incoming)


def extract_usage_from_claude_result(evt: dict) -> dict[str, int] | None:
    if evt.get("type") != "result":
        return None
    usage = evt.get("usage")
    if not isinstance(usage, dict):
        return None
    in_tokens = int(usage.get("input_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
    }


def extract_usage_from_gemini_result(evt: dict) -> dict[str, int] | None:
    if evt.get("type") != "result":
        return None
    stats = evt.get("stats")
    if not isinstance(stats, dict):
        return None
    in_tokens = int(stats.get("input_tokens") or 0)
    out_tokens = int(stats.get("output_tokens") or 0)
    total = int(stats.get("total_tokens") or (in_tokens + out_tokens))
    return {
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": total,
    }

