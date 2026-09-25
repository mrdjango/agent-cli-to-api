from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .openai_compat import ChatCompletionRequest, ChatMessage, normalize_message_content


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Any

    model_config = ConfigDict(extra="allow")


class AnthropicMessagesRequest(BaseModel):
    model: str | None = None
    messages: list[AnthropicMessage]
    system: Any = None
    stream: bool = False
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None

    model_config = ConfigDict(extra="allow")


class AnthropicCountTokensRequest(BaseModel):
    model: str | None = None
    messages: list[AnthropicMessage]
    system: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None

    model_config = ConfigDict(extra="allow")


def _data_url_from_base64_source(source: dict[str, Any]) -> str | None:
    if source.get("type") != "base64":
        return None
    media_type = source.get("media_type")
    data = source.get("data")
    if not isinstance(media_type, str) or not media_type.strip():
        return None
    if not isinstance(data, str) or not data.strip():
        return None
    return f"data:{media_type};base64,{data.strip()}"


def _system_to_text(system: Any) -> str | None:
    if isinstance(system, str):
        text = system.strip()
        return text or None
    if not isinstance(system, list):
        return None

    parts: list[str] = []
    for item in system:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _coalesce_openai_parts(parts: list[dict[str, Any]]) -> Any:
    if not parts:
        return ""
    if all(part.get("type") == "text" for part in parts):
        return "".join(str(part.get("text") or "") for part in parts)
    return parts


def _anthropic_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any] | None:
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        if isinstance(text, str):
            return _text_block(text)
        return None

    if block_type == "image":
        source = block.get("source")
        if not isinstance(source, dict):
            return None
        data_url = _data_url_from_base64_source(source)
        if not data_url:
            return None
        return {"type": "image_url", "image_url": {"url": data_url}}

    if block_type == "document":
        source = block.get("source")
        if not isinstance(source, dict):
            return None
        data_url = _data_url_from_base64_source(source)
        if not data_url:
            return None
        payload: dict[str, Any] = {
            "type": "file",
            "file": {
                "file_data": source.get("data"),
            },
        }
        title = block.get("title")
        if isinstance(title, str) and title.strip():
            payload["file"]["filename"] = title.strip()
        return payload

    return None


def _tool_result_to_chat_message(block: dict[str, Any]) -> ChatMessage | None:
    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        return None
    content = block.get("content")
    if isinstance(content, list):
        content_value: Any = _coalesce_openai_parts(
            [part for part in (_anthropic_block_to_openai_part(item) for item in content if isinstance(item, dict)) if part]
        )
    else:
        content_value = normalize_message_content(content)
    return ChatMessage(role="tool", content=content_value, tool_call_id=tool_use_id.strip())


def _tool_use_to_openai_call(block: dict[str, Any]) -> dict[str, Any] | None:
    tool_id = block.get("id")
    name = block.get("name")
    if not isinstance(tool_id, str) or not tool_id.strip():
        return None
    if not isinstance(name, str) or not name.strip():
        return None

    input_value = block.get("input")
    if not isinstance(input_value, dict):
        input_value = {}

    return {
        "id": tool_id.strip(),
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": json.dumps(input_value, ensure_ascii=False),
        },
    }


def anthropic_messages_to_chat_request(req: AnthropicMessagesRequest | AnthropicCountTokensRequest) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []

    system_text = _system_to_text(getattr(req, "system", None))
    if system_text:
        messages.append(ChatMessage(role="system", content=system_text))

    for message in req.messages:
        if isinstance(message.content, str):
            blocks = [{"type": "text", "text": message.content}]
        elif isinstance(message.content, list):
            blocks = [item for item in message.content if isinstance(item, dict)]
        else:
            blocks = []

        openai_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        pending_messages: list[ChatMessage] = []

        def _flush_user_parts() -> None:
            nonlocal openai_parts
            if openai_parts:
                pending_messages.append(
                    ChatMessage(role="user", content=_coalesce_openai_parts(openai_parts))
                )
                openai_parts = []

        for block in blocks:
            block_type = block.get("type")
            if block_type == "tool_result" and message.role == "user":
                _flush_user_parts()
                tool_msg = _tool_result_to_chat_message(block)
                if tool_msg is not None:
                    pending_messages.append(tool_msg)
                continue
            if block_type == "tool_use" and message.role == "assistant":
                tool_call = _tool_use_to_openai_call(block)
                if tool_call is not None:
                    tool_calls.append(tool_call)
                continue

            part = _anthropic_block_to_openai_part(block)
            if part is not None:
                openai_parts.append(part)

        if openai_parts or message.role == "assistant":
            if message.role == "assistant":
                content = _coalesce_openai_parts(openai_parts)
                extra: dict[str, Any] = {}
                if tool_calls:
                    extra["tool_calls"] = tool_calls
                messages.append(ChatMessage(role=message.role, content=content, **extra))
            else:
                _flush_user_parts()

        if pending_messages:
            messages.extend(pending_messages)

    extra = dict(getattr(req, "model_extra", None) or {})
    for key in ("model", "messages", "system", "stream", "max_tokens", "tools", "tool_choice"):
        extra.pop(key, None)

    tools = getattr(req, "tools", None)
    if isinstance(tools, list) and tools:
        extra["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {},
                },
            }
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str) and tool["name"].strip()
        ]
        extra["tools"] = [tool for tool in extra["tools"] if isinstance(tool, dict)]
    tool_choice = getattr(req, "tool_choice", None)
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "any":
            extra["tool_choice"] = "required"
        elif choice_type == "tool" and isinstance(tool_choice.get("name"), str):
            extra["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
    elif isinstance(tool_choice, str) and tool_choice.strip():
        extra["tool_choice"] = tool_choice.strip()

    return ChatCompletionRequest(
        model=getattr(req, "model", None),
        messages=messages,
        stream=bool(getattr(req, "stream", False)),
        max_tokens=getattr(req, "max_tokens", None),
        **extra,
    )


def _map_finish_reason_to_stop_reason(finish_reason: str | None) -> str | None:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason in {"stop", None}:
        return "end_turn"
    return finish_reason


def _openai_tool_calls_to_anthropic_blocks(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    blocks: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_args = function.get("arguments")
        if isinstance(raw_args, str):
            try:
                input_value = json.loads(raw_args)
            except Exception:
                input_value = {}
        elif isinstance(raw_args, dict):
            input_value = raw_args
        else:
            input_value = {}
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"toolu_{uuid.uuid4().hex[:16]}"
        blocks.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": name.strip(),
                "input": input_value if isinstance(input_value, dict) else {},
            }
        )
    return blocks


def openai_chat_completion_to_anthropic_message(chat: dict[str, Any]) -> dict[str, Any]:
    choices = chat.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}

    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}

    content_blocks: list[dict[str, Any]] = []
    text = normalize_message_content(message.get("content"))
    if text:
        content_blocks.append({"type": "text", "text": text})
    content_blocks.extend(_openai_tool_calls_to_anthropic_blocks(message.get("tool_calls")))

    usage = chat.get("usage") if isinstance(chat.get("usage"), dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)

    chat_id = chat.get("id")
    if isinstance(chat_id, str) and chat_id.strip():
        message_id = f"msg_{chat_id.strip().replace('-', '_')}"
    else:
        message_id = f"msg_{uuid.uuid4().hex}"

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": chat.get("model"),
        "content": content_blocks,
        "stop_reason": _map_finish_reason_to_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


def estimate_anthropic_input_tokens(req: AnthropicCountTokensRequest) -> int:
    payload = req.model_dump(exclude_none=True, mode="json")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


async def _iter_openai_sse_data(chunks: AsyncIterator[bytes | str]) -> AsyncIterator[str]:
    buffer = ""
    async for chunk in chunks:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="ignore")
        else:
            buffer += chunk

        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = [
                line[len("data:") :].lstrip()
                for line in raw_event.splitlines()
                if line.startswith("data:")
            ]
            if data_lines:
                yield "\n".join(data_lines)

    if buffer.strip():
        data_lines = [
            line[len("data:") :].lstrip()
            for line in buffer.splitlines()
            if line.startswith("data:")
        ]
        if data_lines:
            yield "\n".join(data_lines)


def _anthropic_sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def openai_stream_to_anthropic_events(
    chunks: AsyncIterator[bytes | str],
    *,
    model: str | None,
) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex}"
    text_block_open = False
    next_index = 0
    stop_reason = "end_turn"
    usage: dict[str, int] = {"output_tokens": 0}

    yield _anthropic_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    async for data in _iter_openai_sse_data(chunks):
        if not data or data.strip() == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue
        # Trailing `stream_options.include_usage` chunk: usage is only known once generation ends,
        # so it's reported on `message_delta` (which Anthropic allows to carry input tokens too).
        chunk_usage = obj.get("usage")
        if isinstance(chunk_usage, dict):
            usage = {
                "input_tokens": int(chunk_usage.get("prompt_tokens") or 0),
                "output_tokens": int(chunk_usage.get("completion_tokens") or 0),
            }
        choices = obj.get("choices") or []
        choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(choice, dict):
            continue

        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}

        text = delta.get("content")
        if isinstance(text, str) and text:
            if not text_block_open:
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": next_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                text_block_open = True
            yield _anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": next_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            if text_block_open:
                yield _anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": next_index},
                )
                text_block_open = False
                next_index += 1
            for block in _openai_tool_calls_to_anthropic_blocks(tool_calls):
                block_index = next_index
                next_index += 1
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": block,
                    },
                )
                yield _anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block_index},
                )
            stop_reason = "tool_use"

        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            stop_reason = _map_finish_reason_to_stop_reason(finish_reason) or stop_reason

    if text_block_open:
        yield _anthropic_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": next_index},
        )

    yield _anthropic_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        },
    )
    yield _anthropic_sse_event("message_stop", {"type": "message_stop"})
