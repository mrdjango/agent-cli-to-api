"""
Gemini API (generativelanguage.googleapis.com) compatibility.

Converts `models/{model}:generateContent` / `:streamGenerateContent` requests into the gateway's
internal chat request, and chat completions (JSON or SSE) back into `GenerateContentResponse`s,
so Google's `google-genai` SDKs and other Gemini API clients can use any gateway provider.

Request fields may arrive camelCase (REST) or snake_case (proto JSON); both are accepted.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .anthropic_compat import _iter_openai_sse_data
from .openai_compat import ChatCompletionRequest, ChatMessage, normalize_message_content

GEMINI_ACTIONS = {"generateContent", "streamGenerateContent", "countTokens"}

_STATUS_NAMES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    409: "ABORTED",
    413: "INVALID_ARGUMENT",
    422: "INVALID_ARGUMENT",
    429: "RESOURCE_EXHAUSTED",
    499: "CANCELLED",
    501: "UNIMPLEMENTED",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}


class GeminiRequestError(ValueError):
    pass


def gemini_error_body(message: str, status_code: int) -> dict[str, Any]:
    status = _STATUS_NAMES.get(status_code) or ("INTERNAL" if status_code >= 500 else "FAILED_PRECONDITION")
    return {"error": {"code": status_code, "message": message, "status": status}}


def _get(obj: Any, camel: str, snake: str | None = None) -> Any:
    if not isinstance(obj, dict):
        return None
    if camel in obj:
        return obj[camel]
    return obj.get(snake) if snake else None


def split_model_action(model_action: str) -> tuple[str, str]:
    """`gemini-3.8-flash:generateContent` -> (model, action). Model names may contain ':' (agy:...)."""
    model, sep, action = (model_action or "").rpartition(":")
    if not sep or action not in GEMINI_ACTIONS:
        raise GeminiRequestError(f"Unsupported method '{model_action}'.")
    model = model.removeprefix("models/").strip()
    if not model:
        raise GeminiRequestError("A model is required in the URL, e.g. models/gemini-3.8-flash-medium:generateContent.")
    return model, action


def _content_text(content: Any) -> str:
    """Text of a `Content` (or a bare string / list of parts), for system instructions."""
    if isinstance(content, str):
        return content.strip()
    parts = _get(content, "parts") if isinstance(content, dict) else content
    if isinstance(parts, dict):
        parts = [parts]
    if not isinstance(parts, list):
        return ""
    return "\n\n".join(
        p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip()
    ).strip()


def _inline_data_part(blob: dict[str, Any]) -> dict[str, Any]:
    mime = str(_get(blob, "mimeType", "mime_type") or "application/octet-stream").strip()
    data = str(_get(blob, "data") or "").strip()
    if not data:
        raise GeminiRequestError("inlineData.data is empty.")
    data_url = f"data:{mime};base64,{data}"
    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": data_url}}
    file_part: dict[str, Any] = {"file_data": data_url}
    name = _get(blob, "displayName", "display_name")
    if isinstance(name, str) and name.strip():
        file_part["filename"] = name.strip()
    return {"type": "file", "file": file_part}


def _coalesce(parts: list[dict[str, Any]]) -> Any:
    if all(p.get("type") == "text" for p in parts):
        return "".join(str(p.get("text") or "") for p in parts)
    return parts


def _lower_schema_types(schema: Any) -> Any:
    """Gemini `Schema` uses OpenAPI enum names (OBJECT, STRING); JSON Schema wants lowercase."""
    if isinstance(schema, list):
        return [_lower_schema_types(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.lower()
        elif key == "type" and isinstance(value, list):
            out[key] = [v.lower() if isinstance(v, str) else v for v in value]
        elif key == "properties" and isinstance(value, dict):
            out[key] = {name: _lower_schema_types(prop) for name, prop in value.items()}
        else:
            out[key] = _lower_schema_types(value)
    return out


def _tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        for decl in _get(tool, "functionDeclarations", "function_declarations") or []:
            if not isinstance(decl, dict) or not isinstance(decl.get("name"), str) or not decl["name"].strip():
                continue
            params = _get(decl, "parametersJsonSchema", "parameters_json_schema")
            if not isinstance(params, dict):
                params = _lower_schema_types(decl.get("parameters")) if isinstance(decl.get("parameters"), dict) else {}
            fn: dict[str, Any] = {"name": decl["name"].strip(), "parameters": params or {"type": "object", "properties": {}}}
            if isinstance(decl.get("description"), str):
                fn["description"] = decl["description"]
            out.append({"type": "function", "function": fn})
        # Gemini's built-in Google Search tool is a web search opt-in, as on the other APIs.
        if any(k in tool for k in ("googleSearch", "google_search", "googleSearchRetrieval", "google_search_retrieval")):
            out.append({"type": "web_search"})
    return out


def _tool_choice(tool_config: Any) -> Any:
    cfg = _get(tool_config, "functionCallingConfig", "function_calling_config")
    if not isinstance(cfg, dict):
        return None
    mode = str(cfg.get("mode") or "").upper()
    allowed = _get(cfg, "allowedFunctionNames", "allowed_function_names") or []
    if mode == "NONE":
        return "none"
    if mode in {"ANY", "VALIDATED"}:
        if isinstance(allowed, list) and len(allowed) == 1 and isinstance(allowed[0], str):
            return {"type": "function", "function": {"name": allowed[0]}}
        return "required"
    if mode == "AUTO":
        return "auto"
    return None


def _thinking_effort(thinking: Any) -> str | None:
    if not isinstance(thinking, dict):
        return None
    level = _get(thinking, "thinkingLevel", "thinking_level")
    if isinstance(level, str) and level.strip():
        level = level.strip().lower()
        return "low" if level == "minimal" else level
    budget = _get(thinking, "thinkingBudget", "thinking_budget")
    if isinstance(budget, int) and budget >= 0:
        if budget <= 1024:
            return "low"
        return "medium" if budget <= 8192 else "high"
    return None  # -1 (dynamic) or unset: provider default


def gemini_request_to_chat_request(body: dict[str, Any], *, model: str, stream: bool) -> ChatCompletionRequest:
    if not isinstance(body, dict):
        raise GeminiRequestError("Request body must be a JSON object.")
    contents = _get(body, "contents")
    if isinstance(contents, dict):
        contents = [contents]
    if not isinstance(contents, list) or not contents:
        raise GeminiRequestError("contents is required.")

    messages: list[ChatMessage] = []
    system_text = _content_text(_get(body, "systemInstruction", "system_instruction"))
    if system_text:
        messages.append(ChatMessage(role="system", content=system_text))

    # Gemini pairs a functionResponse with its functionCall by id when present, else by name.
    pending_ids: dict[str, list[str]] = {}
    call_seq = 0

    for content in contents:
        if isinstance(content, str):
            content = {"role": "user", "parts": [{"text": content}]}
        if not isinstance(content, dict):
            continue
        role = str(content.get("role") or "user").lower()
        parts = content.get("parts") or []
        if isinstance(parts, dict):
            parts = [parts]

        chat_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_messages: list[ChatMessage] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("thought") is True:
                continue  # earlier turns' thought summaries are not conversation content
            if isinstance(part.get("text"), str):
                chat_parts.append({"type": "text", "text": part["text"]})
                continue
            blob = _get(part, "inlineData", "inline_data")
            if isinstance(blob, dict):
                chat_parts.append(_inline_data_part(blob))
                continue
            if isinstance(_get(part, "fileData", "file_data"), dict):
                raise GeminiRequestError("fileData (uploaded file URIs) is not supported; send the file as inlineData.")
            call = _get(part, "functionCall", "function_call")
            if isinstance(call, dict) and isinstance(call.get("name"), str):
                call_seq += 1
                call_id = str(call.get("id") or f"call_{call_seq}")
                pending_ids.setdefault(call["name"], []).append(call_id)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False)},
                    }
                )
                continue
            resp = _get(part, "functionResponse", "function_response")
            if isinstance(resp, dict) and isinstance(resp.get("name"), str):
                queue = pending_ids.get(resp["name"]) or []
                call_id = resp.get("id")
                if isinstance(call_id, str) and call_id in queue:
                    queue.remove(call_id)
                elif queue:
                    call_id = queue.pop(0)
                else:
                    call_seq += 1
                    call_id = str(call_id or f"call_{call_seq}")
                tool_messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(resp.get("response") or {}, ensure_ascii=False),
                        tool_call_id=call_id,
                    )
                )

        if role == "model":
            if chat_parts or tool_calls:
                extra: dict[str, Any] = {"tool_calls": tool_calls} if tool_calls else {}
                messages.append(ChatMessage(role="assistant", content=_coalesce(chat_parts) if chat_parts else "", **extra))
        else:
            messages.extend(tool_messages)
            if chat_parts:
                messages.append(ChatMessage(role="user", content=_coalesce(chat_parts)))

    if not any(m.role in {"user", "tool"} for m in messages):
        raise GeminiRequestError("contents must include at least one user turn.")

    extra: dict[str, Any] = {}
    gen = _get(body, "generationConfig", "generation_config")
    max_tokens = None
    if isinstance(gen, dict):
        max_tokens = _get(gen, "maxOutputTokens", "max_output_tokens")
        for src_camel, src_snake, dst in (
            ("temperature", None, "temperature"),
            ("topP", "top_p", "top_p"),
            ("stopSequences", "stop_sequences", "stop"),
            ("seed", None, "seed"),
            ("presencePenalty", "presence_penalty", "presence_penalty"),
            ("frequencyPenalty", "frequency_penalty", "frequency_penalty"),
        ):
            value = _get(gen, src_camel, src_snake)
            if value is not None:
                extra[dst] = value
        schema = _get(gen, "responseJsonSchema", "response_json_schema")
        if not isinstance(schema, dict) and isinstance(_get(gen, "responseSchema", "response_schema"), dict):
            schema = _lower_schema_types(_get(gen, "responseSchema", "response_schema"))
        if isinstance(schema, dict):
            extra["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}
        elif _get(gen, "responseMimeType", "response_mime_type") == "application/json":
            extra["response_format"] = {"type": "json_object"}
        effort = _thinking_effort(_get(gen, "thinkingConfig", "thinking_config"))
        if effort:
            extra["reasoning_effort"] = effort

    tools = _tools_to_openai(_get(body, "tools"))
    if tools:
        extra["tools"] = tools
    choice = _tool_choice(_get(body, "toolConfig", "tool_config"))
    if choice is not None and tools:
        extra["tool_choice"] = choice

    return ChatCompletionRequest(
        model=model,
        messages=messages,
        stream=stream,
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        **extra,
    )


def estimate_gemini_tokens(body: dict[str, Any]) -> int:
    serialized = json.dumps(
        {k: body.get(k) for k in ("contents", "systemInstruction", "system_instruction", "tools") if k in body},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, math.ceil(len(serialized) / 4))


_FINISH_REASONS = {"stop": "STOP", "tool_calls": "STOP", "length": "MAX_TOKENS", "content_filter": "SAFETY"}


def _usage_metadata(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "promptTokenCount": prompt,
        "candidatesTokenCount": completion,
        "totalTokenCount": int(usage.get("total_tokens") or (prompt + completion)),
    }


def _function_call_parts(tool_calls: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for call in tool_calls if isinstance(tool_calls, list) else []:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except Exception:
                args = {}
        fc: dict[str, Any] = {"name": fn["name"], "args": args if isinstance(args, dict) else {}}
        if isinstance(call.get("id"), str) and call["id"]:
            fc["id"] = call["id"]
        parts.append({"functionCall": fc})
    return parts


def _response(
    parts: list[dict[str, Any]],
    *,
    model: str | None,
    response_id: str,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": parts}, "index": 0}
    if finish_reason:
        candidate["finishReason"] = finish_reason
    out: dict[str, Any] = {"candidates": [candidate]}
    if usage:
        out["usageMetadata"] = usage
    if model:
        out["modelVersion"] = model
    out["responseId"] = response_id
    return out


def openai_chat_completion_to_gemini_response(chat: dict[str, Any], *, model: str | None) -> dict[str, Any]:
    choices = chat.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    parts: list[dict[str, Any]] = []
    text = normalize_message_content(message.get("content"))
    if text:
        parts.append({"text": text})
    parts.extend(_function_call_parts(message.get("tool_calls")))
    return _response(
        parts,
        model=model or chat.get("model"),
        response_id=str(chat.get("id") or uuid.uuid4().hex),
        finish_reason=_FINISH_REASONS.get(str(choice.get("finish_reason") or "stop"), "OTHER"),
        usage=_usage_metadata(chat.get("usage")),
    )


async def openai_stream_to_gemini_responses(
    chunks: AsyncIterator[bytes | str],
    *,
    model: str | None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield one GenerateContentResponse per text delta, then a final one with finishReason and usage."""
    response_id = uuid.uuid4().hex
    finish_reason = "STOP"
    usage: dict[str, int] | None = None
    async for data in _iter_openai_sse_data(chunks):
        if not data or data.strip() == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("id"), str):
            response_id = obj["id"]
        usage = _usage_metadata(obj.get("usage")) or usage
        choices = obj.get("choices") or []
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        text = delta.get("content")
        if isinstance(text, str) and text:
            yield _response([{"text": text}], model=model, response_id=response_id)
        call_parts = _function_call_parts(delta.get("tool_calls"))
        if call_parts:
            yield _response(call_parts, model=model, response_id=response_id)
        if isinstance(choice.get("finish_reason"), str) and choice["finish_reason"]:
            finish_reason = _FINISH_REASONS.get(choice["finish_reason"], "OTHER")
    yield _response([], model=model, response_id=response_id, finish_reason=finish_reason, usage=usage)


def gemini_model_resource(model_id: str) -> dict[str, Any]:
    return {
        "name": f"models/{model_id}",
        "baseModelId": model_id,
        "version": "gateway",
        "displayName": model_id,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"],
    }
