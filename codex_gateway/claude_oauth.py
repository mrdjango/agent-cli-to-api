from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .http_client import get_async_client, request_json_with_retries
from .openai_compat import ChatCompletionRequest, ChatMessage, RequestInputError, normalize_message_content

_ANTHROPIC_VERSION = "2023-06-01"
# Claude subscription OAuth tokens (`claude setup-token`, OAuth login) are only accepted by the
# Messages API with this beta header and the Claude Code identity as the first system block;
# otherwise the API answers with a 429 rate_limit_error.
_OAUTH_BETA = "oauth-2025-04-20"
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
# Follows the required identity so the model does not act as the Claude Code CLI.
_API_CONTEXT = (
    "Here you are Claude, a general-purpose AI assistant made by Anthropic, reached through an API. "
    "You are not running as the Claude Code CLI: there is no terminal, file system or code execution. "
    "The only tools you can use are the ones defined in this request, if any. Do not describe or "
    "speculate about the infrastructure this conversation runs on."
)
# The Messages API does not know Claude Code's model aliases; resolve them the way `claude --model` does.
_CLAUDE_MODEL_ALIASES = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5-20251001",
}
_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
_DEFAULT_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class ClaudeOAuthCreds:
    access_token: str | None
    refresh_token: str | None
    expires_at_s: int | None
    token_type: str | None


@dataclass(frozen=True)
class ClaudeCliConfig:
    """Configuration extracted from Claude CLI's settings.json."""
    base_url: str | None
    auth_token: str | None
    default_model: str | None


def _load_claude_cli_settings() -> ClaudeCliConfig:
    """Load Claude CLI settings from ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return ClaudeCliConfig(None, None, None)
    if not isinstance(raw, dict):
        return ClaudeCliConfig(None, None, None)
    
    # Extract env from settings.json
    env = raw.get("env") or {}
    if not isinstance(env, dict):
        env = {}
    
    base_url = env.get("ANTHROPIC_BASE_URL")
    auth_token = env.get("ANTHROPIC_AUTH_TOKEN")
    default_model = env.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
    
    return ClaudeCliConfig(
        base_url if isinstance(base_url, str) else None,
        auth_token if isinstance(auth_token, str) else None,
        default_model if isinstance(default_model, str) else None,
    )


# Cache the CLI config at module load time
_cli_config: ClaudeCliConfig | None = None


def get_claude_cli_config() -> ClaudeCliConfig:
    """Get cached Claude CLI configuration."""
    global _cli_config
    if _cli_config is None:
        _cli_config = _load_claude_cli_settings()
    return _cli_config


def _load_creds(path: Path) -> ClaudeOAuthCreds:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ClaudeOAuthCreds(None, None, None, None)
    if not isinstance(raw, dict):
        return ClaudeOAuthCreds(None, None, None, None)
    access_token = raw.get("access_token")
    refresh_token = raw.get("refresh_token")
    expires_at_s = raw.get("expires_at_s")
    token_type = raw.get("token_type")
    return ClaudeOAuthCreds(
        access_token if isinstance(access_token, str) else None,
        refresh_token if isinstance(refresh_token, str) else None,
        int(expires_at_s) if isinstance(expires_at_s, (int, float)) else None,
        token_type if isinstance(token_type, str) else None,
    )


def _save_creds(path: Path, creds: ClaudeOAuthCreds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if creds.access_token:
        payload["access_token"] = creds.access_token
    if creds.refresh_token:
        payload["refresh_token"] = creds.refresh_token
    if creds.expires_at_s is not None:
        payload["expires_at_s"] = int(creds.expires_at_s)
    if creds.token_type:
        payload["token_type"] = creds.token_type
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass


def _is_expired(expires_at_s: int | None, *, skew_s: int = 90) -> bool:
    if not expires_at_s:
        return True
    return expires_at_s <= int(time.time()) + skew_s


async def _refresh_access_token(
    *,
    refresh_token: str,
    oauth_client_id: str,
    base_url: str,
    timeout_s: int,
) -> ClaudeOAuthCreds:
    url = f"{base_url.rstrip('/')}/v1/oauth/token"
    payload = {
        "client_id": oauth_client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    client = await get_async_client("claude-oauth")
    resp = await request_json_with_retries(
        client=client,
        method="POST",
        url=url,
        timeout_s=timeout_s,
        json=payload,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Claude OAuth refresh: invalid JSON response")
    access_token = data.get("access_token")
    new_refresh = data.get("refresh_token") or refresh_token
    expires_in = data.get("expires_in")
    token_type = data.get("token_type") or "Bearer"
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("Claude OAuth refresh: missing access_token")
    expires_at_s = None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at_s = int(time.time() + int(expires_in))
    return ClaudeOAuthCreds(access_token, str(new_refresh), expires_at_s, str(token_type))


async def maybe_refresh_claude_oauth(creds_path: str) -> ClaudeOAuthCreds:
    path = Path(creds_path).expanduser()
    creds = _load_creds(path)
    if creds.access_token and not _is_expired(creds.expires_at_s):
        return creds
    if not creds.refresh_token:
        return creds

    oauth_client_id = settings.claude_oauth_client_id or _DEFAULT_OAUTH_CLIENT_ID
    base_url = settings.claude_oauth_base_url
    refreshed = await _refresh_access_token(
        refresh_token=creds.refresh_token,
        oauth_client_id=oauth_client_id,
        base_url=base_url,
        timeout_s=settings.timeout_seconds,
    )
    _save_creds(path, refreshed)
    return refreshed


def _parse_data_url(data_url: str) -> tuple[str, str] | None:
    # data:<mime>;base64,<payload>
    if not data_url.startswith("data:"):
        return None
    header, _, b64 = data_url.partition(",")
    if not b64:
        return None
    if ";base64" not in header:
        return None
    mime = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
    return mime, b64


def _guess_mime_type(filename: str | None) -> str:
    mime, _ = mimetypes.guess_type((filename or "").strip())
    return mime or "application/octet-stream"


def _parse_openai_file_source(source: dict[str, Any]) -> tuple[str, str, str | None] | None:
    file_id = source.get("file_id")
    if isinstance(file_id, str) and file_id.strip():
        raise RequestInputError("Claude OAuth does not support OpenAI file_id passthrough; send inline file_data instead")

    file_url = source.get("file_url")
    if isinstance(file_url, str) and file_url.strip():
        raise RequestInputError("Claude OAuth does not support file_url inputs; send inline file_data instead")

    file_data = source.get("file_data")
    if not isinstance(file_data, str) or not file_data.strip():
        return None

    filename = source.get("filename")
    clean_name = filename.strip() if isinstance(filename, str) and filename.strip() else None
    raw = file_data.strip()
    if raw.startswith("data:"):
        parsed = _parse_data_url(raw)
        if not parsed:
            raise RequestInputError("Invalid file_data data URL")
        mime, b64 = parsed
        return mime, "".join(b64.split()), clean_name

    b64 = "".join(raw.split())
    try:
        base64.b64decode(b64, validate=False)
    except Exception as e:
        raise RequestInputError("Invalid file_data base64 payload") from e
    return _guess_mime_type(clean_name), b64, clean_name


def _content_to_anthropic_blocks(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        text = content.strip()
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(content, list):
        return []

    blocks: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
        elif t == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = None
            if not isinstance(url, str):
                continue
            parsed = _parse_data_url(url)
            if not parsed:
                continue
            mime, b64 = parsed
            if len(b64) > settings.max_image_bytes * 2:
                continue
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                }
            )
        elif t in {"file", "input_file"}:
            source = item.get("file") if t == "file" else item
            if not isinstance(source, dict):
                continue
            parsed = _parse_openai_file_source(source)
            if not parsed:
                continue
            mime, b64, filename = parsed
            if mime != "application/pdf":
                raise RequestInputError(f"Claude OAuth only supports inline PDF file_data today (got {mime})")
            block: dict[str, Any] = {
                "type": "document",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            }
            if filename:
                block["title"] = filename
            blocks.append(block)
    return blocks


def _openai_messages_to_anthropic(req: ChatCompletionRequest) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    tool_call_id_map: dict[str, str] = {}
    for msg in req.messages:
        if not isinstance(msg, ChatMessage):
            continue
        role = (msg.role or "").strip()
        blocks = _content_to_anthropic_blocks(getattr(msg, "content", None))
        extra = getattr(msg, "model_extra", None) or {}
        tool_calls = extra.get("tool_calls") if isinstance(extra, dict) else None

        if role == "system":
            for b in blocks:
                if b.get("type") == "text":
                    system_parts.append(str(b.get("text") or ""))
            continue

        if role == "tool":
            tool_call_id = None
            if isinstance(extra, dict):
                tool_call_id = extra.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                tool_call_id = None
            if tool_call_id is None:
                tool_call_id = getattr(msg, "tool_call_id", None)
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            content = normalize_message_content(getattr(msg, "content", None))
            result_block = {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}
            # Results of parallel tool calls belong in a single user turn.
            prev = out[-1] if out else None
            if (
                prev
                and prev["role"] == "user"
                and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in prev["content"])
            ):
                prev["content"].append(result_block)
            else:
                out.append({"role": "user", "content": [result_block]})
            continue

        if role not in {"user", "assistant"}:
            continue
        if role == "assistant" and isinstance(tool_calls, list):
            tool_blocks: list[dict[str, Any]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id") or call.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"toolu_{len(tool_blocks) + 1}"
                func = call.get("function")
                name = None
                args = None
                if isinstance(func, dict):
                    name = func.get("name")
                    args = func.get("arguments")
                if not isinstance(name, str) or not name:
                    name = call.get("name") if isinstance(call.get("name"), str) else "tool"
                parsed_args: dict[str, Any] = {}
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        if isinstance(parsed, dict):
                            parsed_args = parsed
                    except Exception:
                        parsed_args = {}
                elif isinstance(args, dict):
                    parsed_args = args
                tool_blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": parsed_args,
                    }
                )
                tool_call_id_map[call_id] = name
            if tool_blocks:
                blocks.extend(tool_blocks)

        if not blocks:
            continue
        out.append({"role": role, "content": blocks})

    system = "\n\n".join([p for p in (s.strip() for s in system_parts) if p]) or None
    return system, out


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not isinstance(name, str) or not name:
            continue
        entry: dict[str, Any] = {"name": name}
        desc = func.get("description")
        if isinstance(desc, str) and desc:
            entry["description"] = desc
        params = func.get("parameters")
        if isinstance(params, dict):
            entry["input_schema"] = params
        out.append(entry)
    return out


def _openai_tool_choice_to_anthropic(choice: Any) -> dict[str, Any] | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        lowered = choice.strip().lower()
        if lowered in {"auto", ""}:
            return None
        if lowered in {"required", "any"}:
            return {"type": "any"}
        if lowered == "none":
            return {"type": "none"}
        return None
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype == "function":
            fn = choice.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"]:
                return {"type": "tool", "name": fn["name"]}
    return None


def _apply_openai_tools(payload: dict[str, Any], req: ChatCompletionRequest) -> None:
    extra = getattr(req, "model_extra", None) or {}
    if not isinstance(extra, dict):
        return
    tools = extra.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    converted = _openai_tools_to_anthropic(tools)
    if not converted:
        return
    payload["tools"] = converted
    mapped_choice = _openai_tool_choice_to_anthropic(extra.get("tool_choice"))
    if extra.get("parallel_tool_calls") is False and (mapped_choice or {}).get("type") != "none":
        mapped_choice = {**(mapped_choice or {"type": "auto"}), "disable_parallel_tool_use": True}
    if mapped_choice is not None:
        payload["tool_choice"] = mapped_choice


def _anthropic_tool_use_to_openai(block_id: Any, name: Any, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    return {
        "id": block_id if isinstance(block_id, str) and block_id else f"toolu_{uuid.uuid4().hex}",
        "type": "function",
        "function": {"name": name if isinstance(name, str) and name else "tool", "arguments": arguments or "{}"},
    }


def _extract_tool_calls_from_anthropic_response(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("content"), list):
        return []
    return [
        _anthropic_tool_use_to_openai(item.get("id"), item.get("name"), item.get("input"))
        for item in data["content"]
        if isinstance(item, dict) and item.get("type") == "tool_use"
    ]


def _extract_text_from_anthropic_response(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    content = data.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def _extract_usage_from_anthropic_response(data: Any) -> dict[str, int] | None:
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    in_tokens = int(usage.get("input_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or 0)
    return {"prompt_tokens": in_tokens, "completion_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}


@dataclass(frozen=True)
class _ClaudeAuth:
    token: str
    base_url: str
    # True for Claude subscription OAuth tokens, which need the beta header and Claude Code identity.
    subscription: bool
    default_sonnet_model: str | None = None


async def _resolve_auth() -> _ClaudeAuth:
    """CLI settings.json (API-key style proxy) first, then CLAUDE_CODE_OAUTH_TOKEN, then the OAuth creds file."""
    cli_config = get_claude_cli_config()
    if cli_config.auth_token and cli_config.base_url:
        return _ClaudeAuth(cli_config.auth_token, cli_config.base_url, False, cli_config.default_model)
    env_token = (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or "").strip()
    if env_token:
        return _ClaudeAuth(env_token, settings.claude_api_base_url, True)
    creds = await maybe_refresh_claude_oauth(settings.claude_oauth_creds_path)
    if not creds.access_token:
        raise RuntimeError(
            "Claude API: no authentication available. Set CLAUDE_CODE_OAUTH_TOKEN (claude setup-token), "
            "configure ANTHROPIC_AUTH_TOKEN in ~/.claude/settings.json, or run claude-oauth-login."
        )
    return _ClaudeAuth(creds.access_token, settings.claude_api_base_url, True)


def _api_model_name(model_name: str, auth: _ClaudeAuth) -> str:
    """Resolve CLI aliases to Messages API model IDs; `[1m]` is dropped because 1M context is the default."""
    name = model_name.strip()
    if name.lower().endswith("[1m]"):
        name = name[: -len("[1m]")]
    alias = name.lower()
    if alias == "sonnet" and auth.default_sonnet_model:
        return auth.default_sonnet_model
    return _CLAUDE_MODEL_ALIASES.get(alias, name)


def _supported_effort(model_name: str, effort: str | None) -> str | None:
    """`output_config.effort` for models that accept the level, else None (the model default)."""
    if effort not in _EFFORT_LEVELS:
        return None
    m = re.match(r"claude-(opus|sonnet|fable|mythos)-(\d+)(?:-(\d+))?(?:-\d{8})?$", model_name)
    if not m:
        return None
    version = (int(m.group(2)), int(m.group(3) or 0))
    if version >= (4, 7):
        return effort
    if version == (4, 6) and effort != "xhigh":
        return effort
    if m.group(1) == "opus" and version == (4, 5) and effort in {"low", "medium", "high"}:
        return effort
    return None


def _build_request(
    req: ChatCompletionRequest,
    *,
    model_name: str,
    auth: _ClaudeAuth,
    stream: bool,
    effort: str | None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    system, messages = _openai_messages_to_anthropic(req)
    payload: dict[str, Any] = {
        "model": model_name,
        "max_tokens": int(req.max_tokens or 16000),
        "messages": messages,
    }
    if stream:
        payload["stream"] = True
    system_blocks: list[dict[str, Any]] = []
    if auth.subscription:
        system_blocks.append({"type": "text", "text": _CLAUDE_CODE_IDENTITY})
        system_blocks.append({"type": "text", "text": _API_CONTEXT})
    if system:
        system_blocks.append({"type": "text", "text": system})
    if system_blocks:
        payload["system"] = system_blocks
    api_effort = _supported_effort(model_name, effort)
    if api_effort:
        payload["output_config"] = {"effort": api_effort}
    _apply_openai_tools(payload, req)

    headers = {
        "Authorization": f"Bearer {auth.token}",
        "anthropic-version": _ANTHROPIC_VERSION,
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if auth.subscription:
        headers["anthropic-beta"] = _OAUTH_BETA
    return f"{auth.base_url.rstrip('/')}/v1/messages", headers, payload


async def generate_oauth(
    *,
    req: ChatCompletionRequest,
    model_name: str,
    effort: str | None = None,
) -> tuple[str, dict[str, int] | None, list[dict[str, Any]]]:
    auth = await _resolve_auth()
    model_name = _api_model_name(model_name, auth)
    url, headers, payload = _build_request(req, model_name=model_name, auth=auth, stream=False, effort=effort)

    t_prepare = time.time()
    logger.debug(
        "claude-oauth request: url=%s model=%s max_tokens=%d msg_count=%d",
        url, model_name, payload["max_tokens"], len(payload["messages"]),
    )

    client = await get_async_client("claude")
    resp = await request_json_with_retries(
        client=client,
        method="POST",
        url=url,
        timeout_s=settings.timeout_seconds,
        json=payload,
        headers=headers,
    )
    
    t_response = time.time()
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        await _log_upstream_error(resp=resp, url=url, model_name=model_name, stream=False)
        raise
    data = resp.json()
    
    t_parse = time.time()
    api_latency_ms = int((t_response - t_prepare) * 1000)
    logger.debug(
        "claude-oauth response: status=%d api_latency_ms=%d",
        resp.status_code, api_latency_ms,
    )

    return (
        _extract_text_from_anthropic_response(data),
        _extract_usage_from_anthropic_response(data),
        _extract_tool_calls_from_anthropic_response(data),
    )


async def _iter_sse_events(resp: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    event: str | None = None
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if line is None:
            continue
        if line.startswith(":"):
            continue
        if not line.strip():
            if data_lines:
                yield event, "\n".join(data_lines)
            event = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip() or None
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue
    if data_lines:
        yield event, "\n".join(data_lines)


def _extract_delta_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    delta = obj.get("delta")
    if isinstance(delta, dict):
        t = delta.get("text")
        if isinstance(t, str) and t:
            return t
    t2 = obj.get("text")
    if isinstance(t2, str) and t2:
        return t2
    content_block = obj.get("content_block")
    if isinstance(content_block, dict):
        t3 = content_block.get("text")
        if isinstance(t3, str) and t3:
            return t3
    message = obj.get("message")
    if isinstance(message, dict):
        return _extract_text_from_anthropic_response(message)
    return ""


def _extract_stream_usage(obj: Any) -> dict[str, int] | None:
    if not isinstance(obj, dict):
        return None
    if "usage" in obj:
        return _extract_usage_from_anthropic_response(obj)
    msg = obj.get("message")
    if isinstance(msg, dict):
        return _extract_usage_from_anthropic_response(msg)
    return None


def _pick_header(headers: httpx.Headers, *names: str) -> str | None:
    for name in names:
        value = headers.get(name)
        if value:
            return value
    return None


def _truncate_log_text(text: str, *, max_len: int = 600) -> str:
    cleaned = text.replace("\r", "").replace("\n", "\\n")
    if len(cleaned) > max_len:
        return f"{cleaned[:max_len]}... (len={len(cleaned)})"
    return cleaned


async def _summarize_error_body(resp: httpx.Response) -> str | None:
    try:
        if not resp.is_closed:
            await resp.aread()
    except Exception:
        pass
    body: str | None = None
    try:
        payload = resp.json()
    except Exception:
        try:
            text = resp.text
        except Exception:
            return None
        body = text if text else None
    else:
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("type")
                if isinstance(msg, str) and msg:
                    body = msg
                else:
                    body = json.dumps(err, ensure_ascii=True)
            elif isinstance(err, str):
                body = err
            elif isinstance(payload.get("message"), str):
                body = payload["message"]
        if body is None:
            body = json.dumps(payload, ensure_ascii=True)
    if not body:
        return None
    return _truncate_log_text(body)


def _summarize_rate_limit_headers(headers: httpx.Headers) -> str | None:
    retry_after = headers.get("retry-after")
    request_id = _pick_header(headers, "x-request-id", "request-id", "anthropic-request-id")
    limit = headers.get("x-ratelimit-limit")
    remaining = headers.get("x-ratelimit-remaining")
    reset = headers.get("x-ratelimit-reset")
    parts: list[str] = []
    if retry_after:
        parts.append(f"retry_after={retry_after}")
    if request_id:
        parts.append(f"request_id={request_id}")
    rate_parts: list[str] = []
    if limit:
        rate_parts.append(f"limit={limit}")
    if remaining:
        rate_parts.append(f"remaining={remaining}")
    if reset:
        rate_parts.append(f"reset={reset}")
    if rate_parts:
        parts.append(f"rate_limit({', '.join(rate_parts)})")
    if not parts:
        return None
    return " ".join(parts)


async def _log_upstream_error(
    *,
    resp: httpx.Response,
    url: str,
    model_name: str,
    stream: bool,
) -> None:
    status = resp.status_code
    mode = "stream" if stream else "request"
    logger.error("claude-oauth upstream error: mode=%s status=%d url=%s model=%s", mode, status, url, model_name)
    header_summary = _summarize_rate_limit_headers(resp.headers)
    if header_summary:
        logger.error("claude-oauth upstream headers: %s", header_summary)
    body_summary = await _summarize_error_body(resp)
    if body_summary:
        logger.error("claude-oauth upstream body: %s", body_summary)


async def iter_oauth_stream_events(
    *,
    req: ChatCompletionRequest,
    model_name: str,
    effort: str | None = None,
) -> AsyncIterator[dict]:
    auth = await _resolve_auth()
    model_name = _api_model_name(model_name, auth)
    url, headers, payload = _build_request(req, model_name=model_name, auth=auth, stream=True, effort=effort)

    usage: dict[str, int] | None = None
    # tool_use blocks by content index: [id, name, partial_json chunks, input from content_block_start]
    tool_blocks: dict[int, list[Any]] = {}
    client = await get_async_client("claude-stream")
    async with client.stream("POST", url, json=payload, headers=headers, timeout=settings.timeout_seconds) as resp:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            await _log_upstream_error(resp=resp, url=url, model_name=model_name, stream=True)
            raise
        async for _, data in _iter_sse_events(resp):
                if not data or data.strip() == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("index"), int):
                    block = obj.get("content_block")
                    if obj.get("type") == "content_block_start" and isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_blocks[obj["index"]] = [block.get("id"), block.get("name"), [], block.get("input")]
                        continue
                    json_delta = obj.get("delta")
                    if (
                        obj.get("type") == "content_block_delta"
                        and isinstance(json_delta, dict)
                        and json_delta.get("type") == "input_json_delta"
                    ):
                        if obj["index"] in tool_blocks and isinstance(json_delta.get("partial_json"), str):
                            tool_blocks[obj["index"]][2].append(json_delta["partial_json"])
                        continue
                delta = _extract_delta_text(obj)
                if delta:
                    yield {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": delta}]},
                    }
                maybe_usage = _extract_stream_usage(obj)
                if maybe_usage:
                    usage = maybe_usage

    if tool_blocks:
        yield {
            "type": "gateway.tool_calls",
            "tool_calls": [
                _anthropic_tool_use_to_openai(block_id, name, "".join(parts) if parts else start_input)
                for block_id, name, parts, start_input in (tool_blocks[i] for i in sorted(tool_blocks))
            ],
        }
    if usage:
        yield {"type": "result", "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]}}
