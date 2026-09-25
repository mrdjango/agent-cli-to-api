import asyncio
import dataclasses
import json
import os
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from codex_gateway import claude_oauth, server
from codex_gateway.codex_responses import collect_codex_responses_text_and_usage
from codex_gateway.openai_compat import ChatCompletionRequest


async def _events(items):
    for item in items:
        yield item


_FUNCTION_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "status": "completed",
    "arguments": '{"prompt":"a red cat"}',
    "call_id": "call_1",
    "name": "gemini_image_gen",
}


class CodexToolCallTests(unittest.TestCase):
    def test_tool_calls_survive_empty_completed_output(self) -> None:
        # The Codex backend streams the function call as an output item but sends
        # `response.completed` with `output: []` (store=false).
        events = [
            {"type": "response.output_text.delta", "delta": "I'll make that image."},
            {"type": "response.output_item.done", "output_index": 1, "item": _FUNCTION_CALL},
            {"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 3, "output_tokens": 5}}},
        ]

        text, usage, tool_calls, _ = asyncio.run(collect_codex_responses_text_and_usage(_events(events)))

        self.assertEqual(text, "I'll make that image.")
        self.assertEqual(usage["total_tokens"], 8)
        self.assertEqual(
            tool_calls,
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "gemini_image_gen", "arguments": '{"prompt":"a red cat"}'},
                }
            ],
        )

    def test_completed_output_is_used_when_present(self) -> None:
        events = [
            {"type": "response.completed", "response": {"output": [_FUNCTION_CALL]}},
        ]

        _, _, tool_calls, _ = asyncio.run(collect_codex_responses_text_and_usage(_events(events)))

        self.assertEqual([call["id"] for call in tool_calls], ["call_1"])

    def test_no_tool_calls_for_plain_text(self) -> None:
        events = [
            {"type": "response.output_text.delta", "delta": "hi"},
            {"type": "response.completed", "response": {"output": []}},
        ]

        _, _, tool_calls, _ = asyncio.run(collect_codex_responses_text_and_usage(_events(events)))

        self.assertIsNone(tool_calls)


_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "gemini_image_gen",
        "description": "Generate an image from a text prompt",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
    },
}

_CLAUDE_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Making it now."},
        {"type": "tool_use", "id": "toolu_1", "name": "gemini_image_gen", "input": {"prompt": "a red cat"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 7},
}


def _claude_sse() -> str:
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "content": [], "usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Making it now."}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "gemini_image_gen", "input": {}},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"prompt": '}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"a red cat"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 10, "output_tokens": 7}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


class _MockAnthropic:
    """Serves the Anthropic Messages API and records the request payloads."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.headers: list[httpx.Headers] = []
        self.urls: list[httpx.URL] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.payloads.append(payload)
        self.headers.append(request.headers)
        self.urls.append(request.url)
        if payload.get("stream"):
            return httpx.Response(200, text=_claude_sse(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_CLAUDE_MESSAGE)

    def patches(self, *, subscription_token: str | None = None):
        """Serve requests from this mock; with `subscription_token`, authenticate like `claude setup-token`."""

        async def client(_name: str = "default") -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

        if subscription_token:
            cli_config = claude_oauth.ClaudeCliConfig(None, None, None)
            env = mock.patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": subscription_token})
        else:
            cli_config = claude_oauth.ClaudeCliConfig("https://anthropic.test", "test-token", None)
            env_without_token = {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_OAUTH_TOKEN"}
            env = mock.patch.dict(os.environ, env_without_token, clear=True)
        return (
            mock.patch.object(claude_oauth, "get_async_client", client),
            mock.patch.object(claude_oauth, "get_claude_cli_config", lambda: cli_config),
            env,
        )


_EXPECTED_CALL = {
    "id": "toolu_1",
    "type": "function",
    "function": {"name": "gemini_image_gen", "arguments": '{"prompt": "a red cat"}'},
}


class ClaudeOAuthToolTests(unittest.TestCase):
    def _run(self, coro, api: _MockAnthropic, **patch_kwargs):
        p1, p2, p3 = api.patches(**patch_kwargs)
        with p1, p2, p3:
            return asyncio.run(coro)

    def test_request_carries_tools_and_tool_choice(self) -> None:
        payload: dict = {}
        req = ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_IMAGE_TOOL],
            tool_choice={"type": "function", "function": {"name": "gemini_image_gen"}},
            parallel_tool_calls=False,
        )
        claude_oauth._apply_openai_tools(payload, req)

        self.assertEqual(payload["tools"][0]["name"], "gemini_image_gen")
        self.assertEqual(payload["tools"][0]["input_schema"]["required"], ["prompt"])
        self.assertEqual(
            payload["tool_choice"], {"type": "tool", "name": "gemini_image_gen", "disable_parallel_tool_use": True}
        )

    def test_tool_choice_none_keeps_tool_definitions(self) -> None:
        payload: dict = {}
        req = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}], tools=[_IMAGE_TOOL], tool_choice="none")
        claude_oauth._apply_openai_tools(payload, req)

        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(payload["tool_choice"], {"type": "none"})

    def test_parallel_tool_results_share_one_user_turn(self) -> None:
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "two images"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "toolu_a", "type": "function", "function": {"name": "gemini_image_gen", "arguments": "{}"}},
                        {"id": "toolu_b", "type": "function", "function": {"name": "gemini_image_gen", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_a", "content": "done a"},
                {"role": "tool", "tool_call_id": "toolu_b", "content": "done b"},
            ],
        )
        _, messages = claude_oauth._openai_messages_to_anthropic(req)

        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual([b["tool_use_id"] for b in messages[2]["content"]], ["toolu_a", "toolu_b"])

    def test_generate_returns_tool_calls(self) -> None:
        api = _MockAnthropic()
        req = ChatCompletionRequest(model="claude-sonnet-5", messages=[{"role": "user", "content": "cat"}], tools=[_IMAGE_TOOL])
        text, usage, tool_calls = self._run(claude_oauth.generate_oauth(req=req, model_name="claude-sonnet-5"), api)

        self.assertEqual(api.payloads[0]["tools"][0]["name"], "gemini_image_gen")
        self.assertEqual(text, "Making it now.")
        self.assertEqual(usage["total_tokens"], 17)
        self.assertEqual(tool_calls, [_EXPECTED_CALL])

    def test_stream_assembles_tool_calls(self) -> None:
        api = _MockAnthropic()
        req = ChatCompletionRequest(model="claude-sonnet-5", messages=[{"role": "user", "content": "cat"}], tools=[_IMAGE_TOOL])

        async def collect():
            return [e async for e in claude_oauth.iter_oauth_stream_events(req=req, model_name="claude-sonnet-5")]

        events = self._run(collect(), api)
        text = "".join(
            b["text"] for e in events if e["type"] == "assistant" for b in e["message"]["content"]
        )
        calls = [e for e in events if e["type"] == "gateway.tool_calls"]

        self.assertEqual(text, "Making it now.")
        self.assertEqual(calls, [{"type": "gateway.tool_calls", "tool_calls": [_EXPECTED_CALL]}])


    def test_subscription_token_sends_beta_header_and_claude_code_identity(self) -> None:
        api = _MockAnthropic()
        req = ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[{"role": "system", "content": "You are Tchat."}, {"role": "user", "content": "cat"}],
            tools=[_IMAGE_TOOL],
        )
        self._run(claude_oauth.generate_oauth(req=req, model_name="opus[1m]"), api, subscription_token="sk-ant-oat01-test")

        headers, payload = api.headers[0], api.payloads[0]
        self.assertEqual(headers["authorization"], "Bearer sk-ant-oat01-test")
        self.assertEqual(headers["anthropic-beta"], "oauth-2025-04-20")
        self.assertEqual(str(api.urls[0]), "https://api.anthropic.com/v1/messages")
        self.assertEqual(
            [b["text"] for b in payload["system"]],
            ["You are Claude Code, Anthropic's official CLI for Claude.", claude_oauth._API_CONTEXT, "You are Tchat."],
        )
        self.assertEqual(payload["model"], "claude-opus-5-5")

    def test_api_key_proxy_sends_client_system_only(self) -> None:
        api = _MockAnthropic()
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "system", "content": "You are Tchat."}, {"role": "user", "content": "cat"}]
        )
        self._run(claude_oauth.generate_oauth(req=req, model_name="claude-sonnet-5"), api)

        self.assertNotIn("anthropic-beta", api.headers[0])
        self.assertEqual([b["text"] for b in api.payloads[0]["system"]], ["You are Tchat."])

    def test_model_aliases_resolve_like_the_cli(self) -> None:
        oauth = claude_oauth._ClaudeAuth("t", "https://api.anthropic.com", True)
        cases = {
            "opus": "claude-opus-5-5",
            "sonnet": "claude-sonnet-5",
            "haiku": "claude-haiku-4-5-20251001",
            "opus[1m]": "claude-opus-5-5",
            "claude-opus-5-5[1m]": "claude-opus-5-5",
            "claude-sonnet-4-6": "claude-sonnet-4-6",
        }
        for requested, expected in cases.items():
            with self.subTest(requested=requested):
                self.assertEqual(claude_oauth._api_model_name(requested, oauth), expected)

    def test_default_sonnet_model_only_replaces_sonnet(self) -> None:
        proxy = claude_oauth._ClaudeAuth("t", "https://proxy.test", False, "vendor-sonnet")

        self.assertEqual(claude_oauth._api_model_name("sonnet", proxy), "vendor-sonnet")
        self.assertEqual(claude_oauth._api_model_name("opus", proxy), "claude-opus-5-5")
        self.assertEqual(claude_oauth._api_model_name("haiku", proxy), "claude-haiku-4-5-20251001")

    def test_effort_is_sent_only_where_the_model_accepts_it(self) -> None:
        cases = [
            ("claude-opus-5-5", "xhigh", "xhigh"),
            ("claude-sonnet-5", "max", "max"),
            ("claude-opus-4-7", "xhigh", "xhigh"),
            ("claude-sonnet-4-6", "high", "high"),
            ("claude-sonnet-4-6", "xhigh", None),
            ("claude-opus-4-5", "medium", "medium"),
            ("claude-opus-4-5", "max", None),
            ("claude-sonnet-4-5", "high", None),
            ("claude-haiku-4-5-20251001", "low", None),
            ("claude-opus-5-5", None, None),
        ]
        for model, effort, expected in cases:
            with self.subTest(model=model, effort=effort):
                self.assertEqual(claude_oauth._supported_effort(model, effort), expected)

    def test_effort_goes_in_output_config(self) -> None:
        api = _MockAnthropic()
        req = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "cat"}])
        self._run(claude_oauth.generate_oauth(req=req, model_name="claude-opus-5-5", effort="high"), api)

        self.assertEqual(api.payloads[0]["output_config"], {"effort": "high"})


class ClaudeOAuthChatCompletionsTests(unittest.TestCase):
    def _post(self, body: dict) -> httpx.Response:
        api = _MockAnthropic()
        oauth_settings = dataclasses.replace(server.settings, claude_use_oauth_api=True, bearer_token=None)
        p1, p2, p3 = api.patches()
        with p1, p2, p3, mock.patch.object(server, "settings", oauth_settings):
            resp = TestClient(server.app).post("/v1/chat/completions", json=body)
        self.assertEqual(api.payloads[0]["tools"][0]["name"], "gemini_image_gen")
        return resp

    def test_non_stream_returns_tool_calls(self) -> None:
        resp = self._post({"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "cat"}], "tools": [_IMAGE_TOOL]})

        choice = resp.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"], [_EXPECTED_CALL])

    def test_stream_returns_indexed_tool_calls(self) -> None:
        resp = self._post(
            {"model": "claude-sonnet-5", "stream": True, "messages": [{"role": "user", "content": "cat"}], "tools": [_IMAGE_TOOL]}
        )
        chunks = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        deltas = [c["choices"][0] for c in chunks if c.get("choices")]
        tool_deltas = [d["delta"]["tool_calls"] for d in deltas if d["delta"].get("tool_calls")]

        self.assertEqual(tool_deltas, [[{"index": 0, **_EXPECTED_CALL}]])
        self.assertEqual(deltas[-1]["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
