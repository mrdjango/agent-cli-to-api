import asyncio
import dataclasses
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from codex_gateway import server


def _parse_sse(body: str) -> list[dict]:
    events = []
    for raw in body.split("\n\n"):
        data = [line[len("data:") :].strip() for line in raw.splitlines() if line.startswith("data:")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


async def _collect(agen) -> str:
    return "".join([chunk async for chunk in agen])


async def _chat_chunks(chunks: list[str]):
    for chunk in chunks:
        yield chunk


def _chat_chunk(delta: dict, finish_reason=None) -> str:
    obj = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(obj)}\n\n"


class ChatStreamToResponsesTests(unittest.TestCase):
    def test_text_and_tool_calls_are_converted(self) -> None:
        chunks = [
            _chat_chunk({"role": "assistant"}),
            ": ping\n\n",
            _chat_chunk({"content": "Hel"}),
            _chat_chunk({"content": "lo"}),
            _chat_chunk(
                {
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}
                    ]
                }
            ),
            _chat_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]\n\n",
        ]
        body = asyncio.run(_collect(server._chat_stream_to_responses_events(_chat_chunks(chunks), model="claude-x")))

        self.assertIn(": ping\n\n", body)
        events = _parse_sse(body)
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "response.created")
        self.assertEqual(types[-1], "response.completed")
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
        self.assertEqual(deltas, ["Hel", "lo"])

        completed = events[-1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["model"], "claude-x")
        message, call = completed["output"]
        self.assertEqual(message["content"][0]["text"], "Hello")
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(call["call_id"], "call_1")
        self.assertEqual(call["name"], "shell")
        self.assertEqual(call["arguments"], '{"cmd":"ls"}')

    def test_incremental_tool_call_arguments_are_merged_by_index(self) -> None:
        chunks = [
            _chat_chunk({"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "f", "arguments": '{"a"'}}]}),
            _chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}),
            _chat_chunk({}, finish_reason="tool_calls"),
        ]
        body = asyncio.run(_collect(server._chat_stream_to_responses_events(_chat_chunks(chunks), model=None)))
        completed = _parse_sse(body)[-1]["response"]
        self.assertEqual(len(completed["output"]), 1)
        self.assertEqual(completed["output"][0]["arguments"], '{"a":1}')


class NativeCodexResponsesStreamingTests(unittest.TestCase):
    def _post(self, fake_events):
        with (
            mock.patch.object(server, "settings", dataclasses.replace(server.settings, bearer_token=None, provider="codex")),
            mock.patch.object(server, "load_codex_auth", return_value=SimpleNamespace(api_key="k", access_token=None, account_id=None)),
            mock.patch.object(server, "iter_codex_responses_events", side_effect=lambda **_: fake_events()),
        ):
            client = TestClient(server.app)
            return client.post(
                "/v1/responses",
                json={"model": "gpt-5.4", "input": "hi", "stream": True},
            )

    def test_upstream_events_are_passed_through(self) -> None:
        upstream = [
            {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_1", "output": []}},
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
            {"type": "response.output_text.delta", "sequence_number": 2, "item_id": "msg_1", "delta": "hello"},
            {
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            },
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": {"id": "resp_1", "output": [], "usage": {"input_tokens": 3, "output_tokens": 1}},
            },
        ]

        async def fake_events():
            for evt in upstream:
                yield evt

        resp = self._post(fake_events)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: response.output_text.delta\n", resp.text)
        events = _parse_sse(resp.text)
        self.assertEqual([e["type"] for e in events], [e["type"] for e in upstream])
        completed = events[-1]["response"]
        self.assertEqual(completed["object"], "response")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(completed["usage"]["input_tokens"], 3)

    def test_upstream_failure_before_first_event_returns_http_error(self) -> None:
        async def fake_events():
            raise RuntimeError("codex responses failed: 429: rate limited")
            yield {}  # pragma: no cover

        resp = self._post(fake_events)
        self.assertEqual(resp.status_code, 429)
        self.assertIn("rate limited", resp.json()["error"]["message"])

    def test_stream_ending_early_emits_error_event(self) -> None:
        async def fake_events():
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_1"}}
            yield {"type": "gateway.upstream_incomplete", "message": "connection reset"}

        resp = self._post(fake_events)
        self.assertEqual(resp.status_code, 200)
        events = _parse_sse(resp.text)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["message"], "connection reset")
        self.assertEqual(events[-1]["sequence_number"], 1)


class StreamUsageTests(unittest.TestCase):
    """Gemini CLI stands in for every CLI provider: its final `result` event carries token stats."""

    def _post(self, path: str, body: dict):
        async def fake_gemini_events(**_):
            yield {"type": "message", "role": "assistant", "content": "hi there"}
            yield {"type": "result", "stats": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}}

        cfg = dataclasses.replace(server.settings, bearer_token=None, provider="auto", gemini_use_cloudcode_api=False)
        with (
            mock.patch.object(server, "settings", cfg),
            mock.patch.object(server, "iter_stream_json_events", side_effect=fake_gemini_events),
        ):
            return TestClient(server.app).post(path, json=body)

    def test_chat_stream_emits_usage_chunk_only_when_requested(self) -> None:
        base = {"model": "gemini", "messages": [{"role": "user", "content": "hi"}], "stream": True}

        plain = _parse_sse(self._post("/v1/chat/completions", base).text.replace("data: [DONE]", ""))
        self.assertFalse(any("usage" in e for e in plain))

        body = dict(base, stream_options={"include_usage": True})
        events = _parse_sse(self._post("/v1/chat/completions", body).text.replace("data: [DONE]", ""))
        self.assertEqual(events[-1]["choices"], [])
        self.assertEqual(
            events[-1]["usage"], {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        )

    def test_responses_stream_reports_usage_for_cli_providers(self) -> None:
        resp = self._post("/v1/responses", {"model": "gemini", "input": "hi", "stream": True})
        self.assertEqual(resp.status_code, 200)
        completed = _parse_sse(resp.text)[-1]
        self.assertEqual(completed["type"], "response.completed")
        self.assertEqual(completed["response"]["output"][0]["content"][0]["text"], "hi there")
        self.assertEqual(
            completed["response"]["usage"], {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
        )


    def test_messages_stream_reports_usage_for_cli_providers(self) -> None:
        resp = self._post(
            "/v1/messages",
            {"model": "gemini", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        self.assertEqual(resp.status_code, 200)
        message_delta = next(e for e in _parse_sse(resp.text) if e["type"] == "message_delta")
        self.assertEqual(message_delta["usage"], {"input_tokens": 12, "output_tokens": 3})


if __name__ == "__main__":
    unittest.main()
