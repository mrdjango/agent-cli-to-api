import asyncio
import dataclasses
import unittest
from unittest import mock

from starlette.requests import Request

from codex_gateway import server
from codex_gateway.openai_compat import ChatCompletionRequest


def _request(path: str = "/v1/chat/completions") -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request({"type": "http", "method": "POST", "path": path, "headers": []}, receive)


def _chat_req(**extra) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-opus-5", messages=[{"role": "user", "content": "hi"}], stream=True, **extra
    )


class ClaudeEffortTests(unittest.TestCase):
    def test_reads_anthropic_output_config_effort(self) -> None:
        self.assertEqual(server._extract_reasoning_effort(_chat_req(output_config={"effort": "max"})), "max")
        self.assertEqual(server._extract_reasoning_effort(_chat_req(reasoning={"effort": "medium"})), "medium")
        self.assertIsNone(server._extract_reasoning_effort(_chat_req(thinking={"type": "adaptive"})))

    def test_normalize_claude_effort(self) -> None:
        for raw, expected in {"max": "max", "XHigh": "xhigh", "none": "low", "bogus": None, None: None}.items():
            with self.subTest(raw=raw):
                self.assertEqual(server._normalize_claude_effort(raw), expected)

    def test_claude_cli_cmd_passes_effort_before_prompt(self) -> None:
        cmd = server._claude_cli_cmd("claude-opus-5", "hello", "high")
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")
        self.assertLess(cmd.index("--effort"), cmd.index("--"))
        self.assertNotIn("--effort", server._claude_cli_cmd("claude-opus-5", "hello"))


class RequestMetadataTests(unittest.TestCase):
    def test_empty_extras_are_hidden(self) -> None:
        md, plain = server._format_request_metadata(
            _chat_req(tools=[], metadata={"user_id": "u"}),
            resolved_model="claude-opus-5",
            provider="claude",
            mode_label="cli",
            reasoning_effort="cli-default",
            effort_source="default",
            request_effort_raw=None,
        )
        self.assertNotIn("tools", md)
        self.assertIn("metadata", md)
        self.assertIn("reasoning_effort=effective=cli-default", plain)

    def test_non_tty_log_keeps_bracketed_text(self) -> None:
        import io

        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None, width=200)
        console._is_real_tty = False  # type: ignore[attr-defined]
        md, _plain = server._format_request_metadata(
            _chat_req(tools=[{"type": "function", "function": {"name": "Bash"}}]),
            resolved_model="claude-opus-5",
            provider="claude",
            mode_label="cli",
            reasoning_effort="cli-default",
            effort_source="default",
            request_effort_raw=None,
        )
        cfg = dataclasses.replace(server.settings, log_render_markdown=True)
        with mock.patch.object(server, "settings", cfg), mock.patch.object(server, "_get_rich_console", return_value=console):
            server._maybe_print_markdown("chatcmpl-e5fcfd8b0000", "REQUEST PARAMS", md)
        out = buf.getvalue()
        self.assertIn("[function:Bash]", out)
        self.assertIn("Request Params [e5fcfd8b]", out)


class StreamOutcomeTests(unittest.TestCase):
    def _run_stream(self, events: list[dict]) -> tuple[str, server.RequestStats, list[list[str]]]:
        captured: list[list[str]] = []

        async def fake_events(*, cmd, **_kwargs):
            captured.append(cmd)
            for evt in events:
                if "_raise" in evt:
                    raise RuntimeError(evt["_raise"])
                yield evt

        cfg = dataclasses.replace(server.settings, strict_models=True, claude_use_oauth_api=False)

        async def run() -> str:
            resp = await server.chat_completions(
                _chat_req(output_config={"effort": "high"}), _request(), None
            )
            return "".join([c if isinstance(c, str) else c.decode() async for c in resp.body_iterator])

        stats = server.RequestStats()
        with (
            mock.patch.object(server, "settings", cfg),
            mock.patch.object(server, "iter_stream_json_events", fake_events),
            mock.patch.object(server, "_request_stats", stats),
            mock.patch.object(server, "_maybe_print_stats"),
        ):
            body = asyncio.run(run())
        return body, stats, captured

    def test_clean_stream_counts_as_success(self) -> None:
        body, stats, captured = self._run_stream(
            [
                {"type": "system", "subtype": "init", "model": "claude-opus-5"},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
                {"type": "result", "result": "ok"},
            ]
        )
        self.assertIn("[DONE]", body)
        self.assertEqual((stats.successful_requests, stats.failed_requests), (1, 0))
        self.assertEqual(captured[0][captured[0].index("--effort") + 1], "high")

    def test_stream_error_counts_as_failure(self) -> None:
        with self.assertLogs(server.logger, level="WARNING") as logs:
            _body, stats, _ = self._run_stream([{"_raise": "boom"}])
        self.assertEqual((stats.successful_requests, stats.failed_requests), (0, 1))
        self.assertTrue(any("stream ended status=error" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
