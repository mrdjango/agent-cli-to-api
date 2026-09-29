import asyncio
import dataclasses
import json
import os
import stat
import tempfile
import textwrap
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from codex_gateway import server
from codex_gateway.stream_json_cli import (
    agy_result_error,
    extract_agy_delta,
    extract_usage_from_agy_result,
)

# Event shapes recorded from `agy --output-format stream-json` (agy 1.2.13).
_USAGE = {"input_tokens": 20176, "output_tokens": 43, "thinking_tokens": 0, "cache_read_tokens": 0, "total_tokens": 20219}


def _step(text: str | None = None, *, step_type: str = "agent_response", state: str = "ACTIVE") -> dict:
    step = {"conversation_id": "c1", "step_index": 1, "state": state, "step_type": step_type}
    if text is not None:
        step["text_delta"] = text
    return {"event": "step_update", "step_update": step}


def _result(response: str, *, status: str = "SUCCESS", **extra) -> dict:
    return {"event": "result", "result": {"conversation_id": "c1", "status": status, "response": response, "usage": _USAGE, **extra}}


class AgyRoutingTests(unittest.TestCase):
    def test_prefixes_and_bare_gemini_ids_route_to_antigravity(self) -> None:
        cases = {
            "agy:gemini-3.8-flash-high": ("antigravity", "gemini-3.8-flash-high"),
            "antigravity:gemini-3.1-pro": ("antigravity", "gemini-3.1-pro"),
            "agy:claude-sonnet-4-6": ("antigravity", "claude-sonnet-4-6"),
            "agy:gpt-oss-120b-medium": ("antigravity", "gpt-oss-120b-medium"),
            "gemini-3.8-flash-low": ("antigravity", "gemini-3.8-flash-low"),
            "agy:": ("antigravity", None),
            "agy": ("antigravity", None),
        }
        for requested, expected in cases.items():
            with self.subTest(requested=requested):
                self.assertEqual(server._parse_provider_model(requested), expected)

    def test_other_providers_keep_their_names(self) -> None:
        # Bare Claude names stay with the Claude provider; the gemini: prefix stays with the Gemini CLI.
        self.assertEqual(server._parse_provider_model("claude-sonnet-4-6"), ("claude", "claude-sonnet-4-6"))
        self.assertEqual(server._parse_provider_model("gemini:gemini-3"), ("gemini", "gemini-3"))
        self.assertEqual(server._parse_provider_model("gpt-5.6-sol"), ("codex", "gpt-5.6-sol"))

    def test_provider_name_aliases(self) -> None:
        self.assertEqual(server._normalize_provider("agy"), "antigravity")
        self.assertEqual(server._normalize_provider("Antigravity"), "antigravity")

    def test_strict_mode(self) -> None:
        strict = dataclasses.replace(server.settings, strict_models=True)
        with mock.patch.object(server, "settings", strict):
            for ok in ("gemini-3.8-flash-high", "agy:claude-opus-4-6-thinking"):
                with self.subTest(ok=ok):
                    self.assertIsNone(server._strict_model_error(ok, *server._parse_provider_model(ok)))
            for bad in ("agy:", "agy"):
                with self.subTest(bad=bad):
                    self.assertIsNotNone(server._strict_model_error(bad, *server._parse_provider_model(bad)))


class AgyCommandTests(unittest.TestCase):
    def test_prompt_is_attached_to_print_flag(self) -> None:
        cmd = server._agy_cli_cmd("gemini-3.8-flash-low", "-rf looks like a flag")
        self.assertEqual(cmd[-1], "--print=-rf looks like a flag")
        self.assertEqual(cmd[cmd.index("--model") + 1], "gemini-3.8-flash-low")
        self.assertIn("--disable-slash-commands", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "stream-json")

    def test_effort_only_for_base_model_ids(self) -> None:
        base = server._agy_cli_cmd("gemini-3.1-pro", "hi", "high")
        self.assertEqual(base[base.index("--effort") + 1], "high")
        # agy rejects --effort with an ID that already pins one.
        self.assertNotIn("--effort", server._agy_cli_cmd("gemini-3.1-pro-low", "hi", "high"))
        self.assertNotIn("--effort", server._agy_cli_cmd("gemini-3.1-pro", "hi", None))

    def test_normalize_effort(self) -> None:
        for raw, expected in {"max": "max", "xhigh": "high", "none": "low", "Medium": "medium", "bogus": None, None: None}.items():
            with self.subTest(raw=raw):
                self.assertEqual(server._normalize_agy_effort(raw), expected)

    def test_sandbox_toggle(self) -> None:
        with mock.patch.object(server, "settings", dataclasses.replace(server.settings, agy_sandbox=True)):
            self.assertIn("--sandbox", server._agy_cli_cmd("m", "hi"))
        with mock.patch.object(server, "settings", dataclasses.replace(server.settings, agy_sandbox=False)):
            self.assertNotIn("--sandbox", server._agy_cli_cmd("m", "hi"))


class AgyEventTests(unittest.TestCase):
    def test_deltas_come_only_from_agent_response_steps(self) -> None:
        self.assertEqual(extract_agy_delta(_step("pong")), "pong")
        self.assertEqual(extract_agy_delta(_step(None, step_type="user_input", state="DONE")), "")
        self.assertEqual(extract_agy_delta(_step("x", step_type="tool")), "")
        self.assertEqual(extract_agy_delta(_result("pong")), "")

    def test_usage_from_result(self) -> None:
        self.assertEqual(
            extract_usage_from_agy_result(_result("ok")),
            {"prompt_tokens": 20176, "completion_tokens": 43, "total_tokens": 20219},
        )
        self.assertIsNone(extract_usage_from_agy_result(_step("ok")))

    def test_result_errors(self) -> None:
        self.assertIsNone(agy_result_error(_result("ok")))
        self.assertIn("not recognized", agy_result_error(_result("", status="ERROR", error="model x is not recognized")))
        denied = _result("", denied_actions=[{"action": "command", "display_name": "RunCommand"}])
        self.assertIn("RunCommand", agy_result_error(denied))
        # A denied tool is fine when the model still answered.
        self.assertIsNone(agy_result_error(_result("answer", denied_actions=[{"action": "command"}])))

    def test_guard_maps_invalid_model_to_client_error(self) -> None:
        async def run() -> None:
            async def events():
                yield _result("", status="ERROR", error='invalid model selection (--model "x")')

            async for _ in server._guard_agy_result(events()):
                pass

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(run())
        self.assertEqual(server._extract_upstream_status_code(ctx.exception), 400)


def _fake_agy(events: list[dict], exit_code: int = 0) -> str:
    """Write an executable that records its argv and prints the given events as NDJSON."""
    d = tempfile.mkdtemp(prefix="fake-agy-")
    path = os.path.join(d, "agy")
    body = "\n".join(json.dumps(e) for e in events)
    with open(path, "w") as f:
        f.write(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json, os, sys
                with open({os.path.join(d, "argv.json")!r}, "w") as f:
                    json.dump(sys.argv[1:], f)
                with open({os.path.join(d, "cwd.txt")!r}, "w") as f:
                    f.write(os.getcwd())
                print({body!r})
                sys.exit({exit_code})
                """
            )
        )
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


class AgyEndToEndTests(unittest.TestCase):
    def _client(self, agy_bin: str) -> tuple[TestClient, mock._patch]:
        patched = dataclasses.replace(
            server.settings,
            provider="auto",
            agy_bin=agy_bin,
            bearer_token=None,
            log_render_markdown=False,
            log_request_curl=False,
        )
        patcher = mock.patch.object(server, "settings", patched)
        patcher.start()
        self.addCleanup(patcher.stop)
        return TestClient(server.app)

    def test_non_stream_chat_completion(self) -> None:
        agy = _fake_agy([{"event": "init", "init": {"model": "gemini-3.8-flash-low"}}, _step("po"), _step("ng"), _result("pong")])
        client = self._client(agy)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gemini-3.8-flash-low", "messages": [{"role": "user", "content": "ping"}]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "pong")
        self.assertEqual(body["model"], "gemini-3.8-flash-low")
        self.assertEqual(body["usage"]["prompt_tokens"], 20176)
        with open(os.path.join(os.path.dirname(agy), "argv.json")) as f:
            argv = json.load(f)
        self.assertEqual(argv[argv.index("--model") + 1], "gemini-3.8-flash-low")
        self.assertTrue(argv[-1].startswith("--print=") and "ping" in argv[-1])

    def test_stream_chat_completion(self) -> None:
        agy = _fake_agy([_step("Hello"), _step(" world"), _result("Hello world")])
        client = self._client(agy)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "agy:claude-sonnet-4-6", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            lines = [l[len("data: "):] for l in resp.iter_lines() if l.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        chunks = [json.loads(l) for l in lines[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        self.assertEqual(text, "Hello world")

    def _run_cwd(self, agy: str, body: dict) -> str:
        client = self._client(agy)
        resp = client.post("/v1/chat/completions", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        with open(os.path.join(os.path.dirname(agy), "cwd.txt")) as f:
            return os.path.realpath(f.read())

    def test_web_tools_blocked_unless_search_requested(self) -> None:
        agy = _fake_agy([_step("ok"), _result("ok")])
        msgs = [{"role": "user", "content": "hi"}]

        cwd = self._run_cwd(agy, {"model": "gemini-3.8-flash-low", "messages": msgs})
        with open(os.path.join(cwd, ".agents", "hooks.json")) as f:
            hooks = json.load(f)
        group = hooks["agent-cli-to-api-no-web"]["PreToolUse"][0]
        self.assertEqual(group["matcher"], "search_web|read_url_content")
        self.assertIn('"decision": "deny"', group["hooks"][0]["command"])

        web_cwd = self._run_cwd(
            agy, {"model": "gemini-3.8-flash-low", "messages": msgs, "tools": [{"type": "web_search"}]}
        )
        self.assertNotEqual(web_cwd, cwd)
        self.assertFalse(os.path.exists(os.path.join(web_cwd, ".agents", "hooks.json")))

    def test_operator_can_refuse_search(self) -> None:
        agy = _fake_agy([_step("ok"), _result("ok")])
        client = self._client(agy)
        with mock.patch.object(server, "settings", dataclasses.replace(server.settings, enable_search=False)):
            client.post(
                "/v1/chat/completions",
                json={"model": "gemini-3.8-flash-low", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "web_search"}]},
            )
        with open(os.path.join(os.path.dirname(agy), "cwd.txt")) as f:
            self.assertTrue(os.path.isfile(os.path.join(f.read(), ".agents", "hooks.json")))

    def test_invalid_model_is_a_400(self) -> None:
        agy = _fake_agy([_result("", status="ERROR", error='invalid model selection (--model "gemini-nope")')], exit_code=1)
        client = self._client(agy)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gemini-nope", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("invalid model selection", resp.text)


if __name__ == "__main__":
    unittest.main()
