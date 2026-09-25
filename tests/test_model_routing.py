import asyncio
import dataclasses
import unittest
from unittest import mock

from codex_gateway import server


class ModelRoutingTests(unittest.TestCase):
    def test_codex_models_route_to_codex(self) -> None:
        self.assertEqual(server._parse_provider_model("gpt-5.6-terra"), ("codex", "gpt-5.6-terra"))
        self.assertEqual(server._parse_provider_model(""), ("codex", None))

    def test_prefixed_claude_models(self) -> None:
        self.assertEqual(server._parse_provider_model("claude:opus"), ("claude", "opus"))
        self.assertEqual(server._parse_provider_model("claude:opus[1m]"), ("claude", "opus[1m]"))
        self.assertEqual(server._parse_provider_model("claude:opus-5.5"), ("claude", "claude-opus-5-5"))
        self.assertEqual(
            server._parse_provider_model("claude:claude-haiku-4-5-20251001"),
            ("claude", "claude-haiku-4-5-20251001"),
        )
        self.assertEqual(server._parse_provider_model("claude:"), ("claude", None))

    def test_anthropic_native_names_route_to_claude(self) -> None:
        cases = {
            "claude-opus-5-5": "claude-opus-5-5",
            "claude-sonnet-5": "claude-sonnet-5",
            "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
            "claude-opus-5-5[1m]": "claude-opus-5-5[1m]",
            "sonnet": "sonnet",
            "opus-5.5": "claude-opus-5-5",
            "opus-5-5": "claude-opus-5-5",
            "sonnet-5": "claude-sonnet-5",
            "haiku-4-5": "claude-haiku-4-5",
            "Opus 5.5": "claude-opus-5-5",
        }
        for requested, expected in cases.items():
            with self.subTest(requested=requested):
                self.assertEqual(server._parse_provider_model(requested), ("claude", expected))

    def test_strict_mode_rejects_missing_and_unknown_models(self) -> None:
        strict = dataclasses.replace(server.settings, strict_models=True)
        with mock.patch.object(server, "settings", strict):
            def error_for(model: str) -> str | None:
                provider, provider_model = server._parse_provider_model(model)
                return server._strict_model_error(model, provider, provider_model)

            for ok in ("gpt-5.6-terra", "o3", "claude-opus-5-5", "opus-5.5", "claude:sonnet", "haiku"):
                with self.subTest(ok=ok):
                    self.assertIsNone(error_for(ok))
            for bad in ("", "default", "llama-3", "claude:", "claude", "gemini:gemini-3", "cursor:auto"):
                with self.subTest(bad=bad):
                    self.assertIsNotNone(error_for(bad))

    def test_strict_mode_off_keeps_legacy_routing(self) -> None:
        legacy = dataclasses.replace(server.settings, strict_models=False)
        with mock.patch.object(server, "settings", legacy):
            self.assertIsNone(server._strict_model_error("", "codex", None))
            self.assertIsNone(server._strict_model_error("llama-3", "codex", "llama-3"))

    def test_strict_mode_blocks_claude_cli_substitution(self) -> None:
        async def run(requested: str, resolved: str) -> list[dict]:
            async def events():
                yield {"type": "system", "subtype": "init", "model": resolved}
                yield {"type": "result", "result": "ok"}

            return [e async for e in server._guard_claude_substitution(events(), requested)]

        strict = dataclasses.replace(server.settings, strict_models=True)
        with mock.patch.object(server, "settings", strict):
            with self.assertRaisesRegex(RuntimeError, "model_substituted"):
                asyncio.run(run("claude-opus-4-1", "claude-opus-5-5"))
            # Same model (date suffix / [1m] differences) and aliases pass.
            self.assertEqual(len(asyncio.run(run("claude-haiku-4-5", "claude-haiku-4-5-20251001"))), 2)
            self.assertEqual(len(asyncio.run(run("claude-opus-5-5[1m]", "claude-opus-5-5[1m]"))), 2)
            self.assertEqual(len(asyncio.run(run("opus", "claude-opus-5-5"))), 2)
        self.assertEqual(
            server._extract_upstream_status_code(RuntimeError("[claude-code:model_substituted] x")), 400
        )

    def test_unrecognized_claude_model_is_a_client_error(self) -> None:
        err = RuntimeError('[claude-code:unrecognized_model] {"model":"opus-9"}')
        self.assertEqual(server._extract_upstream_status_code(err), 400)


if __name__ == "__main__":
    unittest.main()
