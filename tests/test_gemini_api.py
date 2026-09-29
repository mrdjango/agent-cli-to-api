import asyncio
import dataclasses
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from codex_gateway import server
from codex_gateway.gemini_compat import (
    GeminiRequestError,
    gemini_request_to_chat_request,
    openai_chat_completion_to_gemini_response,
    openai_stream_to_gemini_responses,
    split_model_action,
)

from tests.test_antigravity import _fake_agy, _result, _step


class ModelActionTests(unittest.TestCase):
    def test_split(self) -> None:
        self.assertEqual(split_model_action("gemini-3.8-flash:generateContent"), ("gemini-3.8-flash", "generateContent"))
        self.assertEqual(split_model_action("models/gpt-5.6-sol:streamGenerateContent"), ("gpt-5.6-sol", "streamGenerateContent"))
        # Provider prefixes contain ':' too.
        self.assertEqual(split_model_action("agy:claude-sonnet-4-6:countTokens"), ("agy:claude-sonnet-4-6", "countTokens"))
        for bad in ("gemini-3.8-flash", "gemini:embedContent", ":generateContent"):
            with self.subTest(bad=bad), self.assertRaises(GeminiRequestError):
                split_model_action(bad)


class RequestConversionTests(unittest.TestCase):
    def test_contents_system_and_generation_config(self) -> None:
        req = gemini_request_to_chat_request(
            {
                "systemInstruction": {"parts": [{"text": "Be terse."}]},
                "contents": [
                    {"role": "user", "parts": [{"text": "Hi"}]},
                    {"role": "model", "parts": [{"text": "hidden", "thought": True}, {"text": "Hello"}]},
                    {"role": "user", "parts": [{"text": "Again"}, {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}]},
                ],
                "generationConfig": {
                    "maxOutputTokens": 64,
                    "temperature": 0.2,
                    "stopSequences": ["END"],
                    "thinkingConfig": {"thinkingLevel": "high"},
                },
            },
            model="gemini-3.8-flash-low",
            stream=True,
        )
        self.assertEqual([m.role for m in req.messages], ["system", "user", "assistant", "user"])
        self.assertEqual(req.messages[0].content, "Be terse.")
        self.assertEqual(req.messages[2].content, "Hello")
        self.assertEqual(req.messages[3].content[1], {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})
        self.assertEqual((req.model, req.stream, req.max_tokens), ("gemini-3.8-flash-low", True, 64))
        extra = req.model_extra
        self.assertEqual((extra["temperature"], extra["stop"], extra["reasoning_effort"]), (0.2, ["END"], "high"))

    def test_snake_case_fields(self) -> None:
        req = gemini_request_to_chat_request(
            {"system_instruction": "Be terse.", "contents": [{"parts": [{"text": "Hi"}]}], "generation_config": {"max_output_tokens": 5}},
            model="m",
            stream=False,
        )
        self.assertEqual(req.messages[0].content, "Be terse.")
        self.assertEqual(req.max_tokens, 5)

    def test_function_call_round_trip_and_tools(self) -> None:
        req = gemini_request_to_chat_request(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "add 1 and 2"}]},
                    {"role": "model", "parts": [{"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "add", "response": {"result": 3}}}]},
                ],
                "tools": [
                    {"functionDeclarations": [{"name": "add", "parameters": {"type": "OBJECT", "properties": {"a": {"type": "INTEGER"}}}}]},
                    {"googleSearch": {}},
                ],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["add"]}},
            },
            model="gpt-5.6-sol",
            stream=False,
        )
        assistant, tool = req.messages[1], req.messages[2]
        call = assistant.model_extra["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"a": 1, "b": 2})
        # The functionResponse (no id) is paired with the call by name.
        self.assertEqual((tool.role, tool.model_extra["tool_call_id"], json.loads(tool.content)), ("tool", call["id"], {"result": 3}))
        tools = req.model_extra["tools"]
        self.assertEqual(tools[0]["function"]["parameters"], {"type": "object", "properties": {"a": {"type": "integer"}}})
        self.assertEqual(tools[1], {"type": "web_search"})
        self.assertEqual(req.model_extra["tool_choice"], {"type": "function", "function": {"name": "add"}})

    def test_json_schema_output(self) -> None:
        req = gemini_request_to_chat_request(
            {
                "contents": [{"parts": [{"text": "x"}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "OBJECT", "properties": {"n": {"type": "STRING"}}}},
            },
            model="m",
            stream=False,
        )
        schema = req.model_extra["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema, {"type": "object", "properties": {"n": {"type": "string"}}})
        self.assertEqual(server._response_json_schema(req), schema)

    def test_rejects_bad_input(self) -> None:
        for body in ({}, {"contents": []}, {"contents": [{"role": "user", "parts": [{"fileData": {"fileUri": "gs://x"}}]}]}):
            with self.subTest(body=body), self.assertRaises(GeminiRequestError):
                gemini_request_to_chat_request(body, model="m", stream=False)


class ResponseConversionTests(unittest.TestCase):
    def test_chat_completion_to_gemini(self) -> None:
        out = openai_chat_completion_to_gemini_response(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "message": {
                            "content": "Sure.",
                            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
            model="gpt-5.6-sol",
        )
        cand = out["candidates"][0]
        self.assertEqual(cand["content"]["parts"], [{"text": "Sure."}, {"functionCall": {"name": "add", "args": {"a": 1}, "id": "call_1"}}])
        self.assertEqual(cand["finishReason"], "STOP")
        self.assertEqual(out["usageMetadata"], {"promptTokenCount": 10, "candidatesTokenCount": 3, "totalTokenCount": 13})
        self.assertEqual((out["modelVersion"], out["responseId"]), ("gpt-5.6-sol", "chatcmpl-1"))

    def test_stream_conversion(self) -> None:
        def chunk(**choice) -> str:
            return "data: " + json.dumps({"id": "c1", "choices": [{"index": 0, **choice}]}) + "\n\n"

        async def source():
            yield ": ping\n\n"
            yield chunk(delta={"content": "Hel"}, finish_reason=None)
            yield chunk(delta={"content": "lo"}, finish_reason=None)
            yield chunk(delta={}, finish_reason="length")
            yield "data: " + json.dumps({"id": "c1", "choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}) + "\n\n"
            yield "data: [DONE]\n\n"

        async def collect():
            return [r async for r in openai_stream_to_gemini_responses(source(), model="m")]

        items = asyncio.run(collect())
        self.assertEqual([i["candidates"][0]["content"]["parts"] for i in items], [[{"text": "Hel"}], [{"text": "lo"}], []])
        self.assertEqual(items[-1]["candidates"][0]["finishReason"], "MAX_TOKENS")
        self.assertEqual(items[-1]["usageMetadata"]["totalTokenCount"], 3)


class GeminiEndpointTests(unittest.TestCase):
    def _client(self, **overrides) -> TestClient:
        patched = dataclasses.replace(
            server.settings,
            provider="auto",
            bearer_token="devtoken",
            log_render_markdown=False,
            log_request_curl=False,
            **overrides,
        )
        patcher = mock.patch.object(server, "settings", patched)
        patcher.start()
        self.addCleanup(patcher.stop)
        return TestClient(server.app)

    def test_auth(self) -> None:
        client = self._client()
        self.assertEqual(client.get("/v1beta/models").json()["error"]["status"], "UNAUTHENTICATED")
        self.assertEqual(client.get("/v1beta/models", headers={"x-goog-api-key": "bad"}).status_code, 403)
        self.assertEqual(client.get("/v1beta/models", headers={"x-goog-api-key": "devtoken"}).status_code, 200)
        self.assertEqual(client.get("/v1beta/models?key=devtoken").status_code, 200)
        self.assertEqual(client.get("/v1beta/models", headers={"authorization": "Bearer devtoken"}).status_code, 200)

    def test_list_get_and_count(self) -> None:
        client = self._client()
        h = {"x-goog-api-key": "devtoken"}
        names = [m["name"] for m in client.get("/v1beta/models", headers=h).json()["models"]]
        self.assertIn("models/gpt-5.6-sol", names)
        self.assertNotIn("models/default", names)
        self.assertEqual(client.get("/v1beta/models/gemini-3.8-flash-low", headers=h).json()["name"], "models/gemini-3.8-flash-low")
        count = client.post("/v1beta/models/m:countTokens", headers=h, json={"contents": [{"parts": [{"text": "hello world"}]}]})
        self.assertGreater(count.json()["totalTokens"], 0)
        self.assertEqual(client.post("/v1beta/models/m:embedContent", headers=h, json={}).status_code, 404)

    def test_generate_and_stream_via_agy(self) -> None:
        agy = _fake_agy([_step("po"), _step("ng"), _result("pong")])
        client = self._client(agy_bin=agy)
        h = {"x-goog-api-key": "devtoken"}
        body = {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]}

        resp = client.post("/v1beta/models/gemini-3.8-flash-low:generateContent", headers=h, json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["candidates"][0]["content"]["parts"], [{"text": "pong"}])
        self.assertEqual(resp.json()["usageMetadata"]["promptTokenCount"], 20176)

        with client.stream("POST", "/v1beta/models/gemini-3.8-flash-low:streamGenerateContent?alt=sse", headers=h, json=body) as resp:
            self.assertEqual(resp.headers["content-type"].split(";")[0], "text/event-stream")
            items = [json.loads(l[len("data: "):]) for l in resp.iter_lines() if l.startswith("data: ")]
        self.assertEqual("".join(p.get("text", "") for i in items for p in i["candidates"][0]["content"]["parts"]), "pong")
        self.assertEqual(items[-1]["candidates"][0]["finishReason"], "STOP")
        self.assertIn("usageMetadata", items[-1])

        # Without alt=sse the Gemini API streams a single JSON array.
        resp = client.post("/v1beta/models/gemini-3.8-flash-low:streamGenerateContent", headers=h, json=body)
        self.assertEqual(len(resp.json()), 3)

    def test_errors_use_gemini_format(self) -> None:
        agy = _fake_agy([_result("", status="ERROR", error='invalid model selection (--model "gemini-nope")')], exit_code=1)
        client = self._client(agy_bin=agy)
        resp = client.post(
            "/v1beta/models/gemini-nope:generateContent",
            headers={"x-goog-api-key": "devtoken"},
            json={"contents": [{"parts": [{"text": "hi"}]}]},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["status"], "INVALID_ARGUMENT")
        bad = client.post("/v1beta/models/gemini-nope:generateContent", headers={"x-goog-api-key": "devtoken"}, json={"contents": []})
        self.assertEqual(bad.status_code, 400)

    def test_query_key_is_redacted(self) -> None:
        self.assertEqual(server._redact_query_key("alt=sse&key=secret"), "alt=sse&key=<redacted>")
        self.assertEqual(server._redact_query_key("key=secret"), "key=<redacted>")
        self.assertEqual(server._redact_query_key("monkey=1"), "monkey=1")


if __name__ == "__main__":
    unittest.main()
