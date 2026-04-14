import importlib.util
import json
import pathlib
import sys
import unittest
from unittest.mock import patch
from urllib.error import URLError


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / ".github" / "skills" / "lightrag-query" / "lightrag_query_tool.py"
MODULE_SPEC = importlib.util.spec_from_file_location("lightrag_query_tool", MODULE_PATH)
lightrag_query_tool = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = lightrag_query_tool
MODULE_SPEC.loader.exec_module(lightrag_query_tool)


class _DummyHeaders:
    def get_content_charset(self):
        return "utf-8"


class _DummyResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")
        self.headers = _DummyHeaders()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


class TestLightRAGQueryPayload(unittest.TestCase):
    def test_build_query_payload_includes_optional_fields(self):
        parser = lightrag_query_tool.build_parser()
        args = parser.parse_args([
            "query",
            "sdp hierarchy",
            "--mode",
            "hybrid",
            "--include-references",
            "--top-k",
            "5",
            "--hl-keywords",
            "sdp,moment",
            "--ll-keywords",
            "hierarchy",
        ])

        payload = lightrag_query_tool.build_query_payload(args)

        self.assertEqual(payload["query"], "sdp hierarchy")
        self.assertEqual(payload["mode"], "hybrid")
        self.assertTrue(payload["include_references"])
        self.assertEqual(payload["top_k"], 5)
        self.assertEqual(payload["hl_keywords"], ["sdp", "moment"])
        self.assertEqual(payload["ll_keywords"], ["hierarchy"])

    def test_format_query_text_includes_references(self):
        payload = {
            "response": "SDP hierarchy is used in polynomial optimization.",
            "references": [
                {"reference_id": "ref-1", "file_path": "notes/sdp.md"},
            ],
        }

        text = lightrag_query_tool.format_query_text(payload)

        self.assertIn("SDP hierarchy", text)
        self.assertIn("References:", text)
        self.assertIn("ref-1", text)


class TestLightRAGFallback(unittest.TestCase):
    def test_candidate_base_urls_include_fallback(self):
        client = lightrag_query_tool.LightRAGClient(
            base_url=lightrag_query_tool.DEFAULT_URL,
            timeout=5.0,
            fallback_url=lightrag_query_tool.DEFAULT_ALT_URL,
        )

        self.assertEqual(
            client.candidate_base_urls(),
            (lightrag_query_tool.DEFAULT_URL, lightrag_query_tool.DEFAULT_ALT_URL),
        )

    def test_post_json_tries_fallback_after_default_failure(self):
        client = lightrag_query_tool.LightRAGClient(
            base_url=lightrag_query_tool.DEFAULT_URL,
            timeout=5.0,
            fallback_url=lightrag_query_tool.DEFAULT_ALT_URL,
        )
        attempted_urls = []

        def fake_urlopen(request, timeout):
            attempted_urls.append(request.full_url)
            if len(attempted_urls) == 1:
                raise URLError("primary down")
            return _DummyResponse('{"response": "ok"}')

        with patch.object(lightrag_query_tool, "urlopen", side_effect=fake_urlopen):
            payload = client.post_json("/query", {"query": "sdp hierarchy", "mode": "mix"})

        self.assertEqual(payload["response"], "ok")
        self.assertEqual(
            attempted_urls,
            [
                "http://192.168.1.70:9621/query",
                "http://mghasemi.ddns.net:9621/query",
            ],
        )

    def test_build_client_disables_fallback_for_custom_primary_url(self):
        parser = lightrag_query_tool.build_parser()
        args = parser.parse_args(["--url", "http://example.com:9621", "query", "sdp hierarchy"])

        client = lightrag_query_tool.build_client(args)

        self.assertEqual(client.base_url, "http://example.com:9621")
        self.assertIsNone(client.fallback_url)

    def test_build_client_keeps_fallback_with_cli_alt_url(self):
        parser = lightrag_query_tool.build_parser()
        args = parser.parse_args(
            [
                "--url",
                "http://example.com:9621",
                "--alt-url",
                "http://backup.example.com:9621",
                "query",
                "sdp hierarchy",
            ]
        )

        client = lightrag_query_tool.build_client(args)

        self.assertEqual(client.base_url, "http://example.com:9621")
        self.assertEqual(client.fallback_url, "http://backup.example.com:9621")


@unittest.skipUnless(
    lightrag_query_tool.os.getenv("LIGHTRAG_RUN_LIVE_TESTS") == "1",
    "Set LIGHTRAG_RUN_LIVE_TESTS=1 to enable live LightRAG integration tests.",
)
class TestLightRAGLiveIntegration(unittest.TestCase):
    def setUp(self):
        parser = lightrag_query_tool.build_parser()
        args = parser.parse_args(["query", "sdp hierarchy"])
        self.client = lightrag_query_tool.build_client(args)

    def test_live_query_returns_response(self):
        payload = self.client.post_json("/query", {"query": "sdp hierarchy", "mode": "mix"})

        self.assertIsInstance(payload, dict)
        self.assertIn("response", payload)

    def test_live_query_data_returns_payload(self):
        payload = self.client.post_json("/query/data", {"query": "sdp hierarchy", "mode": "mix"})

        self.assertIsInstance(payload, dict)
        self.assertTrue(payload)


if __name__ == "__main__":
    unittest.main()
