import importlib.util
import json
import pathlib
import sys
import unittest
from unittest.mock import patch
from urllib.error import URLError


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / ".github" / "skills" / "zimi" / "zimi_tool.py"
MODULE_SPEC = importlib.util.spec_from_file_location("zimi_tool", MODULE_PATH)
zimi_tool = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = zimi_tool
MODULE_SPEC.loader.exec_module(zimi_tool)


class TestZimiNormalization(unittest.TestCase):
    def test_normalize_suggest_response_from_grouped_sources(self):
        payload = {
            "mathoverflow.net": [
                {"title": "Topological space", "path": "questions/1"},
            ],
            "wikipedia_en_mathematics_nopic": [
                {"title": "Topological data analysis", "path": "Topological_data_analysis"},
                {"title": "Topology", "path": "Topology"},
            ],
        }

        titles = zimi_tool.normalize_suggest_response(payload)

        self.assertEqual(
            titles,
            ["Topological space", "Topological data analysis", "Topology"],
        )

    def test_normalize_search_response_from_results_wrapper(self):
        payload = {
            "results": [
                {"zim": "wikipedia_en_mathematics_nopic", "path": "Semidefinite_programming", "title": "Semidefinite programming"},
                {"zim": "mathoverflow.net", "path": "questions/123", "title": "What is the SDP hierarchy?"},
            ],
            "total": 2,
        }

        results = zimi_tool.normalize_search_response(payload)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Semidefinite programming")

    def test_extract_read_text_prefers_content_field(self):
        payload = json.dumps(
            {
                "zim": "wikipedia_en_mathematics_nopic",
                "path": "Semidefinite_programming",
                "content": "Semidefinite programming is a subfield of convex optimization.",
                "truncated": True,
            }
        )

        text = zimi_tool.extract_read_text(payload)

        self.assertEqual(text, "Semidefinite programming is a subfield of convex optimization.")

    def test_extract_read_text_raises_on_error_payload(self):
        payload = json.dumps({"error": "ZIM not found"})

        with self.assertRaisesRegex(RuntimeError, "ZIM not found"):
            zimi_tool.extract_read_text(payload)


class TestZimiRanking(unittest.TestCase):
    def test_default_source_quality_prefers_reference_sources(self):
        query = "sdp hierarchy"
        results = [
            {
                "zim": "mathoverflow.net",
                "path": "questions/1",
                "title": "SDP hierarchy",
                "snippet": "Discussion of the SDP hierarchy.",
            },
            {
                "zim": "wikipedia_en_mathematics_nopic",
                "path": "Semidefinite_programming",
                "title": "Semidefinite programming",
                "snippet": "The SDP hierarchy appears in convex optimization and polynomial optimization.",
            },
        ]

        ranked = zimi_tool.best_search_results(query, results)

        self.assertEqual(ranked[0]["zim"], "wikipedia_en_mathematics_nopic")

    def test_explicit_preference_can_promote_requested_source(self):
        query = "sdp hierarchy"
        results = [
            {
                "zim": "planetmath.org",
                "path": "planetmath.org/semidefiniteprogramming",
                "title": "semidefinite programming",
                "snippet": "SDP hierarchy in polynomial optimization.",
            },
            {
                "zim": "wikipedia_en_mathematics_nopic",
                "path": "Semidefinite_programming",
                "title": "Semidefinite programming",
                "snippet": "SDP hierarchy in convex optimization.",
            },
        ]

        ranked = zimi_tool.best_search_results(query, results, preferred_sources=("planetmath.org",))

        self.assertEqual(ranked[0]["zim"], "planetmath.org")

    def test_source_preference_does_not_beat_irrelevant_result(self):
        query = "sdp hierarchy"
        results = [
            {
                "zim": "wikipedia_en_mathematics_nopic",
                "path": "K-means_clustering",
                "title": "K-means clustering",
                "snippet": "Clustering algorithm.",
            },
            {
                "zim": "mathoverflow.net",
                "path": "questions/331217/sdp-representation-of-ideal-polynomials",
                "title": "SDP representation of ideal polynomials",
                "snippet": "Discussion of the SDP hierarchy for positivstellensatz refutations.",
            },
        ]

        ranked = zimi_tool.best_search_results(
            query,
            results,
            preferred_sources=("wikipedia_en_mathematics_nopic",),
        )

        self.assertEqual(ranked[0]["zim"], "mathoverflow.net")

    def test_parse_source_preferences_supports_repeated_and_csv_values(self):
        preferences = zimi_tool.parse_source_preferences([
            "wikipedia_en_mathematics_nopic,planetmath.org",
            "mathoverflow.net",
        ])

        self.assertEqual(
            preferences,
            ("wikipedia_en_mathematics_nopic", "planetmath.org", "mathoverflow.net"),
        )


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


class TestZimiFallback(unittest.TestCase):
    def test_candidate_base_urls_include_fallback(self):
        client = zimi_tool.ZimiClient(
            base_url=zimi_tool.DEFAULT_URL,
            timeout=5.0,
            fallback_url=zimi_tool.DEFAULT_ALT_URL,
        )

        self.assertEqual(
            client.candidate_base_urls(),
            (zimi_tool.DEFAULT_URL, zimi_tool.DEFAULT_ALT_URL),
        )

    def test_get_text_tries_fallback_after_default_failure(self):
        client = zimi_tool.ZimiClient(
            base_url=zimi_tool.DEFAULT_URL,
            timeout=5.0,
            fallback_url=zimi_tool.DEFAULT_ALT_URL,
        )
        attempted_urls = []

        def fake_urlopen(url, timeout):
            attempted_urls.append(url)
            if len(attempted_urls) == 1:
                raise URLError("primary down")
            return _DummyResponse("[]")

        with patch.object(zimi_tool, "urlopen", side_effect=fake_urlopen):
            payload = client.get_text("/search", q="sdp hierarchy", limit=1)

        self.assertEqual(payload, "[]")
        self.assertEqual(
            attempted_urls,
            [
                "http://192.168.1.70:8899/search?q=sdp+hierarchy&limit=1",
                "http://mghasemi.ddns.net:8899/search?q=sdp+hierarchy&limit=1",
            ],
        )

    def test_build_client_disables_fallback_for_custom_primary_url(self):
        with patch.dict(
            zimi_tool.os.environ,
            {"ZIMI_URL": "http://example.com:8899", "ZIMI_ALT_URL": "http://mghasemi.ddns.net:8899"},
            clear=False,
        ):
            client = zimi_tool.build_client()

        self.assertEqual(client.base_url, "http://example.com:8899")
        self.assertIsNone(client.fallback_url)

    def test_build_client_keeps_fallback_when_cli_alt_url_is_provided(self):
        client = zimi_tool.build_client(
            base_url_override="http://example.com:8899",
            fallback_url_override="http://backup.example.com:8899",
        )

        self.assertEqual(client.base_url, "http://example.com:8899")
        self.assertEqual(client.fallback_url, "http://backup.example.com:8899")

    def test_build_client_drops_duplicate_cli_alt_url(self):
        client = zimi_tool.build_client(
            base_url_override="http://example.com:8899",
            fallback_url_override="http://example.com:8899",
        )

        self.assertEqual(client.base_url, "http://example.com:8899")
        self.assertIsNone(client.fallback_url)


@unittest.skipUnless(
    zimi_tool.os.getenv("ZIMI_RUN_LIVE_TESTS") == "1",
    "Set ZIMI_RUN_LIVE_TESTS=1 to enable live ZIMI integration tests.",
)
class TestZimiLiveIntegration(unittest.TestCase):
    def setUp(self):
        self.client = zimi_tool.build_client()

    def test_live_search_returns_normalized_results(self):
        results = self.client.search("sdp hierarchy", limit=5)

        self.assertIsInstance(results, list)
        self.assertTrue(results)
        top = results[0]
        self.assertIn("zim", top)
        self.assertIn("path", top)
        self.assertIn("title", top)

    def test_live_read_returns_text_for_top_result(self):
        results = self.client.search("sdp hierarchy", limit=1)
        self.assertTrue(results)

        top = results[0]
        text = self.client.read(top["zim"], top["path"], max_length=600)

        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())


if __name__ == "__main__":
    unittest.main()