import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / ".github" / "skills" / "wolfram-alpha" / "wolfram_alpha_tool.py"
MODULE_SPEC = importlib.util.spec_from_file_location("wolfram_alpha_tool", MODULE_PATH)
wolfram_alpha_tool = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = wolfram_alpha_tool
MODULE_SPEC.loader.exec_module(wolfram_alpha_tool)


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


VERIFY_XML = """
<queryresult success="true" error="false" numpods="2" datatypes="Math,Number" timedout="" timedoutpods="" timing="0.42" parsetiming="0.01" parsetimedout="false">
  <pod title="Input interpretation" scanner="Identity" id="Input" position="100" error="false" numsubpods="1">
    <subpod title="">
      <plaintext>sin(30)</plaintext>
    </subpod>
  </pod>
  <pod title="Result" scanner="Numeric" id="Result" position="200" error="false" numsubpods="1" primary="true">
    <subpod title="">
      <plaintext>1 / 2</plaintext>
      <minput>1/2</minput>
      <moutput>1/2</moutput>
    </subpod>
    <states count="1">
      <state name="Decimal form" input="Result__Decimal form" />
    </states>
  </pod>
  <assumptions count="1">
    <assumption type="AngleUnit" word="30" template="Assuming degrees. Use radians instead" count="2">
      <value name="Degrees" desc="degrees" input="*A.Sin-_*AngleUnit.Degrees-" />
      <value name="Radians" desc="radians" input="*A.Sin-_*AngleUnit.Radians-" />
    </assumption>
  </assumptions>
  <warnings count="1">
    <reinterpret text="Using closest Wolfram|Alpha interpretation:" new="sin(30 degrees)">
      <alternative>sin(30 radians)</alternative>
    </reinterpret>
  </warnings>
  <sources count="1">
    <source text="Wolfram Functions" url="https://example.com/functions" />
  </sources>
</queryresult>
""".strip()


VALIDATE_XML = """
<validatequeryresult success="true" error="false" timing="0.02" parsetiming="0.01">
  <assumptions count="1">
    <assumption type="Function" word="log" template="Assuming natural log. Use base 10 log instead" count="2">
      <value name="Log" desc="the natural logarithm" input="*FunClash.log-_*Log.Log10-" />
      <value name="Log10" desc="the base 10 logarithm" input="*FunClash.log-_*Log10.Log-" />
    </assumption>
  </assumptions>
</validatequeryresult>
""".strip()


class TestWolframAlphaClient(unittest.TestCase):
    def test_build_client_reads_env_configuration(self):
        parser = wolfram_alpha_tool.build_parser()
        args = parser.parse_args(["answer", "integrate x^2"])

        with patch.dict(
            wolfram_alpha_tool.os.environ,
            {
                "WOLFRAM_ALPHA_APPID": "ENV-APPID",
                "WOLFRAM_ALPHA_TIMEOUT": "11.5",
            },
            clear=False,
        ):
            client = wolfram_alpha_tool.build_client(args)

        self.assertEqual(client.appid, "ENV-APPID")
        self.assertEqual(client.timeout, 11.5)

    def test_build_client_prefers_cli_appid(self):
        parser = wolfram_alpha_tool.build_parser()
        args = parser.parse_args(["--appid", "CLI-APPID", "answer", "integrate x^2"])

        with patch.dict(wolfram_alpha_tool.os.environ, {"WOLFRAM_ALPHA_APPID": "ENV-APPID"}, clear=False):
            client = wolfram_alpha_tool.build_client(args)

        self.assertEqual(client.appid, "CLI-APPID")

    def test_short_answer_uses_result_api_parameter(self):
        parser = wolfram_alpha_tool.build_parser()
        args = parser.parse_args(["--appid", "APPID", "answer", "integrate x^2"])
        client = wolfram_alpha_tool.build_client(args)
        seen_urls = []

        def fake_urlopen(url, timeout):
            seen_urls.append(url)
            return _DummyResponse("x^3/3")

        with patch.object(wolfram_alpha_tool, "urlopen", side_effect=fake_urlopen):
            answer = client.short_answer("integrate x^2")

        self.assertEqual(answer, "x^3/3")
        self.assertEqual(
            seen_urls,
            ["https://api.wolframalpha.com/v1/result?appid=APPID&i=integrate+x%5E2"],
        )


class TestWolframAlphaNormalization(unittest.TestCase):
    def test_normalize_query_result_xml_extracts_pods_and_assumptions(self):
        payload = wolfram_alpha_tool.normalize_query_result_xml(VERIFY_XML)

        self.assertTrue(payload["success"])
        self.assertEqual(payload["datatypes"], ["Math", "Number"])
        self.assertEqual(payload["pods"][1]["id"], "Result")
        self.assertEqual(payload["pods"][1]["subpods"][0]["plaintext"], "1 / 2")
        self.assertEqual(payload["assumptions"][0]["type"], "AngleUnit")
        self.assertEqual(payload["warnings"][0]["kind"], "reinterpret")

    def test_extract_primary_plaintext_prefers_primary_pod(self):
        payload = wolfram_alpha_tool.normalize_query_result_xml(VERIFY_XML)

        answer = wolfram_alpha_tool.extract_primary_plaintext(payload)

        self.assertEqual(answer, "1 / 2")

    def test_format_verify_text_includes_warning_and_assumption(self):
        payload = wolfram_alpha_tool.normalize_query_result_xml(VERIFY_XML)

        text = wolfram_alpha_tool.format_verify_text(payload)

        self.assertIn("Primary result: 1 / 2", text)
        self.assertIn("Warnings:", text)
        self.assertIn("Assumptions:", text)
        self.assertIn("sin(30 degrees)", text)

    def test_normalize_validate_result_xml_extracts_parse_assumptions(self):
        payload = wolfram_alpha_tool.normalize_validate_result_xml(VALIDATE_XML)

        self.assertTrue(payload["success"])
        self.assertEqual(payload["assumptions"][0]["type"], "Function")

    def test_build_verify_params_supports_repeated_values(self):
        parser = wolfram_alpha_tool.build_parser()
        args = parser.parse_args(
            [
                "verify",
                "pi",
                "--assumption",
                "*C.pi-_*Movie-",
                "--includepodid",
                "Result",
                "--includepodid",
                "DecimalApproximation",
                "--podstate",
                "DecimalApproximation__More digits",
            ]
        )

        params = wolfram_alpha_tool.build_verify_params(args)

        self.assertIn(("assumption", "*C.pi-_*Movie-"), params)
        self.assertIn(("includepodid", "Result"), params)
        self.assertIn(("includepodid", "DecimalApproximation"), params)
        self.assertIn(("podstate", "DecimalApproximation__More digits"), params)

    def test_build_verify_params_applies_symbolic_profile_defaults(self):
        parser = wolfram_alpha_tool.build_parser()
        args = parser.parse_args(["verify", "integrate x^2", "--profile", "symbolic"])

        params = wolfram_alpha_tool.build_verify_params(args)

        self.assertIn(("includepodid", "Result"), params)
        self.assertIn(("podtitle", "Exact result"), params)
        self.assertIn(("podtitle", "Possible intermediate steps"), params)
        self.assertIn(("format", wolfram_alpha_tool.DEFAULT_QUERY_FORMATS), params)


@unittest.skipUnless(
    wolfram_alpha_tool.os.getenv("WOLFRAM_ALPHA_RUN_LIVE_TESTS") == "1" and wolfram_alpha_tool.os.getenv("WOLFRAM_ALPHA_APPID"),
    "Set WOLFRAM_ALPHA_RUN_LIVE_TESTS=1 and WOLFRAM_ALPHA_APPID to enable live Wolfram|Alpha tests.",
)
class TestWolframAlphaLiveIntegration(unittest.TestCase):
    def setUp(self):
        parser = wolfram_alpha_tool.build_parser()
        args = parser.parse_args(["answer", "integrate x^2"])
        self.client = wolfram_alpha_tool.build_client(args)

    def test_live_answer_returns_text(self):
        answer = self.client.short_answer("integrate x^2")

        self.assertIsInstance(answer, str)
        self.assertTrue(answer)

    def test_live_verify_returns_pods(self):
        raw = self.client.query([("input", "sin(30)"), ("format", wolfram_alpha_tool.DEFAULT_QUERY_FORMATS)])
        payload = wolfram_alpha_tool.normalize_query_result_xml(raw)

        self.assertIn("pods", payload)
        self.assertTrue(payload["pods"])


if __name__ == "__main__":
    unittest.main()