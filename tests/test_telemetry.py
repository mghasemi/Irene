"""Tests for Irene.telemetry — execution telemetry module.

Imports telemetry directly via importlib to avoid heavy package-level deps
(cvxpy, scipy) that live in other Irene submodules.
"""

import json
import os
import sys
import tempfile
import time
import importlib.util

# ---------------------------------------------------------------------------
# Load telemetry.py as a standalone module (bypasses Irene/__init__.py)
# ---------------------------------------------------------------------------
_telemetry_path = os.path.join(
    os.path.dirname(__file__), "..", "Irene", "telemetry.py"
)
_spec = importlib.util.spec_from_file_location("telemetry_standalone", _telemetry_path)
_telemetry_mod = importlib.util.module_from_spec(_spec)
sys.modules["telemetry_standalone"] = _telemetry_mod  # needed for dataclass introspection
_spec.loader.exec_module(_telemetry_mod)

timed = _telemetry_mod.timed
TelemetryContext = _telemetry_mod.TelemetryContext
get_telemetry = _telemetry_mod.get_telemetry
clear_telemetry = _telemetry_mod.clear_telemetry
export_json = _telemetry_mod.export_json


# ---------------------------------------------------------------------------
# Tests — get_telemetry() returns list[dict], not dataclass instances
# ---------------------------------------------------------------------------

class TestTimedDecorator:
    def setup_method(self):
        clear_telemetry()

    def test_basic_timing(self):
        @timed("test_phase")
        def slow_func():
            time.sleep(0.05)
            return 42

        result = slow_func()
        assert result == 42
        reports = get_telemetry()
        assert len(reports) >= 1
        test_report = [r for r in reports if r["phase"] == "test_phase"]
        assert len(test_report) >= 1
        assert test_report[0]["timings"]["test_phase"]["wall_clock_s"] >= 0.04

    def test_nested_timing(self):
        @timed("outer")
        def outer():
            time.sleep(0.02)
            inner()
            return "done"

        @timed("inner")
        def inner():
            time.sleep(0.01)
            return 1

        result = outer()
        assert result == "done"
        reports = get_telemetry()
        phases = [r["phase"] for r in reports]
        assert "outer" in phases
        assert "inner" in phases


class TestTelemetryContext:
    def setup_method(self):
        clear_telemetry()

    def test_context_manager_basic(self):
        with TelemetryContext("test_ctx") as ctx:
            time.sleep(0.02)
            ctx.set("key1", "value1")
            ctx.set("count", 42)

        reports = get_telemetry()
        assert len(reports) >= 1
        test_report = [r for r in reports if r["phase"] == "test_ctx"]
        assert len(test_report) >= 1
        report = test_report[0]
        assert report["metadata"]["key1"] == "value1"
        assert report["metadata"]["count"] == 42

    def test_context_manager_exception(self):
        try:
            with TelemetryContext("error_ctx") as ctx:
                time.sleep(0.01)
                ctx.set("before_error", True)
                raise ValueError("test error")
        except ValueError:
            pass

        reports = get_telemetry()
        test_report = [r for r in reports if r["phase"] == "error_ctx"]
        assert len(test_report) >= 1
        report = test_report[0]
        assert report["metadata"]["before_error"] is True


class TestTelemetryRegistry:
    def setup_method(self):
        clear_telemetry()

    def test_clear_telemetry(self):
        @timed("temp")
        def dummy():
            pass
        dummy()
        assert len(get_telemetry()) >= 1
        clear_telemetry()
        assert len(get_telemetry()) == 0


class TestJSONExport:
    def setup_method(self):
        clear_telemetry()

    def test_export_json_file(self):
        @timed("export_test")
        def work():
            time.sleep(0.01)
            return "ok"
        work()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_json(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "records" in data
            records = data["records"]
            assert len(records) >= 1
            phases = [entry["phase"] for entry in records]
            assert "export_test" in phases
        finally:
            os.unlink(path)

    def test_export_json_returns_path(self):
        @timed("path_test")
        def work():
            return 1
        work()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = export_json(path)
            assert os.path.isabs(result)
        finally:
            os.unlink(path)


class TestEnvGating:
    def setup_method(self):
        clear_telemetry()

    def test_env_gate_read_at_import(self):
        """Verify _TELEMETRY_ENABLED is a simple boolean read at module load."""
        assert hasattr(_telemetry_mod, "_TELEMETRY_ENABLED")
        # Default should be True (env not set to "0").
        assert _telemetry_mod._TELEMETRY_ENABLED is True
