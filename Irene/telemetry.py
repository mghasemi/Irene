"""
Execution telemetry for IreneRewrite SDP pipeline.

Provides zero-overhead phase timing, structured diagnostics collection,
and JSON export for benchmarking.  Controlled by the environment variable
``IRENE_TELEMETRY`` (default ``"1"`` -- enabled).  Set to ``"0"`` to disable
all telemetry with no runtime cost.

Public API
----------
- ``@timed(phase)`` -- decorator that logs wall-clock time for a named phase.
- ``TelemetryContext`` -- context manager that collects structured metrics.
- ``get_telemetry()`` -- retrieve the current session's telemetry dict.
- ``clear_telemetry()`` -- reset the session state.
- ``export_json(path)`` -- write telemetry to a JSON file.

Usage example
-------------
>>> from Irene.telemetry import timed, TelemetryContext, export_json
>>> @timed("basis_construction")
... def build_basis(deg): ...
>>> with TelemetryContext("sdp_solve", monomial_count=42):
...     solve_sdp()
>>> export_json("benchmark.json")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Environment gating -- zero overhead when disabled
# ---------------------------------------------------------------------------

_TELEMETRY_ENABLED: bool = os.environ.get("IRENE_TELEMETRY", "1") != "0"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseTiming:
    """Wall-clock timing for a single pipeline phase."""
    wall_clock_s: float = 0.0


@dataclass
class TelemetryRecord:
    """Structured telemetry record for one SDP relaxation run.

    Attributes
    ----------
    phase : str
        Human-readable phase name (e.g. ``"basis_construction"``).
    timings : dict[str, PhaseTiming]
        Per-sub-phase wall-clock times accumulated via the ``@timed`` decorator.
    metadata : dict[str, Any]
        Arbitrary key-value pairs set by the caller (monomial counts, block dims, etc.).
    """
    phase: str = ""
    timings: Dict[str, PhaseTiming] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"phase": self.phase}
        if self.timings:
            result["timings"] = {k: asdict(v) for k, v in self.timings.items()}
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# Module-level singleton -- holds the current session's telemetry.
_session_records: list[TelemetryRecord] = []
_active_record: Optional[TelemetryRecord] = None


def _ensure_enabled() -> bool:
    """Return True if telemetry is active (env-var gate)."""
    return _TELEMETRY_ENABLED


# ---------------------------------------------------------------------------
# Decorator -- @timed(phase_name)
# ---------------------------------------------------------------------------

def timed(phase: str):
    """Decorator that logs wall-clock time for a named sub-phase.

    When telemetry is disabled (``IRENE_TELEMETRY=0``), this becomes a no-op
    wrapper with zero overhead beyond the function call itself.

    Parameters
    ----------
    phase : str
        Name of the pipeline phase to time (e.g. ``"basis_construction"``,
        ``"matrix_assembly"``, ``"solve"``, ``"post_processing"``).

    Example
    -------
    >>> @timed("basis_construction")
    ... def build_basis(deg):
    ...     return [monomials]
    """
    def decorator(fn: Callable) -> Callable:
        if not _TELEMETRY_ENABLED:
            # Zero-overhead path: just return the original function.
            return fn

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                _record_timing(phase, elapsed)

        # Preserve original function metadata.
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


def _record_timing(phase: str, wall_s: float):
    """Internal: record a timing entry on the active telemetry record."""
    global _active_record
    if _active_record is None or _active_record.phase != phase:
        # Create a fresh record for this phase and push to session.
        new_rec = TelemetryRecord(phase=phase)
        new_rec.timings[phase] = PhaseTiming(wall_clock_s=wall_s)
        _session_records.append(new_rec)
    else:
        if phase not in _active_record.timings:
            _active_record.timings[phase] = PhaseTiming()
        _active_record.timings[phase].wall_clock_s += wall_s


# ---------------------------------------------------------------------------
# Context manager -- TelemetryContext
# ---------------------------------------------------------------------------

class TelemetryContext:
    """Context manager that collects structured metrics for one SDP run.

    Parameters
    ----------
    phase : str
        Top-level phase name (e.g. ``"sdp_solve"``, ``"init_sdp_serial"``).
    **kwargs
        Arbitrary metadata key-value pairs attached to this record.

    Example
    -------
    >>> with TelemetryContext("solve", monomial_count=42, block_dims=[10, 5]):
    ...     sdp.solve()
    """

    def __init__(self, phase: str, **kwargs):
        self.phase = phase
        self._metadata = dict(kwargs)
        self._prev_record: Optional[TelemetryRecord] = None

    def __enter__(self) -> TelemetryContext:
        if not _TELEMETRY_ENABLED:
            return self

        global _active_record
        self._prev_record = _active_record
        _active_record = TelemetryRecord(phase=self.phase, metadata=dict(self._metadata))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _TELEMETRY_ENABLED:
            return False

        global _active_record
        if _active_record is not None:
            _session_records.append(_active_record)
            _active_record = self._prev_record
        return False  # do not suppress exceptions

    def set(self, key: str, value: Any):
        """Attach or update a metadata field on the active record."""
        if not _TELEMETRY_ENABLED:
            return
        global _active_record
        if _active_record is not None:
            _active_record.metadata[key] = value


# ---------------------------------------------------------------------------
# Session-level accessors
# ---------------------------------------------------------------------------

def get_telemetry() -> list[dict]:
    """Return all telemetry records as a list of plain dicts.

    Each dict has keys ``phase``, optionally ``timings`` and ``metadata``.
    """
    if not _TELEMETRY_ENABLED:
        return []
    return [rec.to_dict() for rec in _session_records]


def get_active_record() -> Optional[dict]:
    """Return the currently active telemetry record (if any) as a dict."""
    if not _TELEMETRY_ENABLED or _active_record is None:
        return None
    return _active_record.to_dict()


def clear_telemetry():
    """Reset all session-level telemetry state."""
    global _session_records, _active_record
    _session_records.clear()
    _active_record = None


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(path: str) -> str:
    """Write the full session telemetry to a JSON file.

    Parameters
    ----------
    path : str
        File path for the output JSON.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    import os.path as osp

    records = get_telemetry()
    payload = {
        "irene_telemetry": True,
        "enabled": _TELEMETRY_ENABLED,
        "record_count": len(records),
        "records": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return osp.abspath(path)
