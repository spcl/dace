# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collect the SDFGs of ``tests/schedule_tree/roundtrip_test.py`` for the formal corpus.

Every test function of the roundtrip test (each parametrization separately) is run with ``SDFG.as_schedule_tree``
patched: the first call records a copy of the SDFG it is called on and aborts the test, so nothing is compiled or
converted back.  Tests that never build a tree from a whole SDFG are reported and skipped.

The collected SDFGs are named ``rt.<test>[.<params>]`` and written by ``dump_corpus.py`` like the other corpus
programs (SDFG JSON plus its schedule tree).
"""

import importlib.util
import inspect
import itertools
import os
import sys
from collections.abc import Iterator

import dace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROUNDTRIP_TEST = os.path.join(REPO_ROOT, "tests", "schedule_tree", "roundtrip_test.py")


class _Captured(Exception):
    """Raised by the patched ``as_schedule_tree`` to stop the test after the SDFG was recorded."""

    def __init__(self, sdfg: dace.SDFG) -> None:
        super().__init__("captured")
        self.sdfg = sdfg


def _load_module():
    spec = importlib.util.spec_from_file_location("roundtrip_test", ROUNDTRIP_TEST)
    module = importlib.util.module_from_spec(spec)
    # Library nodes defined in the test are deserialized by importing their class path (``roundtrip_test.<class>``).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _param_combinations(fn) -> Iterator[dict]:
    """All argument combinations of the ``pytest.mark.parametrize`` marks of ``fn``."""
    axes = []
    for mark in getattr(fn, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names, values = mark.args[0], mark.args[1]
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        rows = [v if len(names) > 1 else (v,) for v in values]
        axes.append([dict(zip(names, row)) for row in rows])
    for combo in itertools.product(*axes):
        merged = {}
        for part in combo:
            merged.update(part)
        yield merged


def _suffix(params: dict) -> str:
    parts = []
    for key, value in params.items():
        if isinstance(value, bool):
            parts.append(key if value else f"no_{key}")
        else:
            parts.append(str(value))
    return "".join(f".{p}" for p in parts)


def collect() -> list[tuple[str, dace.SDFG | None, str]]:
    """Return ``(name, sdfg, note)`` per roundtrip test case; ``sdfg`` is ``None`` when nothing was captured.

    The patch is undone before returning, so callers may build schedule trees of the collected SDFGs.
    """
    module = _load_module()
    cases = []
    original = dace.SDFG.as_schedule_tree

    def patched(self, in_place: bool = False):
        raise _Captured(dace.SDFG.from_json(self.to_json()))

    dace.SDFG.as_schedule_tree = patched
    try:
        for fname, fn in inspect.getmembers(module, inspect.isfunction):
            if not fname.startswith("test_") or fn.__module__ != module.__name__:
                continue
            if any(m.name == "skip" for m in getattr(fn, "pytestmark", [])):
                continue
            for params in _param_combinations(fn):
                name = "rt." + fname[len("test_") :] + _suffix(params)
                try:
                    fn(**params)
                except _Captured as captured:
                    cases.append((name, captured.sdfg, ""))
                except Exception as ex:  # The test failed before building a tree: report, do not abort the dump
                    cases.append((name, None, f"{type(ex).__name__}: {ex}"))
                else:
                    cases.append((name, None, "no whole-SDFG schedule tree built"))
    finally:
        dace.SDFG.as_schedule_tree = original
    return cases
