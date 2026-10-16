from dace.transformation.passes.vectorization.enums import ISA
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the refactored DaCe Python-backend timing instrumentation.

This suite exercises :class:`PythonTimerProvider` end-to-end. After the
refactor the provider:

* always uses ``time.perf_counter_ns() // 1000`` (no C++/ctypes timer);
* emits the timer setup and the ``__dace_perf_events`` list **once** at module
  scope for the top-level SDFG, and resets the list per invocation with
  ``.clear()`` (so repeated calls do not accumulate events);
* supports SDFG-, state-, scope/map-, and node-level (Tasklet) timing for the
  plain Python backend;
* for the cuTile backend, rejects node-level instrumentation inside kernels and
  always emits ``cupy.cuda.get_current_stream().synchronize()`` after every
  ``ct.launch(...)`` (wrapping launch+sync with the scope timer when the map is
  instrumented).

Coverage groups:

* A. Plain Python backend -- runtime (run, compare to NumPy, assert report).
* B. cuTile target -- runtime on a real GPU (GPU-gated).
* C. cuTile target -- codegen/structural (no GPU required).
* D. Provider selection (unit).
"""

import glob
import itertools
import json
import os
import re

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.instrumentation.py_timer import PythonTimerProvider
from dace.dtypes import (
    BackendLanguage,
    InstrumentationType,
    Language,
    ScheduleType,
    StorageType,
)
from dace.memlet import Memlet
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

_SDFG_COUNTER = itertools.count()


def _unique(prefix: str) -> str:
    """Return a process-unique SDFG name for the given prefix."""
    return f"{prefix}_{next(_SDFG_COUNTER)}"


# ---------------------------------------------------------------------------
# Plain Python backend SDFG builders
# ---------------------------------------------------------------------------


def _build_vadd_sdfg(name: str, size, dtype=dace.float64) -> SDFG:
    """Build ``C[i] = A[i] + B[i]`` as a single mapped tasklet.

    :param name: Unique SDFG name.
    :param size: Array size (an int or a symbol expression usable in a range).
    :param dtype: Element data type.
    :returns: The constructed SDFG with ``backend == Python``.
    """
    sdfg = SDFG(name)
    sdfg.backend = BackendLanguage.Python
    sdfg.add_array('A', [size], dtype)
    sdfg.add_array('B', [size], dtype)
    sdfg.add_array('C', [size], dtype)

    state = sdfg.add_state('compute', is_start_block=True)
    state.add_mapped_tasklet(
        'add',
        {'i': f'0:{size}'},
        {'a_in': Memlet('A[i]'), 'b_in': Memlet('B[i]')},
        'c_out = a_in + b_in',
        {'c_out': Memlet('C[i]')},
        external_edges=True,
    )
    return sdfg


def _instrument_sdfg(sdfg: SDFG) -> None:
    """Mark the top-level SDFG for timer instrumentation."""
    sdfg.instrument = InstrumentationType.PythonTimer


def _instrument_states(sdfg: SDFG) -> None:
    """Mark every state for timer instrumentation."""
    for state in sdfg.states():
        state.instrument = InstrumentationType.PythonTimer


def _instrument_maps(sdfg: SDFG) -> None:
    """Mark every map entry for timer instrumentation."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.instrument = InstrumentationType.PythonTimer


def _instrument_tasklets(sdfg: SDFG) -> None:
    """Mark every tasklet for timer instrumentation (node-level timing)."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet):
            node.instrument = InstrumentationType.PythonTimer


def _event_names(report) -> list:
    """Return the list of event names in an instrumentation report."""
    return [ev.name for ev in report.events]


def _assert_all_nonnegative_durations(report) -> None:
    """Assert every event in the report has a non-negative duration."""
    for ev in report.events:
        assert ev.duration >= 0, f"event {ev.name!r} has negative duration {ev.duration}"


def _load_latest_report_json(sdfg: SDFG) -> dict:
    """Load and return the raw Chrome-tracing JSON of the latest report.

    :param sdfg: The SDFG whose ``{build_folder}/perf/report-*.json`` to read.
    :returns: The parsed JSON dictionary.
    """
    path = sdfg.get_latest_report_path()
    assert path is not None, "expected a report file to have been written"
    with open(path, 'r') as fp:
        return json.load(fp)


def _frozen_hash_in_generated_code(sdfg: SDFG) -> str:
    """Return the ``sdfgHash`` string frozen into the generated Python source.

    The Python timer provider computes ``sdfg.hash_sdfg()`` at code-generation
    time (matching the C++ backend) and embeds it verbatim in the report-saving
    code. We recover that frozen value from the generated source so a test can
    compare it against the value written into the runtime report.

    :param sdfg: An SDFG with ``backend == Python`` and Timer instrumentation.
    :returns: The frozen SDFG hash string.
    """
    codes = sdfg.generate_code()
    src = '\n'.join(c.clean_code for c in codes)
    match = re.search(r"['\"]sdfgHash['\"]:\s*['\"](\w+)['\"]", src)
    assert match is not None, "no frozen sdfgHash found in generated Python source"
    return match.group(1)


# ===========================================================================
# A. Plain Python backend -- RUNTIME
# ===========================================================================


class TestPlainPythonRuntime:
    """Run the plain Python backend with each instrumentation granularity,
    verify numerics against NumPy, and assert the report contains the expected
    events with non-negative durations."""

    def test_sdfg_level_timing(self):
        """A1: SDFG-level timing produces a ``SDFG ...`` event."""
        sdfg = _build_vadd_sdfg(_unique('sdfg_level'), 16)
        _instrument_sdfg(sdfg)

        a = np.random.rand(16)
        b = np.random.rand(16)
        c = np.zeros(16)
        sdfg(A=a, B=b, C=c)

        np.testing.assert_allclose(c, a + b)
        report = sdfg.get_latest_report()
        assert any(n.startswith('SDFG ') for n in _event_names(report))
        _assert_all_nonnegative_durations(report)

    def test_state_level_timing(self):
        """A2: State-level timing produces a ``State ...`` event."""
        sdfg = _build_vadd_sdfg(_unique('state_level'), 16)
        _instrument_states(sdfg)

        a = np.random.rand(16)
        b = np.random.rand(16)
        c = np.zeros(16)
        sdfg(A=a, B=b, C=c)

        np.testing.assert_allclose(c, a + b)
        report = sdfg.get_latest_report()
        assert any(n.startswith('State ') for n in _event_names(report))
        _assert_all_nonnegative_durations(report)

    def test_map_scope_timing(self):
        """A3: Map/scope-level timing produces a ``Map ...`` event."""
        sdfg = _build_vadd_sdfg(_unique('map_level'), 16)
        _instrument_maps(sdfg)

        a = np.random.rand(16)
        b = np.random.rand(16)
        c = np.zeros(16)
        sdfg(A=a, B=b, C=c)

        np.testing.assert_allclose(c, a + b)
        report = sdfg.get_latest_report()
        assert any(n.startswith('Map ') for n in _event_names(report))
        _assert_all_nonnegative_durations(report)

    def test_node_level_tasklet_timing(self):
        """A4: Node-level (Tasklet) timing -- the brand-new capability --
        produces ``Tasklet ...`` events."""
        sdfg = _build_vadd_sdfg(_unique('node_level'), 16)
        _instrument_tasklets(sdfg)

        a = np.random.rand(16)
        b = np.random.rand(16)
        c = np.zeros(16)
        sdfg(A=a, B=b, C=c)

        np.testing.assert_allclose(c, a + b)
        report = sdfg.get_latest_report()
        tasklet_events = [n for n in _event_names(report) if n.startswith('Tasklet ')]
        assert tasklet_events, f"expected at least one Tasklet event, got {_event_names(report)}"
        _assert_all_nonnegative_durations(report)

    def test_node_level_event_count_matches_trip_count(self):
        """A4b: node-level (Tasklet) timing has per-iteration semantics. A map
        with a known trip count ``K`` must produce exactly ``K`` ``Tasklet ...``
        events (matching the C++ backend, where the tasklet body and its timing
        live inside the loop). The assertion is exact (``==``)."""
        K = 5
        sdfg = _build_vadd_sdfg(_unique('node_count'), K)
        _instrument_tasklets(sdfg)

        a = np.random.rand(K)
        b = np.random.rand(K)
        c = np.zeros(K)
        sdfg(A=a, B=b, C=c)

        np.testing.assert_allclose(c, a + b)
        report = sdfg.get_latest_report()
        tasklet_events = [n for n in _event_names(report) if n.startswith('Tasklet ')]
        assert len(tasklet_events) == K, (
            f"expected exactly {K} Tasklet events (one per map iteration), "
            f"got {len(tasklet_events)}: {_event_names(report)}")
        _assert_all_nonnegative_durations(report)

    def test_per_invocation_reset_event_count_constant(self):
        """A5: Calling the SAME compiled program three times must not grow the
        report event count (regression for the old module-scope accumulation
        bug). The per-invocation ``.clear()`` keeps the count constant."""
        sdfg = _build_vadd_sdfg(_unique('reset'), 16)
        # Instrument multiple levels so there are several events per call.
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        _instrument_maps(sdfg)

        csdfg = sdfg.compile()

        a = np.random.rand(16)
        b = np.random.rand(16)
        c = np.zeros(16)

        counts = []
        for _ in range(3):
            csdfg(A=a, B=b, C=c)
            np.testing.assert_allclose(c, a + b)
            report = sdfg.get_latest_report()
            counts.append(len(report.events))

        assert counts[0] > 0, "expected at least one event per invocation"
        assert counts[0] == counts[1] == counts[2], (
            f"event count grew across invocations: {counts} (accumulation bug)")

    def test_symbolic_non_divisible_size_runtime(self):
        """A6: Symbolic array size with a non-power-of-two / non-divisible
        runtime extent, run and compared against NumPy with instrumentation."""
        N = dace.symbol('N')
        sdfg = _build_vadd_sdfg(_unique('symbolic'), N)
        _instrument_maps(sdfg)

        n = 13  # Non-power-of-two, deliberately "odd".
        a = np.random.rand(n)
        b = np.random.rand(n)
        c = np.zeros(n)
        sdfg(A=a, B=b, C=c, N=n)

        np.testing.assert_allclose(c, a + b)
        report = sdfg.get_latest_report()
        assert any(n_.startswith('Map ') for n_ in _event_names(report))
        _assert_all_nonnegative_durations(report)

    def test_all_levels_instrumented_runtime(self):
        """All four granularities at once produce SDFG/State/Map/Tasklet
        events in a single run."""
        sdfg = _build_vadd_sdfg(_unique('all_levels'), 20)
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        _instrument_maps(sdfg)
        _instrument_tasklets(sdfg)

        a = np.random.rand(20)
        b = np.random.rand(20)
        c = np.zeros(20)
        sdfg(A=a, B=b, C=c)

        np.testing.assert_allclose(c, a + b)
        report = sdfg.get_latest_report()
        names = _event_names(report)
        assert any(n.startswith('SDFG ') for n in names)
        assert any(n.startswith('State ') for n in names)
        assert any(n.startswith('Map ') for n in names)
        assert any(n.startswith('Tasklet ') for n in names)
        _assert_all_nonnegative_durations(report)


# ===========================================================================
# A'. Plain Python backend -- report JSON / schema, hashing, no-instrumentation
# (ported from the previous test file, adapted to the refactored provider)
# ===========================================================================


class TestPlainPythonReportSchema:
    """Verify the produced report obeys the cross-backend Chrome-tracing
    contract, carries the codegen-frozen SDFG hash, and that an
    uninstrumented program emits neither events nor timer code."""

    def test_report_json_chrome_tracing_schema(self):
        """Run an instrumented program and assert the raw report JSON matches the
        Chrome-tracing schema: a top-level ``traceEvents`` list and ``sdfgHash``
        key, and every duration event has the required fields/values. Ported
        from the old ``test_report_json_format``."""
        sdfg = _build_vadd_sdfg(_unique('report_schema'), 4)
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        _instrument_maps(sdfg)

        a = np.ones(4)
        b = np.ones(4)
        c = np.zeros(4)
        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b)

        data = _load_latest_report_json(sdfg)
        assert 'traceEvents' in data
        assert 'sdfgHash' in data
        assert isinstance(data['traceEvents'], list)
        assert len(data['traceEvents']) > 0

        for event in data['traceEvents']:
            assert 'name' in event
            assert event['cat'] == 'Timer'
            assert event['ph'] == 'X'
            assert isinstance(event['ts'], int)
            assert isinstance(event['dur'], int)
            assert event['dur'] >= 0
            assert 'pid' in event
            assert event['tid'] == -1
            assert 'args' in event and isinstance(event['args'], dict)
            # The uuid triple is carried via the args dict; at minimum a
            # cfg_id must be present for any non-trivial event.
            assert 'cfg_id' in event['args']

    def test_report_has_codegen_frozen_sdfg_hash(self):
        """The report's ``sdfgHash`` must equal the hash frozen at code-generation
        time (``sdfg.hash_sdfg()`` evaluated during codegen, matching the C++
        backend). We recover the frozen value from the generated source and
        compare it to the value written into the runtime report. Ported from the
        old ``test_report_has_correct_sdfg_hash``.

        Note: comparing directly against ``sdfg.hash_sdfg()`` *after* the run is
        unreliable because the SDFG object is mutated during code generation;
        the frozen-at-codegen value is the authoritative one written to disk."""
        sdfg = _build_vadd_sdfg(_unique('report_hash'), 6)
        _instrument_sdfg(sdfg)

        frozen = _frozen_hash_in_generated_code(sdfg)

        a = np.ones(6)
        b = np.ones(6)
        c = np.zeros(6)
        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b)

        data = _load_latest_report_json(sdfg)
        assert isinstance(data['sdfgHash'], str)
        assert len(data['sdfgHash']) > 0
        assert data['sdfgHash'] == frozen, (
            f"report sdfgHash {data['sdfgHash']!r} != codegen-frozen hash {frozen!r}")

    def test_no_instrumentation_no_events(self):
        """An SDFG with NO ``.instrument`` set anywhere produces either no report
        or a report with zero Timer events. Ported from the old
        ``test_no_instrumentation_no_events``."""
        sdfg = _build_vadd_sdfg(_unique('no_instr_events'), 5)
        # Deliberately set no instrumentation at any level.
        try:
            sdfg.clear_instrumentation_reports()
        except FileNotFoundError:
            pass

        a = np.ones(5)
        b = np.ones(5)
        c = np.zeros(5)
        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b)

        path = sdfg.get_latest_report_path()
        if path is not None:
            with open(path) as fp:
                data = json.load(fp)
            assert len(data.get('traceEvents', [])) == 0

    def test_no_instrumentation_no_timer_code(self):
        """Without instrumentation the generated Python source must contain no
        ``__dace_tbegin_`` (or ``__dace_tend_``) timer code. Ported from the old
        ``test_no_instrumentation_no_timer_code``."""
        sdfg = _build_vadd_sdfg(_unique('no_instr_code'), 5)
        codes = sdfg.generate_code()
        source = '\n'.join(c.clean_code for c in codes)

        assert '__dace_tbegin_' not in source
        assert '__dace_tend_' not in source


class TestPlainPythonMultiDim:
    """Multidimensional (non-cuTile) map timing on the plain Python backend."""

    def test_multidimensional_map_timing(self):
        """A 2D elementwise map instrumented at the map level runs correctly and
        yields a ``Map`` Timer event. Ported from the old
        ``test_multidimensional_map``."""
        sdfg = SDFG(_unique('map2d'))
        sdfg.backend = BackendLanguage.Python
        sdfg.add_array('A', [4, 4], dace.float64)
        sdfg.add_array('B', [4, 4], dace.float64)

        state = sdfg.add_state('s0', is_start_block=True)
        a = state.add_access('A')
        b = state.add_access('B')
        me, mx = state.add_map('m2d', {'i': '0:4', 'j': '0:4'})
        t = state.add_tasklet('neg', {'inp'}, {'out'}, 'out = -inp')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet('A[i, j]'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet('B[i, j]'))
        me.map.instrument = InstrumentationType.PythonTimer

        arr_a = np.arange(16, dtype=np.float64).reshape(4, 4)
        arr_b = np.zeros((4, 4), dtype=np.float64)
        sdfg(A=arr_a, B=arr_b)

        np.testing.assert_allclose(arr_b, -arr_a)
        report = sdfg.get_latest_report()
        assert report is not None
        assert any('Map' in ev.name for ev in report.events)
        _assert_all_nonnegative_durations(report)


class TestPlainPythonEventValidity:
    """Structural validity of the report events across an all-levels run."""

    def test_event_durations_nonnegative(self):
        """Every event duration in an all-levels-instrumented run is
        non-negative. Ported from the old ``test_event_duration_nonnegative``."""
        sdfg = _build_vadd_sdfg(_unique('dur_nonneg'), 100)
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        _instrument_maps(sdfg)

        a = np.random.rand(100)
        b = np.random.rand(100)
        c = np.zeros(100)
        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b)

        report = sdfg.get_latest_report()
        assert len(report.events) > 0
        for event in report.events:
            assert event.duration >= 0

    def test_event_uuids_valid(self):
        """Every event's uuid triple is well-formed: a cfg_id is present, and
        the state_id / node_id components are integers (``-1`` when not
        applicable). Ported from the old ``test_event_uuids_valid``."""
        sdfg = _build_vadd_sdfg(_unique('uuids'), 8)
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        _instrument_maps(sdfg)

        a = np.ones(8)
        b = np.ones(8)
        c = np.zeros(8)
        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b)

        report = sdfg.get_latest_report()
        assert len(report.events) > 0
        for event in report.events:
            cfg_id, state_id, node_id = event.uuid
            assert cfg_id >= 0, f"event {event.name!r} missing cfg_id"
            assert isinstance(state_id, int)
            assert isinstance(node_id, int)


class TestPlainPythonContextManager:
    """The ``dace.instrument`` context-manager API on the Python backend."""

    def test_context_manager_instrument(self):
        """``with dace.instrument(...)`` annotates and collects a report for a
        Python-backend program run inside the context. Ported from the old
        ``test_context_manager_instrument`` (the API still exists)."""
        sdfg = _build_vadd_sdfg(_unique('ctx_mgr'), 10)

        a = np.ones(10)
        b = np.ones(10) * 2
        c = np.zeros(10)

        with dace.instrument(
                InstrumentationType.PythonTimer,
                filter='*',
                annotate_maps=True,
                annotate_states=True,
                annotate_sdfgs=True,
        ) as profiler:
            sdfg(A=a, B=b, C=c)

        np.testing.assert_allclose(c, a + b)
        assert len(profiler.reports) > 0
        report = profiler.report
        assert len(report.events) > 0
        _assert_all_nonnegative_durations(report)


# ===========================================================================
# B & C. cuTile target shared helpers
# ===========================================================================


def _apply_cutile_pipeline(sdfg: SDFG, widths=(8, )) -> None:
    """Apply the canonicalize-free cuTile lowering used by the integration
    tests: the ``VectorizeCuTile`` orchestrator (vectorize with
    ``target_isa=ISA.CUTILE``, stamp CuTile schedules/storage/implementations,
    expand library nodes, and select the Python backend).

    :param sdfg: SDFG to transform in place.
    :param widths: Per-dim tile widths (must be powers of two).
    """
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})


def _build_cutile_vadd_sdfg(name: str, dtype=dace.float64) -> SDFG:
    """Build a symbolic-sized ``C[i] = A[i] + B[i]`` SDFG (pre-pipeline)."""
    N = dace.symbol("N")
    sdfg = SDFG(name)
    sdfg.add_array("A", (N, ), dtype)
    sdfg.add_array("B", (N, ), dtype)
    sdfg.add_array("C", (N, ), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"_a": Memlet("A[i]"), "_b": Memlet("B[i]")},
        "_c = _a + _b",
        {"_c": Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


def _instrument_cutile_map(sdfg: SDFG) -> int:
    """Instrument every CuTile-scheduled map entry. Returns the count."""
    count = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == ScheduleType.CuTile:
            node.map.instrument = InstrumentationType.PythonTimer
            count += 1
    return count


def _run_cutile(sdfg: SDFG, **kwargs) -> dict:
    """Compile and run a cuTile SDFG on the GPU.

    With the new pipeline, ``VectorizeCuTile`` always includes data copies
    via ``apply_gpu_transformations()``, so the SDFG expects NumPy (host)
    arrays and handles device transfer internally.

    Returns a dict of array-name -> numpy result.
    """
    csdfg = sdfg.compile()
    csdfg(**kwargs)
    return dict(kwargs)


def _cutile_available() -> bool:
    """Return True if cupy, cuda.tile, and an NVIDIA GPU are all present."""
    try:
        import cupy  # noqa: F401
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_requires_cutile = pytest.mark.skipif(
    not _cutile_available(),
    reason="cupy + cuda.tile + an NVIDIA GPU are required for cuTile runtime tests",
)


# ===========================================================================
# B. cuTile target -- RUNTIME on the real GPU
# ===========================================================================


@pytest.mark.gpu
@_requires_cutile
class TestCuTileRuntime:
    """Run real cuTile kernels on the GPU and verify both numerics and that
    the kernel scope is measurable end-to-end via the always-emitted sync."""

    def test_instrumented_kernel_runtime_and_report(self):
        """B1: an instrumented cuTile map runs on the GPU, produces correct
        results, and yields a Map/scope timer event with non-negative
        duration (proving the kernel is measurable with the always-sync)."""
        sdfg = _build_cutile_vadd_sdfg(_unique('cutile_instr'))
        _apply_cutile_pipeline(sdfg, widths=(8, ))
        assert _instrument_cutile_map(sdfg) >= 1, "expected a CuTile map to instrument"

        n = 64
        rng = np.random.default_rng(7)
        a = rng.random(n)
        b = rng.random(n)
        c = np.zeros(n)
        results = _run_cutile(sdfg, A=a, B=b, C=c, N=n)

        np.testing.assert_allclose(results["C"], a + b, rtol=1e-12)
        report = sdfg.get_latest_report()
        assert any(name.startswith('Map ') for name in _event_names(report)), \
            f"expected a Map event for the cuTile kernel, got {_event_names(report)}"
        _assert_all_nonnegative_durations(report)

    def test_uninstrumented_kernel_runtime_correct(self):
        """B2: without instrumentation the kernel still runs correctly --
        the always-emitted ``synchronize()`` must not break execution."""
        sdfg = _build_cutile_vadd_sdfg(_unique('cutile_noinstr'))
        _apply_cutile_pipeline(sdfg, widths=(8, ))
        # Deliberately do NOT instrument.

        n = 64
        rng = np.random.default_rng(11)
        a = rng.random(n)
        b = rng.random(n)
        c = np.zeros(n)
        results = _run_cutile(sdfg, A=a, B=b, C=c, N=n)

        np.testing.assert_allclose(results["C"], a + b, rtol=1e-12)

    def test_instrumented_kernel_non_divisible_size_runtime(self):
        """B3: power-of-two tile width with a non-divisible global size (exercises
        the scalar width-1 remainder path) runs correctly and is measured."""
        sdfg = _build_cutile_vadd_sdfg(_unique('cutile_remainder'))
        _apply_cutile_pipeline(sdfg, widths=(8, ))
        assert _instrument_cutile_map(sdfg) >= 1

        n = 17  # Not divisible by the tile width of 8.
        rng = np.random.default_rng(13)
        a = rng.random(n)
        b = rng.random(n)
        c = np.zeros(n)
        results = _run_cutile(sdfg, A=a, B=b, C=c, N=n)

        np.testing.assert_allclose(results["C"], a + b, rtol=1e-12)
        report = sdfg.get_latest_report()
        assert any(name.startswith('Map ') for name in _event_names(report))
        _assert_all_nonnegative_durations(report)


# ===========================================================================
# C. cuTile target -- codegen / structural (no GPU needed)
# ===========================================================================


def _build_cutile_tile_sdfg(name: str, instrument_map: bool) -> SDFG:
    """Build a minimal, codegen-ready cuTile SDFG with tile transients.

    Structure: ``A -> MapEntry -> _tile_A -> Tasklet -> _tile_B -> MapExit -> B``
    with a CuTile-scheduled map. Optionally instruments the map entry.

    :param name: Unique SDFG name.
    :param instrument_map: If True, set Timer instrumentation on the map.
    :returns: The constructed SDFG with ``backend == Python``.
    """
    sdfg = SDFG(name)
    sdfg.backend = BackendLanguage.Python
    sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_A", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_B", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:32:32"},
                           schedule=ScheduleType.CuTile)
    if instrument_map:
        me.map.instrument = InstrumentationType.PythonTimer

    a = state.add_read("A")
    b = state.add_write("B")
    tile_a = state.add_access("_tile_A")
    tile_b = state.add_access("_tile_B")
    tasklet = state.add_tasklet("compute", {"inp"}, {"out"},
                                "out = inp * 2.0", language=Language.Python)

    state.add_memlet_path(a, me, tile_a, dst_conn=None,
                          memlet=Memlet(data="A", subset="0:32"))
    state.add_edge(tile_a, None, tasklet, "inp",
                   Memlet(data="_tile_A", subset="0:32"))
    state.add_edge(tasklet, "out", tile_b, None,
                   Memlet(data="_tile_B", subset="0:32"))
    state.add_memlet_path(tile_b, mx, b, src_conn=None,
                          memlet=Memlet(data="B", subset="0:32"))
    return sdfg


def _python_backend_code(sdfg: SDFG) -> str:
    """Return the generated Python-backend source (via the CodeObject's
    ``clean_code``) for the given SDFG."""
    codes = sdfg.generate_code()
    for code_obj in codes:
        if code_obj.target.target_name == 'python':
            text = code_obj.clean_code
            if 'ct.launch(' in text or 'cutile' in text:
                return text
    # Fall back to concatenating all python targets.
    return "\n".join(c.clean_code for c in codes if c.target.target_name == 'python')


class TestCuTileCodegen:
    """Structural checks on the generated cuTile code (no GPU required)."""

    def test_instrumented_order_tbegin_launch_sync_tend(self):
        """C1: an instrumented cuTile map emits, in order, ``__dace_tbegin_``,
        ``ct.launch(``, ``synchronize()``, then ``__dace_tend_``."""
        sdfg = _build_cutile_tile_sdfg(_unique('cutile_cg_instr'), instrument_map=True)
        code = _python_backend_code(sdfg)

        markers = ['__dace_tbegin_', 'ct.launch(', 'synchronize()', '__dace_tend_']
        positions = []
        for marker in markers:
            idx = code.find(marker)
            assert idx != -1, f"marker {marker!r} not found in generated code"
            positions.append(idx)
        assert positions == sorted(positions), (
            f"markers out of order: {list(zip(markers, positions))}")

        m_begin = re.search(r'__dace_tbegin_(\w+)', code)
        m_end = re.search(r'__dace_tend_(\w+)', code)
        assert m_begin and m_end and m_begin.group(1) == m_end.group(1), \
            "tbegin/tend ids must match (regression guard: on_scope_exit must receive the exit node, not the entry)"

    def test_uninstrumented_emits_sync_no_timer(self):
        """C2: a non-instrumented cuTile map still emits ``synchronize()`` after
        ``ct.launch(``, and emits no ``__dace_tbegin_``."""
        sdfg = _build_cutile_tile_sdfg(_unique('cutile_cg_noinstr'), instrument_map=False)
        code = _python_backend_code(sdfg)

        launch_idx = code.find('ct.launch(')
        sync_idx = code.find('synchronize()')
        assert launch_idx != -1, "ct.launch( not found"
        assert sync_idx != -1, "synchronize() not found"
        assert sync_idx > launch_idx, "synchronize() must come after ct.launch("
        assert '__dace_tbegin_' not in code, "no timer code expected without instrumentation"

    def test_tasklet_instrumentation_in_cutile_raises(self):
        """C3: a Tasklet with Timer instrumentation inside a cuTile scope makes
        code generation raise RuntimeError."""
        sdfg = _build_cutile_tile_sdfg(_unique('cutile_cg_tasklet'), instrument_map=False)
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.Tasklet):
                node.instrument = InstrumentationType.PythonTimer

        with pytest.raises(RuntimeError, match="Node-level instrumentation is not supported"):
            sdfg.generate_code()


# ===========================================================================
# D. Provider selection (unit)
# ===========================================================================


class TestProviderSelection:
    """Provider-registration unit tests for the two distinct timer types."""

    def test_python_timer_type_maps_to_python_provider(self):
        """D1: the PythonTimer type is registered to PythonTimerProvider."""
        mapping = InstrumentationProvider.get_provider_mapping()
        assert mapping[InstrumentationType.PythonTimer] is PythonTimerProvider

    def test_timer_type_maps_to_cpp_provider(self):
        """D2: the C++ Timer type still maps to the C++ TimerProvider, distinct
        from the Python provider (no collision between the two types)."""
        from dace.codegen.instrumentation.timer import TimerProvider

        mapping = InstrumentationProvider.get_provider_mapping()
        assert mapping[InstrumentationType.Timer] is TimerProvider
        assert mapping[InstrumentationType.Timer] is not PythonTimerProvider


# ===========================================================================
# E. report_each_invocation config (parity with the C++ backend)
# ===========================================================================


class TestReportEachInvocation:
    """The ``instrumentation.report_each_invocation`` config flag must behave
    like the C++ backend: when true (default), reset and save a fresh report on
    every call; when false, accumulate across all invocations and save a single
    report at finalization (from the generated ``__dace_exit_<name>()`` run by
    ``PythonCompiledSDFG.finalize()``, the analog of the C++ ``__dace_exit``)."""

    def test_true_mode_resets_and_saves_per_call_codegen(self):
        """E1: with the flag true (default), the generated code clears the event
        list per invocation and does not register an atexit saver."""
        sdfg = _build_vadd_sdfg(_unique('rei_true_cg'), 16)
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        code = '\n'.join(c.clean_code for c in sdfg.generate_code())
        assert '__dace_perf_events.clear()' in code
        assert '__dace_atexit' not in code

    def test_false_mode_accumulates_and_saves_in_exit_fn_codegen(self):
        """E2: with the flag false, the generated code does NOT clear per call
        and instead writes the report from the ``__dace_exit_<name>()``
        finalizer (not an atexit handler)."""
        name = _unique('rei_false_cg')
        sdfg = _build_vadd_sdfg(name, 16)
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
            code = '\n'.join(c.clean_code for c in sdfg.generate_code())
        assert '__dace_perf_events.clear()' not in code
        assert '__dace_atexit' not in code
        # The report writer lives in the exit function body.
        exit_fn = f'def __dace_exit_{name}('
        assert exit_fn in code
        assert code.index('__dace_perf_report') > code.index(exit_fn)

    def test_true_mode_writes_report_each_call_runtime(self):
        """E3: with the flag true, each invocation writes its own report file and
        the per-call event count stays constant (no accumulation)."""
        sdfg = _build_vadd_sdfg(_unique('rei_true_rt'), 16)
        _instrument_sdfg(sdfg)
        _instrument_states(sdfg)
        csdfg = sdfg.compile()
        perf_dir = os.path.join(os.path.abspath(sdfg.build_folder), 'perf')
        for f in glob.glob(os.path.join(perf_dir, 'report-*.json')):
            os.remove(f)

        a = np.random.rand(16)
        b = np.random.rand(16)
        c = np.zeros(16)
        csdfg(A=a, B=b, C=c)
        csdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b)

        # Each call clears then re-appends, so the live list never grows.
        assert len(csdfg._namespace['__dace_perf_events']) == 2
        # A report is written during the runs (>= 1; successive calls within the
        # same millisecond collide on the report-{ms}.json name, as in C++), in
        # contrast to false mode which writes nothing until finalization.
        assert len(glob.glob(os.path.join(perf_dir, 'report-*.json'))) >= 1

    def test_false_mode_accumulates_single_report_runtime(self):
        """E4: with the flag false, invocations accumulate events and write no
        report until ``finalize()`` runs the exit function; the saved report
        then contains the union of all invocations' events."""
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
            sdfg = _build_vadd_sdfg(_unique('rei_false_rt'), 16)
            _instrument_sdfg(sdfg)
            _instrument_states(sdfg)
            csdfg = sdfg.compile()
            perf_dir = os.path.join(os.path.abspath(sdfg.build_folder), 'perf')
            for f in glob.glob(os.path.join(perf_dir, 'report-*.json')):
                os.remove(f)

            a = np.random.rand(16)
            b = np.random.rand(16)
            c = np.zeros(16)
            csdfg(A=a, B=b, C=c)
            csdfg(A=a, B=b, C=c)
            np.testing.assert_allclose(c, a + b)

            # No report written during the runs (save happens at finalization).
            assert len(glob.glob(os.path.join(perf_dir, 'report-*.json'))) == 0
            # Events accumulated across both calls (2 events/call * 2 calls).
            assert len(csdfg._namespace['__dace_perf_events']) == 4

            # finalize() runs __dace_exit_<name>(), which writes the single
            # report holding the union of all invocations' events.
            csdfg.finalize()
            report_json = _load_latest_report_json(sdfg)
            assert len(report_json['traceEvents']) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
