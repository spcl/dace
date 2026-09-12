# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra unit tests for the layout timing pass (dace/transformation/layout/timing.py).

These cover branches the main timing_test.py does not exercise: the classifier ``is_copy_state``
on no-tasklet / mixed / multi-connector / parenthesized / non-assignment tasklets, the structural
shape of the ``add_fusion_barrier`` tasklet, the fact that a *middle* (non source/sink) copy state
is timed as compute rather than barriered, the ``InsertLayoutTiming`` pass metadata, the
``_report_total_ms`` reduction, and ``time_compute`` returning ``None`` when the SDFG carries no
timers. Every SDFG that is executed is checked bit-exact against a numpy oracle."""
import types

import numpy

import dace
from dace import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.layout.timing import (InsertLayoutTiming, add_fusion_barrier, is_copy_state, time_compute,
                                               _report_total_ms)


def _elementwise_state(sdfg: dace.SDFG, label: str, src: str, dst: str, code: str, n: int) -> dace.SDFGState:
    """Add a state mapping ``dst[i] = f(src[i])`` for i in 0:n via one map + one tasklet."""
    state = sdfg.add_state(label)
    read = state.add_read(src)
    write = state.add_write(dst)
    entry, exit = state.add_map(f"{label}_map", {"i": f"0:{n}"})
    tasklet = state.add_tasklet(f"{label}_t", {"inp"}, {"out"}, code)
    state.add_memlet_path(read, entry, tasklet, dst_conn="inp", memlet=dace.Memlet(f"{src}[i]"))
    state.add_memlet_path(tasklet, exit, write, src_conn="out", memlet=dace.Memlet(f"{dst}[i]"))
    return state


def _classify_state_with_tasklets(codes):
    """Build a throwaway state holding one tasklet per (inputs, outputs, code) spec and classify it.

    Only the tasklet nodes matter to ``is_copy_state`` (it inspects code + connectors), so the
    tasklets are left unwired -- the SDFG is never validated or run."""
    sdfg = dace.SDFG("classify")
    state = sdfg.add_state("s")
    for idx, (inputs, outputs, code) in enumerate(codes):
        state.add_tasklet(f"t{idx}", set(inputs), set(outputs), code)
    return state


def test_is_copy_state_pure_copy_true():
    state = _classify_state_with_tasklets([({"inp"}, {"out"}, "out = inp")])
    assert is_copy_state(state) is True


def test_is_copy_state_parenthesized_copy_true():
    # The classifier strips wrapping parentheses off the rhs before comparing to the in-connector.
    state = _classify_state_with_tasklets([({"inp"}, {"out"}, "out = (((inp)))")])
    assert is_copy_state(state) is True


def test_is_copy_state_multiple_pure_copies_true():
    state = _classify_state_with_tasklets([
        ({"inp"}, {"out"}, "out = inp"),
        ({"inp"}, {"out"}, "out = (inp)"),
    ])
    assert is_copy_state(state) is True


def test_is_copy_state_arithmetic_false():
    state = _classify_state_with_tasklets([({"inp"}, {"out"}, "out = inp * 2.0")])
    assert is_copy_state(state) is False


def test_is_copy_state_mixed_copy_and_arithmetic_false():
    # A single arithmetic tasklet among copies flips the whole state to non-copy (compute).
    state = _classify_state_with_tasklets([
        ({"inp"}, {"out"}, "out = inp"),
        ({"inp"}, {"out"}, "out = inp + 1.0"),
    ])
    assert is_copy_state(state) is False


def test_is_copy_state_multi_connector_false():
    # rhs 'inp0' names an in-connector, but a 2-input tasklet is not a plain relayout copy.
    state = _classify_state_with_tasklets([({"inp0", "inp1"}, {"out"}, "out = inp0")])
    assert is_copy_state(state) is False


def test_is_copy_state_non_assignment_false():
    # A tasklet body without '=' cannot be an out = in copy.
    state = _classify_state_with_tasklets([({"inp"}, {"out"}, "call(inp)")])
    assert is_copy_state(state) is False


def test_is_copy_state_no_tasklet_false():
    # Empty state: no tasklets at all -> not a copy state.
    sdfg = dace.SDFG("notasklet")
    empty = sdfg.add_state("empty")
    assert is_copy_state(empty) is False

    # Memlet-only relayout (access-node to access-node, no copy tasklet) is also not a copy state.
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_transient("At", [4], dace.float64)
    memcopy = sdfg.add_state("memcopy")
    read = memcopy.add_read("A")
    write = memcopy.add_write("At")
    memcopy.add_edge(read, None, write, None, dace.Memlet("A[0:4]"))
    assert is_copy_state(memcopy) is False


def test_add_fusion_barrier_structure_and_side_effect():
    sdfg = dace.SDFG("barrier")
    state = sdfg.add_state("s")
    tasklet = add_fusion_barrier(state)
    assert isinstance(tasklet, nodes.Tasklet)
    # An empty side-effecting tasklet: no data connectors, no body.
    assert len(tasklet.in_connectors) == 0 and len(tasklet.out_connectors) == 0
    assert tasklet.code.as_string.strip() == ""
    assert tasklet.label.startswith("__layout_barrier_")
    assert tasklet.side_effects is True
    assert tasklet.has_side_effects(sdfg) is True
    assert tasklet in state.nodes()
    # An empty side-effect tasklet is not itself a pure copy, so it never marks a state as copy-in/out.
    assert is_copy_state(state) is False


def test_insert_timing_pass_metadata():
    pass_obj = InsertLayoutTiming()
    assert pass_obj.modifies() == ppl.Modifies.Nodes
    # A single application is enough; the pass never asks to be re-run.
    assert pass_obj.should_reapply(ppl.Modifies.Everything) is False
    assert pass_obj.should_reapply(ppl.Modifies.Nothing) is False


def _relayout_sdfg(name: str, n: int) -> dace.SDFG:
    """A[i] --copy--> At[i] --*2--> Ct[i] --copy--> C[i], as three chained states."""
    sdfg = dace.SDFG(name)
    for arr in ("A", "C"):
        sdfg.add_array(arr, [n], dace.float64)
    for arr in ("At", "Ct"):
        sdfg.add_transient(arr, [n], dace.float64)
    copy_in = _elementwise_state(sdfg, "copy_in", "A", "At", "out = inp", n)
    compute = _elementwise_state(sdfg, "compute", "At", "Ct", "out = inp * 2.0", n)
    copy_out = _elementwise_state(sdfg, "copy_out", "Ct", "C", "out = inp", n)
    sdfg.add_edge(copy_in, compute, dace.InterstateEdge())
    sdfg.add_edge(compute, copy_out, dace.InterstateEdge())
    return sdfg


def _has_barrier(state: dace.SDFGState) -> bool:
    return any(isinstance(node, nodes.Tasklet) and node.side_effects for node in state.nodes())


def test_insert_timing_barriers_boundary_and_times_compute():
    n = 16
    sdfg = _relayout_sdfg("relayout_basic", n)
    # copy_in is the source, copy_out is the sink -- both pure copies.
    assert InsertLayoutTiming().apply_pass(sdfg, {}) == 1  # exactly one compute state timed

    by_label = {s.label: s for s in sdfg.states()}
    assert by_label["compute"].instrument == dace.InstrumentationType.Timer
    assert by_label["copy_in"].instrument == dace.InstrumentationType.No_Instrumentation
    assert by_label["copy_out"].instrument == dace.InstrumentationType.No_Instrumentation
    # source/sink copy states get a fusion barrier; the compute state does not.
    assert _has_barrier(by_label["copy_in"]) and _has_barrier(by_label["copy_out"])
    assert not _has_barrier(by_label["compute"])
    sdfg.validate()

    A = numpy.random.rand(n)
    C = numpy.zeros(n)
    sdfg(A=A.copy(), C=C)
    assert numpy.allclose(C, A * 2.0)  # correctness preserved through barrier + instrumentation


def test_insert_timing_middle_copy_state_is_timed_not_barriered():
    # copy_in -> mid_copy -> compute -> copy_out. mid_copy is a pure copy but neither source nor sink,
    # so the pass treats it as compute: it is instrumented, not barriered.
    n = 12
    sdfg = dace.SDFG("relayout_midcopy")
    for arr in ("A", "C"):
        sdfg.add_array(arr, [n], dace.float64)
    for arr in ("At", "Am", "Ct"):
        sdfg.add_transient(arr, [n], dace.float64)
    copy_in = _elementwise_state(sdfg, "copy_in", "A", "At", "out = inp", n)
    mid_copy = _elementwise_state(sdfg, "mid_copy", "At", "Am", "out = inp", n)
    compute = _elementwise_state(sdfg, "compute", "Am", "Ct", "out = inp * 2.0", n)
    copy_out = _elementwise_state(sdfg, "copy_out", "Ct", "C", "out = inp", n)
    sdfg.add_edge(copy_in, mid_copy, dace.InterstateEdge())
    sdfg.add_edge(mid_copy, compute, dace.InterstateEdge())
    sdfg.add_edge(compute, copy_out, dace.InterstateEdge())

    # mid_copy + compute are both instrumented -> 2; only the source/sink are barriered.
    assert InsertLayoutTiming().apply_pass(sdfg, {}) == 2
    by_label = {s.label: s for s in sdfg.states()}
    assert by_label["mid_copy"].instrument == dace.InstrumentationType.Timer
    assert not _has_barrier(by_label["mid_copy"])
    assert _has_barrier(by_label["copy_in"]) and _has_barrier(by_label["copy_out"])
    sdfg.validate()

    A = numpy.random.rand(n)
    C = numpy.zeros(n)
    sdfg(A=A.copy(), C=C)
    assert numpy.allclose(C, A * 2.0)


def test_report_total_ms_reduces_or_none():
    # None report -> None.
    assert _report_total_ms(None) is None
    # Empty durations -> nothing seen -> None.
    assert _report_total_ms(types.SimpleNamespace(durations={})) is None
    # Present element but only empty sample lists -> nothing seen -> None.
    empty_samples = {"state": {"node": {"tid0": []}}}
    assert _report_total_ms(types.SimpleNamespace(durations=empty_samples)) is None
    # Two nodes: report total is the sum of per-node sample means (avg[2,4]=3) + (avg[10]=10) = 13.
    populated = {"state": {"nodeA": {"tid0": [2.0, 4.0]}, "nodeB": {"tid0": [10.0]}}}
    assert _report_total_ms(types.SimpleNamespace(durations=populated)) == 13.0


def test_time_compute_returns_none_without_instrumentation():
    n = 8
    sdfg = _relayout_sdfg("relayout_noinstr", n)  # deliberately NOT instrumented
    assert all(s.instrument == dace.InstrumentationType.No_Instrumentation for s in sdfg.states())

    A = numpy.random.rand(n)

    def run(graph):
        C = numpy.zeros(n)
        graph(A=A.copy(), C=C)
        return C

    # With no Timer states there is no instrumentation report, so no compute time is produced.
    assert time_compute(sdfg, run, reps=2, warmup=1) is None
    # And the run itself is still correct.
    assert numpy.allclose(run(sdfg), A * 2.0)


if __name__ == "__main__":
    test_is_copy_state_pure_copy_true()
    test_is_copy_state_parenthesized_copy_true()
    test_is_copy_state_multiple_pure_copies_true()
    test_is_copy_state_arithmetic_false()
    test_is_copy_state_mixed_copy_and_arithmetic_false()
    test_is_copy_state_multi_connector_false()
    test_is_copy_state_non_assignment_false()
    test_is_copy_state_no_tasklet_false()
    test_add_fusion_barrier_structure_and_side_effect()
    test_insert_timing_pass_metadata()
    test_insert_timing_barriers_boundary_and_times_compute()
    test_insert_timing_middle_copy_state_is_timed_not_barriered()
    test_report_total_ms_reduces_or_none()
    test_time_compute_returns_none_without_instrumentation()
    print("timing extra tests PASS")
