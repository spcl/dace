# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fusing two Maps ordered by a happens-before edge must not introduce a WAR/RAW/WAW hazard.

`MapFusionHorizontal` may fuse Maps that are only ordered by empty Memlets, replacing a whole-Map
ordering with a per-iteration one. `analyze_happens_before_fusion` decides that from the boundary
Memlets alone, so an access those do not spell out is a hazard it cannot see -- and the ordering
edge is then dropped with nothing in its place.

Each test runs the UNFUSED SDFG as the oracle, then fuses and runs again. Refusing to fuse is
always acceptable; changing the numbers never is.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation import dataflow as dftrans

from .map_fusion_vertical_test import count_nodes, unique_name

N = 16
RNG = f"1:{N - 2}"


def _writer_map(state, sdfg, label, tag, array, subset, value, name_inner_container=False):
    """A Map writing ``value`` into ``array[subset]``, optionally through a copy-Memlet.

    With ``name_inner_container`` the write leaves the scope as ``buf[0] -> array[subset]``: the
    Memlet's ``data`` is the inner scalar and only ``other_subset`` carries the real destination.
    """
    map_entry, map_exit = state.add_map(label, dict(i=RNG))
    tasklet = state.add_tasklet(f"t{tag}", set(), {"o"}, f"o = {value}")
    state.add_nedge(map_entry, tasklet, dace.Memlet())
    map_exit.add_in_connector(f"IN_{array}")
    map_exit.add_out_connector(f"OUT_{array}")

    if name_inner_container:
        buffer_name = f"buf{tag}"
        sdfg.add_scalar(buffer_name, dace.float64, transient=True)
        buffer_node = state.add_access(buffer_name)
        state.add_edge(tasklet, "o", buffer_node, None, dace.Memlet(data=buffer_name, subset="0"))
        state.add_edge(buffer_node, None, map_exit, f"IN_{array}",
                       dace.Memlet(data=buffer_name, subset="0", other_subset=subset))
    else:
        state.add_edge(tasklet, "o", map_exit, f"IN_{array}", dace.Memlet(data=array, subset=subset))

    written = state.add_access(array)
    state.add_edge(map_exit, f"OUT_{array}", written, None, dace.Memlet(data=array, subset=f"0:{N}"))
    return map_entry, written


def _reader_map(state, sdfg, label, tag, source, read_array, read_subset, out_array, out_subset):
    """A Map copying ``read_array[read_subset]`` into ``out_array[out_subset]``."""
    map_entry, map_exit = state.add_map(label, dict(i=RNG))
    tasklet = state.add_tasklet(f"t{tag}", {"a"}, {"o"}, "o = a")
    map_entry.add_in_connector(f"IN_{read_array}")
    map_entry.add_out_connector(f"OUT_{read_array}")
    state.add_edge(source, None, map_entry, f"IN_{read_array}",
                   dace.Memlet(data=source.data, subset=f"0:{source.desc(sdfg).shape[0]}"))
    state.add_edge(map_entry, f"OUT_{read_array}", tasklet, "a", dace.Memlet(data=read_array, subset=read_subset))

    map_exit.add_in_connector(f"IN_{out_array}")
    map_exit.add_out_connector(f"OUT_{out_array}")
    state.add_edge(tasklet, "o", map_exit, f"IN_{out_array}", dace.Memlet(data=out_array, subset=out_subset))
    written = state.add_access(out_array)
    state.add_edge(map_exit, f"OUT_{out_array}", written, None, dace.Memlet(data=out_array, subset=f"0:{N}"))
    return map_entry, written


def _base(name):
    sdfg = dace.SDFG(unique_name(name))
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    return sdfg, sdfg.add_state("s", is_start_block=True)


def _run(sdfg):
    args = {"A": np.arange(N, dtype=np.float64) + 1.0, "B": np.zeros(N, dtype=np.float64)}
    sdfg(**args)
    return args


def _assert_fusion_preserves_meaning(build, what: str):
    """Fuse or refuse, but never change the numbers."""
    oracle = _run(build())

    fused = build()
    applied = fused.apply_transformations_repeated(dftrans.MapFusionHorizontal, validate_all=True)
    if applied == 0:
        return  # refusing is always sound
    got = _run(fused)
    for name, expected in oracle.items():
        assert np.array_equal(got[name], expected), (f"fusing across the ordering edge changed {name} "
                                                     f"({what}): {got[name]} != {expected}")


def test_copy_memlet_naming_the_inner_container_hides_the_write():
    """A write leaving the scope as ``buf[0] -> A[i-1]`` is still a write to ``A``.

    The boundary scan reads ``edge.data.data``/``edge.data.subset``, which name the inner scalar,
    so the write is booked as ``buf[0]`` and the cross-iteration WAW against ``A[i]`` is invisible.
    """

    def build():
        sdfg, state = _base("copy_memlet_hides_write")
        _, first_out = _writer_map(state, sdfg, "m1", "1", "A", "i - 1", 1.0, name_inner_container=True)
        second_entry, _ = _writer_map(state, sdfg, "m2", "2", "A", "i", 2.0)
        state.add_nedge(first_out, second_entry, dace.Memlet())
        sdfg.validate()
        return sdfg

    _assert_fusion_preserves_meaning(build, "copy-Memlet naming the inner container")


def test_view_of_the_same_array_is_not_a_different_array():
    """A WAR through a View: Map 1 reads ``V[i-1]`` where ``V`` views ``A``, Map 2 writes ``A[i]``.

    Reads and writes are keyed on the data NAME, so ``V`` and ``A`` look unrelated and the hazard
    is never found.
    """

    def build():
        sdfg, state = _base("view_aliases_array")
        sdfg.add_view("V", [N - 1], dace.float64)
        source = state.add_access("A")
        view = state.add_access("V")
        state.add_edge(source, None, view, "views", dace.Memlet(data="A", subset=f"0:{N - 1}"))
        _, first_out = _reader_map(state, sdfg, "m1", "1", view, "V", "i - 1", "B", "i")
        second_entry, _ = _writer_map(state, sdfg, "m2", "2", "A", "i", 99.0)
        state.add_nedge(first_out, second_entry, dace.Memlet())
        sdfg.validate()
        return sdfg

    _assert_fusion_preserves_meaning(build, "View aliasing the written array")


def test_an_unknown_boundary_subset_must_not_erase_the_access_sets():
    """One Memlet without a subset must make the analysis refuse, not forget everything else.

    ``can_be_applied`` deliberately runs before ``try_initialize``, so an uninitialised boundary
    Memlet is normal. Returning empty read/write sets for it makes every hazard in BOTH Maps
    invisible at once -- the analysis fails open.
    """
    from dace.transformation.dataflow import map_fusion_helper as mfhelper

    sdfg, state = _base("unknown_boundary_subset")
    first_entry, first_out = _writer_map(state, sdfg, "m1", "1", "A", "i - 1", 1.0)
    second_entry, _ = _writer_map(state, sdfg, "m2", "2", "A", "i", 2.0)
    state.add_nedge(first_out, second_entry, dace.Memlet())
    sdfg.validate()

    # Blank one boundary subset, exactly as an uninitialised Memlet would present it.
    blanked = next(e for e in state.in_edges(state.exit_node(first_entry)) if not e.data.is_empty())
    blanked.data.subset = None

    reads, writes = mfhelper._scope_boundary_accesses(state, first_entry, None)
    # An empty dict is the fail-open answer: indistinguishable from "this Map touches nothing".
    assert writes != {}, "an unknown subset erased the write set instead of being reported as unknown"

    param_repl = mfhelper.find_parameter_remapping(first_entry.map, second_entry.map)
    assert mfhelper.analyze_happens_before_fusion(state=state,
                                                  sdfg=sdfg,
                                                  first_map_entry=first_entry,
                                                  second_map_entry=second_entry,
                                                  param_repl=param_repl) is None


def _inner_write_then_read_sdfg() -> dace.SDFG:
    """``T`` is written in one state and read back in the next: a cross-state use-def chain."""
    inner = dace.SDFG("inner_write_then_read")
    inner.add_array("T", [1], dace.float64)
    inner.add_array("X", [1], dace.float64)

    first = inner.add_state("s1", is_start_block=True)
    writer = first.add_tasklet("w", set(), {"o"}, "o = 7.0")
    first.add_edge(writer, "o", first.add_access("T"), None, dace.Memlet(data="T", subset="0"))

    second = inner.add_state_after(first, "s2")
    reader = second.add_tasklet("r", {"a"}, {"o"}, "o = a * 2.0")
    second.add_edge(second.add_access("T"), None, reader, "a", dace.Memlet(data="T", subset="0"))
    second.add_edge(reader, "o", second.add_access("X"), None, dace.Memlet(data="X", subset="0"))
    return inner


def _make_inout_split_sdfg() -> dace.SDFG:
    """`T -> map1(nsdfg) -> T -> map2 -> B`, where the NestedSDFG has `T` as an InOut connector."""
    sdfg = dace.SDFG(unique_name("inout_split_cross_state"))
    for name in ("T", "X", "B"):
        sdfg.add_array(name, [N], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)

    first_entry, first_exit = state.add_map("m1", dict(i=f"0:{N}"))
    nsdfg = state.add_nested_sdfg(_inner_write_then_read_sdfg(), {"T"}, {"T", "X"})
    first_entry.add_in_connector("IN_T")
    first_entry.add_out_connector("OUT_T")
    state.add_edge(state.add_access("T"), None, first_entry, "IN_T", dace.Memlet(data="T", subset=f"0:{N}"))
    state.add_edge(first_entry, "OUT_T", nsdfg, "T", dace.Memlet(data="T", subset="i"))
    for name in ("T", "X"):
        first_exit.add_in_connector(f"IN_{name}")
        first_exit.add_out_connector(f"OUT_{name}")
        state.add_edge(nsdfg, name, first_exit, f"IN_{name}", dace.Memlet(data=name, subset="i"))
    intermediate = state.add_access("T")
    state.add_edge(first_exit, "OUT_T", intermediate, None, dace.Memlet(data="T", subset=f"0:{N}"))
    state.add_edge(first_exit, "OUT_X", state.add_access("X"), None, dace.Memlet(data="X", subset=f"0:{N}"))

    second_entry, second_exit = state.add_map("m2", dict(j=f"0:{N}"))
    second_entry.add_in_connector("IN_T")
    second_entry.add_out_connector("OUT_T")
    state.add_edge(intermediate, None, second_entry, "IN_T", dace.Memlet(data="T", subset=f"0:{N}"))
    consumer = state.add_tasklet("cons", {"a"}, {"o"}, "o = a")
    state.add_edge(second_entry, "OUT_T", consumer, "a", dace.Memlet(data="T", subset="j"))
    second_exit.add_in_connector("IN_B")
    second_exit.add_out_connector("OUT_B")
    state.add_edge(consumer, "o", second_exit, "IN_B", dace.Memlet(data="B", subset="j"))
    state.add_edge(second_exit, "OUT_B", state.add_access("B"), None, dace.Memlet(data="B", subset=f"0:{N}"))
    sdfg.validate()
    return sdfg


def test_inout_split_must_not_redirect_a_cross_state_read():
    """Splitting an InOut connector may only rename reads that really come from outside.

    ``_inout_split_is_safe`` classifies each inner AccessNode per STATE, so a `T` written in one
    state and read back in the next looks like a pure write plus a pure read. The split then points
    that read at the fresh input array, and the NestedSDFG silently reads the OUTER value instead
    of the one it just wrote.
    """

    def run(sdfg):
        args = {"T": np.full(N, 1000.0), "X": np.zeros(N), "B": np.zeros(N)}
        sdfg(**args)
        return args

    oracle = run(_make_inout_split_sdfg())
    # The inner SDFG writes 7 and reads it straight back, so X is 14 and never 2*1000.
    assert np.array_equal(oracle["X"], np.full(N, 14.0)), f"oracle itself is wrong: {oracle['X']}"

    fused = _make_inout_split_sdfg()
    if fused.apply_transformations_repeated(dftrans.MapFusionVertical, validate_all=True) == 0:
        return  # refusing is always sound
    got = run(fused)
    for name, expected in oracle.items():
        assert np.array_equal(got[name], expected), (f"the InOut split changed {name}: {got[name]} != {expected}")


def _wcr_shape(reduction_is_consumed: bool) -> dace.SDFG:
    """`map1` reduces into `out` (WCR) and writes `C`; `out *= 2`; `map2` writes `D`.

    With `reduction_is_consumed` the second Map reads the reduced `out`, which is the shape that
    would need a global barrier between the reduction and its consumer. Otherwise it reads `C`,
    `A` and `B` only, so the reduction is independent of it and just rides along.
    """
    sdfg = dace.SDFG(unique_name(f"wcr_consumed_{reduction_is_consumed}"))
    for name in ("A", "B", "D"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64, transient=True)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    reduce_wcr = "lambda old, new: old + new"

    a_read, b_read = state.add_read("A"), state.add_read("B")
    first_entry, first_exit = state.add_map("map1", dict(i=f"0:{N}"))
    first = state.add_tasklet("body1", {"a", "b"}, {"wcr_o", "c_o"}, "wcr_o = a * b\nc_o = a + b")
    for name, node in (("A", a_read), ("B", b_read)):
        first_entry.add_in_connector(f"IN_{name}")
        first_entry.add_out_connector(f"OUT_{name}")
        state.add_edge(node, None, first_entry, f"IN_{name}", dace.Memlet(f"{name}[0:{N}]"))
    state.add_edge(first_entry, "OUT_A", first, "a", dace.Memlet("A[i]"))
    state.add_edge(first_entry, "OUT_B", first, "b", dace.Memlet("B[i]"))
    for name in ("out", "C"):
        first_exit.add_in_connector(f"IN_{name}")
        first_exit.add_out_connector(f"OUT_{name}")
    state.add_edge(first, "wcr_o", first_exit, "IN_out", dace.Memlet("out[0]", wcr=reduce_wcr))
    state.add_edge(first, "c_o", first_exit, "IN_C", dace.Memlet("C[i]"))

    reduced = state.add_access("out")
    state.add_edge(first_exit, "OUT_out", reduced, None, dace.Memlet("out[0]", wcr=reduce_wcr))
    c_node = state.add_access("C")
    state.add_edge(first_exit, "OUT_C", c_node, None, dace.Memlet(f"C[0:{N}]"))

    scale = state.add_tasklet("scale", {"iv"}, {"ov"}, "ov = iv * 2.0")
    scaled = state.add_access("out")
    state.add_edge(reduced, None, scale, "iv", dace.Memlet("out[0]"))
    state.add_edge(scale, "ov", scaled, None, dace.Memlet("out[0]"))

    second_entry, second_exit = state.add_map("map2", dict(i=f"0:{N}"))
    if reduction_is_consumed:
        second = state.add_tasklet("body2", {"ov2"}, {"d"}, "d = ov2")
        second_entry.add_in_connector("IN_out")
        second_entry.add_out_connector("OUT_out")
        state.add_edge(scaled, None, second_entry, "IN_out", dace.Memlet("out[0]"))
        state.add_edge(second_entry, "OUT_out", second, "ov2", dace.Memlet("out[0]"))
    else:
        second = state.add_tasklet("body2", {"cv", "av", "bv"}, {"d"}, "d = cv + av + bv")
        for name, node in (("C", c_node), ("A", a_read), ("B", b_read)):
            second_entry.add_in_connector(f"IN_{name}")
            second_entry.add_out_connector(f"OUT_{name}")
            state.add_edge(node, None, second_entry, f"IN_{name}", dace.Memlet(f"{name}[0:{N}]"))
        state.add_edge(second_entry, "OUT_C", second, "cv", dace.Memlet("C[i]"))
        state.add_edge(second_entry, "OUT_A", second, "av", dace.Memlet("A[i]"))
        state.add_edge(second_entry, "OUT_B", second, "bv", dace.Memlet("B[i]"))
    second_exit.add_in_connector("IN_D")
    second_exit.add_out_connector("OUT_D")
    state.add_edge(second, "d", second_exit, "IN_D", dace.Memlet("D[i]"))
    state.add_edge(second_exit, "OUT_D", state.add_write("D"), None, dace.Memlet(f"D[0:{N}]"))
    sdfg.validate()
    return sdfg


def _run_wcr(sdfg) -> dict:
    args = {
        "A": np.arange(1, N + 1, dtype=np.float64),
        "B": np.arange(2, N + 2, dtype=np.float64),
        "D": np.zeros(N, dtype=np.float64),
        "out": np.zeros(1, dtype=np.float64),
    }
    sdfg(**args)
    return args


def test_reduction_consumed_by_the_second_map_is_not_fused():
    """Fusing a reduction into the Map that reads its result would need a global barrier.

    Every iteration of the second Map needs the COMPLETED sum, so per-iteration interleaving is
    wrong and the tree reduction would have to be serialised.
    """
    sdfg = _wcr_shape(reduction_is_consumed=True)
    before = count_nodes(sdfg.start_block, nodes.MapEntry)
    for xform in (dftrans.MapFusionVertical, dftrans.MapFusionHorizontal):
        assert sdfg.apply_transformations_repeated(xform, validate_all=True) == 0
    assert count_nodes(sdfg.start_block, nodes.MapEntry) == before


def test_reduction_independent_of_the_second_map_rides_along():
    """A reduction the second Map never reads must not block fusion, and must survive it."""
    oracle = _run_wcr(_wcr_shape(reduction_is_consumed=False))

    sdfg = _wcr_shape(reduction_is_consumed=False)
    assert sdfg.apply_transformations_repeated(dftrans.MapFusionVertical, validate_all=True) == 1
    assert count_nodes(sdfg.start_block, nodes.MapEntry) == 1

    exits = count_nodes(sdfg.start_block, nodes.MapExit, return_nodes=True)
    wcrs = [e.data.wcr for x in exits for e in sdfg.start_block.out_edges(x) if e.data.wcr is not None]
    assert wcrs, "fusion dropped the reduction's WCR"

    got = _run_wcr(sdfg)
    for name in ("D", "out"):
        assert np.array_equal(got[name], oracle[name]), f"fusion changed {name}: {got[name]} != {oracle[name]}"


if __name__ == "__main__":
    pytest.main([__file__])
