# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Size-gated CPU expansion selection for ``CopyLibraryNode`` / ``FillLibraryNode``, plus the
fork/join cost model the CPU specialization band consumes.

A contiguous CPU transfer expands to the element map (``MappedTasklet`` / ``pure``, parallel
across OpenMP threads at top level) unless its element count is PROVABLY below
``compiler.cpu.parallel_transfer_min_elements``, in which case it keeps a single ``std::memcpy``
/ ``std::memset`` (``MemcpyCPU`` / ``CPU``). A symbolic count is assumed big enough, so it takes
the parallel map. Size is the ONLY thing the expansion decides; whether an enclosing scope
re-enters the transfer is the band's call, via the helpers pinned at the bottom of this file.
These assert the ``Auto`` selection only (no compile).
"""
import functools

import pytest

import dace
from dace.libraries.standard.helper import (cpu_transfer_parallelizes, is_parallel_cpu_transfer_size,
                                            is_reentered_cpu_transfer, is_short_loop)
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.sdfg.state import LoopRegion

N = dace.symbol("N")

TEST_THRESHOLD = 1024
BIG_ELEMS = 1 << 18
SMALL_ELEMS = 100


def pin_threshold(func):
    """Pin ``compiler.cpu.parallel_transfer_min_elements`` to ``TEST_THRESHOLD`` so the
    size-gated selection is deterministic regardless of the schema default or a user override."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        orig = dace.config.Config.get("compiler", "cpu", "parallel_transfer_min_elements")
        try:
            dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=TEST_THRESHOLD)
            return func(*args, **kwargs)
        finally:
            dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=orig)

    return wrapper


def _copy_libnode_sdfg(n):
    sdfg = dace.SDFG(f"copy_{n}")
    sdfg.add_array("src", [n], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("dst", [n], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s")
    ln = CopyLibraryNode(name="cp")
    state.add_edge(state.add_access("src"), None, ln, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet(f"src[0:{n}]"))
    state.add_edge(ln, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None, dace.Memlet(f"dst[0:{n}]"))
    sdfg.validate()
    return sdfg, ln


def _memset_libnode_sdfg(n):
    sdfg = dace.SDFG(f"memset_{n}")
    sdfg.add_array("dst", [n], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s")
    ln = FillLibraryNode(name="ms")
    state.add_edge(ln, FillLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None, dace.Memlet(f"dst[0:{n}]"))
    sdfg.validate()
    return sdfg, ln


@pin_threshold
def test_large_copy_selects_mapped():
    sdfg, ln = _copy_libnode_sdfg(BIG_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MappedTasklet'


@pin_threshold
def test_small_copy_selects_memcpy():
    sdfg, ln = _copy_libnode_sdfg(SMALL_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MemcpyCPU'


@pin_threshold
def test_symbolic_copy_selects_mapped():
    """A symbolic extent cannot be proven small, so it takes the parallel element map. Reading it
    as "small" is what single-threaded every dynamically sized bulk copy."""
    sdfg, ln = _copy_libnode_sdfg(N)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MappedTasklet'


@pin_threshold
def test_symbolic_memset_selects_pure():
    sdfg, ln = _memset_libnode_sdfg(N)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'pure'


@pin_threshold
def test_large_memset_selects_pure():
    sdfg, ln = _memset_libnode_sdfg(BIG_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'pure'


@pin_threshold
def test_small_memset_selects_cpu():
    sdfg, ln = _memset_libnode_sdfg(SMALL_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'CPU'


def test_threshold_config_flips_selection():
    """The selector reads the config live: a static copy at or above the threshold takes the
    parallel element map, one below stays a single memcpy."""
    orig = dace.config.Config.get("compiler", "cpu", "parallel_transfer_min_elements")
    try:
        dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=4096)
        below, ln_below = _copy_libnode_sdfg(2048)
        below.expand_library_nodes(recursive=True)
        assert ln_below.implementation == 'MemcpyCPU'
        at, ln_at = _copy_libnode_sdfg(4096)
        at.expand_library_nodes(recursive=True)
        assert ln_at.implementation == 'MappedTasklet'
    finally:
        dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=orig)


# ---------------------------------------------------------------------------
# The size gate itself
# ---------------------------------------------------------------------------
@pin_threshold
@pytest.mark.parametrize("count,expected", [(SMALL_ELEMS, False), (TEST_THRESHOLD - 1, False), (TEST_THRESHOLD, True),
                                            (BIG_ELEMS, True), (N, True), (2 * N, True), (N * dace.symbol("M"), True)])
def test_size_gate_defaults_to_parallel(count, expected):
    """Only a PROVABLY sub-threshold count is serial. Every symbolic count -- one symbol, a
    multiple, or a product of two -- is assumed big enough, with no symbol outranking another."""
    assert is_parallel_cpu_transfer_size(count) is expected


# ---------------------------------------------------------------------------
# Fork/join cost model: owned here, consumed by the CPU specialization band
# ---------------------------------------------------------------------------
def _loop_nested_copy(name: str, trip: str):
    """``for k in range(trip): dst[0:BIG_ELEMS, k] = src[:]`` with the copy as a libnode."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("T", dace.int32)
    sdfg.add_array("src", [BIG_ELEMS], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("dst", [BIG_ELEMS, 8], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    loop = LoopRegion("lp", f"k < {trip}", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("body", is_start_block=True)
    ln = CopyLibraryNode(name="cp")
    state.add_edge(state.add_access("src"), None, ln, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.Memlet(f"src[0:{BIG_ELEMS}]"))
    state.add_edge(ln, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None,
                   dace.Memlet(f"dst[0:{BIG_ELEMS}, k]"))
    sdfg.validate()
    return sdfg, state, ln, loop


def test_is_short_loop():
    """A provably short ascending loop is short; a symbolic bound is not (unknown is not small)."""
    _, _, _, short = _loop_nested_copy("cost_short", "4")
    _, _, _, long_loop = _loop_nested_copy("cost_long", "T")
    assert is_short_loop(short) is True
    assert is_short_loop(long_loop) is False


def test_reentry_and_combined_verdict():
    """A long loop re-enters the transfer, a provably short one does not, and a top-level one has
    nothing above it. ``cpu_transfer_parallelizes`` is the size gate AND the re-entry verdict."""
    _, long_state, long_ln, _ = _loop_nested_copy("cost_reentry_long", "T")
    _, short_state, short_ln, _ = _loop_nested_copy("cost_reentry_short", "4")
    top_sdfg, top_ln = _copy_libnode_sdfg(BIG_ELEMS)
    top_state = top_sdfg.states()[0]

    assert is_reentered_cpu_transfer(long_ln, long_state) is True
    assert is_reentered_cpu_transfer(short_ln, short_state) is False
    assert is_reentered_cpu_transfer(top_ln, top_state) is False

    assert cpu_transfer_parallelizes(long_ln, long_state, BIG_ELEMS) is False
    assert cpu_transfer_parallelizes(short_ln, short_state, BIG_ELEMS) is True
    assert cpu_transfer_parallelizes(top_ln, top_state, BIG_ELEMS) is True
    assert cpu_transfer_parallelizes(top_ln, top_state, SMALL_ELEMS) is False


def test_parallel_map_scope_is_reentry():
    """A parallel enclosing map is always re-entry (its own region would nest); the same map
    marked ``Sequential`` is not."""
    sdfg = dace.SDFG("cost_reentry_map")
    sdfg.add_array("src", [BIG_ELEMS], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("dst", [8, BIG_ELEMS], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s", is_start_block=True)
    me, mx = state.add_map("outer", dict(k="0:8"), schedule=dace.dtypes.ScheduleType.CPU_Multicore)
    ln = CopyLibraryNode(name="cp")
    state.add_memlet_path(state.add_access("src"),
                          me,
                          ln,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet(f"src[0:{BIG_ELEMS}]"))
    state.add_memlet_path(ln,
                          mx,
                          state.add_access("dst"),
                          src_conn=CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet(f"dst[k, 0:{BIG_ELEMS}]"))
    sdfg.validate()
    assert is_reentered_cpu_transfer(ln, state) is True
    me.map.schedule = dace.dtypes.ScheduleType.Sequential
    assert is_reentered_cpu_transfer(ln, state) is False


@pin_threshold
def test_expansion_ignores_reentry():
    """The expansion delegates: a copy the cost model calls re-entered still expands to the
    parallel element map. Sequentializing it is the CPU specialization band's job."""
    sdfg, state, ln, _ = _loop_nested_copy("expand_ignores_reentry", "T")
    assert cpu_transfer_parallelizes(ln, state, BIG_ELEMS) is False
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MappedTasklet'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
