# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the LoopRegion lift and the CPU expansion selection of
``CopyLibraryNode`` / ``MemsetLibraryNode``.

Covers:
* ``AssignmentAndCopyKernelToMemsetAndMemcpy`` lifting a single-statement contiguous copy /
  zero ``LoopRegion`` (``for i: dst[i] = src[i]`` / ``for i: dst[i] = 0``) to a library node.
* The auto path selecting the element-map expansion (``MappedTasklet`` / ``pure``) for a
  contiguous CPU transfer whose size is a compile-time constant ``>=`` the ~1 MiB byte
  threshold, versus a single serial ``std::memcpy`` / ``std::memset`` (``MemcpyCPU`` / ``CPU``)
  for a small or symbolic (unknown-at-compile-time) size. The map is a plain element map that
  DaCe schedules across OpenMP threads at top level, so the generated code carries a
  ``#pragma omp parallel for`` only in the statically-large case.
"""
import functools

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import (
    AssignmentAndCopyKernelToMemsetAndMemcpy)
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
    CleanAccessNodeToScalarSliceToTaskletPattern)

N = dace.symbol("N")

# The tests pin the element threshold to TEST_THRESHOLD below, so BIG is clearly above it
# (-> parallel element map) and SMALL clearly below (-> single std::memcpy / std::memset).
TEST_THRESHOLD = 1024
BIG_ELEMS = 1 << 18
SMALL_ELEMS = 100


def _count(sdfg: dace.SDFG, cls) -> int:
    return sum(isinstance(n, cls) for n, _ in sdfg.all_nodes_recursive())


def _has_loop(sdfg: dace.SDFG) -> bool:
    return any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))


def _generated_code(sdfg: dace.SDFG) -> str:
    return "\n".join(obj.code for obj in sdfg.generate_code())


def temporarily_disable_autoopt_and_serialization(func):
    """Disable autoopt + serialization and pin ``compiler.cpu.parallel_transfer_min_elements``
    to ``TEST_THRESHOLD`` so the size-gated selection is deterministic regardless of the schema
    default or a user override."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        orig_autoopt = dace.config.Config.get("optimizer", "autooptimize")
        orig_serialization = dace.config.Config.get("testing", "serialization")
        orig_threshold = dace.config.Config.get("compiler", "cpu", "parallel_transfer_min_elements")
        try:
            dace.config.Config.set("optimizer", "autooptimize", value=False)
            dace.config.Config.set("testing", "serialization", value=False)
            dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=TEST_THRESHOLD)
            return func(*args, **kwargs)
        finally:
            dace.config.Config.set("optimizer", "autooptimize", value=orig_autoopt)
            dace.config.Config.set("testing", "serialization", value=orig_serialization)
            dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=orig_threshold)

    return wrapper


@dace.program
def _copy_loop(src: dace.float64[N], dst: dace.float64[N]):
    for i in range(N):
        dst[i] = src[i]


@dace.program
def _zero_loop(dst: dace.float64[N]):
    for i in range(N):
        dst[i] = 0.0


# ---------------------------------------------------------------------------
# LoopRegion lift
# ---------------------------------------------------------------------------
@temporarily_disable_autoopt_and_serialization
def test_copy_loop_lifts_to_copy_libnode():
    """``for i: dst[i] = src[i]`` -> a single ``CopyLibraryNode``, the loop gone, bit-exact."""
    sdfg = _copy_loop.to_sdfg(simplify=True)
    # The pipeline's structural cleanup folds the frontend ``AccessNode -> scalar-slice ->
    # Tasklet`` bridge before the lift runs; mirror that here so the ``_out = _in`` detector matches.
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})

    lifted = AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert lifted == 1
    assert _count(sdfg, CopyLibraryNode) == 1
    assert _count(sdfg, MemsetLibraryNode) == 0
    assert not _has_loop(sdfg)

    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    src = np.arange(50, dtype=np.float64)
    dst = np.zeros(50, dtype=np.float64)
    sdfg(src=src, dst=dst, N=50)
    assert np.array_equal(src, dst)


@temporarily_disable_autoopt_and_serialization
def test_zero_loop_lifts_to_memset_libnode():
    """``for i: dst[i] = 0`` -> a single ``MemsetLibraryNode``, the loop gone, bit-exact."""
    sdfg = _zero_loop.to_sdfg(simplify=True)

    lifted = AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert lifted == 1
    assert _count(sdfg, MemsetLibraryNode) == 1
    assert _count(sdfg, CopyLibraryNode) == 0
    assert not _has_loop(sdfg)

    sdfg.expand_library_nodes(recursive=True)
    sdfg.validate()
    dst = np.ones(50, dtype=np.float64)
    sdfg(dst=dst, N=50)
    assert np.all(dst == 0.0)


@temporarily_disable_autoopt_and_serialization
def test_self_referential_copy_loop_is_not_lifted():
    """``for i: a[i] = a[i - 1]`` is a carried dependence (same array), not a pure copy -- left alone."""

    @dace.program
    def _shift(a: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1]

    sdfg = _shift.to_sdfg(simplify=True)
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    assert _count(sdfg, CopyLibraryNode) == 0


# ---------------------------------------------------------------------------
# Parallel vs serial CPU expansion selection + generated code
# ---------------------------------------------------------------------------
def _copy_libnode_sdfg(n) -> tuple:
    sdfg = dace.SDFG(f"copy_{n}")
    sdfg.add_array("src", [n], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("dst", [n], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s")
    ln = CopyLibraryNode(name="cp")
    state.add_edge(state.add_access("src"), None, ln, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet(f"src[0:{n}]"))
    state.add_edge(ln, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None, dace.Memlet(f"dst[0:{n}]"))
    sdfg.validate()
    return sdfg, ln


def _memset_libnode_sdfg(n) -> tuple:
    sdfg = dace.SDFG(f"memset_{n}")
    sdfg.add_array("dst", [n], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s")
    ln = MemsetLibraryNode(name="ms")
    state.add_edge(ln, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None,
                   dace.Memlet(f"dst[0:{n}]"))
    sdfg.validate()
    return sdfg, ln


@temporarily_disable_autoopt_and_serialization
def test_large_copy_selects_mapped_parallel():
    """A large contiguous copy takes the element-map path (``MappedTasklet``), which DaCe
    schedules across OpenMP threads at top level (``#pragma omp parallel for``)."""
    sdfg, ln = _copy_libnode_sdfg(BIG_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MappedTasklet'
    assert "#pragma omp parallel for" in _generated_code(sdfg)

    src = np.arange(BIG_ELEMS, dtype=np.float64)
    dst = np.zeros(BIG_ELEMS, dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(src, dst)


@temporarily_disable_autoopt_and_serialization
def test_small_copy_selects_serial_no_pragma():
    sdfg, ln = _copy_libnode_sdfg(SMALL_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MemcpyCPU'
    code = _generated_code(sdfg)
    assert "#pragma omp parallel for" not in code
    assert "memcpy" in code

    src = np.arange(SMALL_ELEMS, dtype=np.float64)
    dst = np.zeros(SMALL_ELEMS, dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(src, dst)


@temporarily_disable_autoopt_and_serialization
def test_large_memset_selects_mapped_parallel():
    """A large contiguous zero takes the element-map path (``pure``), which DaCe schedules
    across OpenMP threads at top level (``#pragma omp parallel for``)."""
    sdfg, ln = _memset_libnode_sdfg(BIG_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'pure'
    assert "#pragma omp parallel for" in _generated_code(sdfg)

    dst = np.ones(BIG_ELEMS, dtype=np.float64)
    sdfg(dst=dst)
    assert np.all(dst == 0.0)


@temporarily_disable_autoopt_and_serialization
def test_small_memset_selects_serial_no_pragma():
    sdfg, ln = _memset_libnode_sdfg(SMALL_ELEMS)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'CPU'
    code = _generated_code(sdfg)
    assert "#pragma omp parallel for" not in code
    assert "memset" in code

    dst = np.ones(SMALL_ELEMS, dtype=np.float64)
    sdfg(dst=dst)
    assert np.all(dst == 0.0)


@temporarily_disable_autoopt_and_serialization
def test_symbolic_copy_selects_serial_memcpy():
    """A symbolic-size copy has an unknown compile-time byte count, so the selector keeps a
    single ``std::memcpy`` (``MemcpyCPU``) rather than fork an OpenMP region for a size that may
    be tiny at runtime. Only a statically-large size takes the parallel element map."""
    sdfg = dace.SDFG("copy_sym")
    sdfg.add_array("src", [N], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("dst", [N], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s")
    ln = CopyLibraryNode(name="cp")
    state.add_edge(state.add_access("src"), None, ln, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("src[0:N]"))
    state.add_edge(ln, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None, dace.Memlet("dst[0:N]"))
    sdfg.validate()
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MemcpyCPU'
    code = _generated_code(sdfg)
    assert "#pragma omp parallel for" not in code
    assert "memcpy" in code

    src = np.arange(5000, dtype=np.float64)
    dst = np.zeros(5000, dtype=np.float64)
    sdfg(src=src, dst=dst, N=5000)
    assert np.array_equal(src, dst)


@temporarily_disable_autoopt_and_serialization
def test_threshold_config_flips_selection():
    """The selector reads ``compiler.cpu.parallel_transfer_min_elements`` live: a static copy at or
    above it takes the parallel element map, one below stays a single ``std::memcpy``."""
    dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=4096)

    below, ln_below = _copy_libnode_sdfg(2048)
    below.expand_library_nodes(recursive=True)
    assert ln_below.implementation == 'MemcpyCPU'

    at, ln_at = _copy_libnode_sdfg(4096)
    at.expand_library_nodes(recursive=True)
    assert ln_at.implementation == 'MappedTasklet'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
