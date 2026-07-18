# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Size-gated CPU expansion selection for ``CopyLibraryNode`` / ``MemsetLibraryNode``.

A contiguous CPU transfer whose element count is a compile-time constant ``>=``
``compiler.cpu.parallel_transfer_min_elements`` expands to the element map (``MappedTasklet`` /
``pure``, parallel across OpenMP threads at top level); a smaller -- or a symbolic
(unknown-at-compile-time) -- size keeps a single ``std::memcpy`` / ``std::memset``
(``MemcpyCPU`` / ``CPU``). These assert the ``Auto`` selection only (no compile).
"""
import functools

import pytest

import dace
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

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
    ln = MemsetLibraryNode(name="ms")
    state.add_edge(ln, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None,
                   dace.Memlet(f"dst[0:{n}]"))
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
def test_symbolic_copy_selects_memcpy():
    sdfg, ln = _copy_libnode_sdfg(N)
    sdfg.expand_library_nodes(recursive=True)
    assert ln.implementation == 'MemcpyCPU'


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
