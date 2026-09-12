# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU-only regression tests for :class:`DefaultSharedMemorySync` shared-memory write detection."""
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync, is_shared_memory_write

import pytest


def _loopregion_with_shared_write(name: str) -> dace.SDFG:
    """SDFG with a single GPU_Shared write inside a LoopRegion body."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("s", [1], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    loop = LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    t = body.add_tasklet("w", {}, {"_o"}, "_o = 0")
    s = body.add_access("s")
    body.add_edge(t, "_o", s, None, dace.Memlet("s[0]"))
    return sdfg


def test_writes_to_smem_inside_loopregion_detects_write():
    """A GPU_Shared write inside a LoopRegion is detected without raising.

    Regression: ``writes_to_smem_inside_loopregion`` iterated ``(subnode, parent)`` but queried
    ``in_edges`` on the outer LoopRegion node instead of the AccessNode, raising a KeyError exactly
    in this case -- the race hazard the check exists to flag.
    """
    sdfg = _loopregion_with_shared_write("smem_loopregion")
    assert DefaultSharedMemorySync().writes_to_smem_inside_loopregion(sdfg) is True


def test_writes_to_smem_inside_loopregion_absent():
    """A GPU_Shared write that is NOT inside a LoopRegion returns False (and does not raise)."""
    sdfg = dace.SDFG("no_smem_loop")
    sdfg.add_array("s", [1], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state()
    t = state.add_tasklet("w", {}, {"_o"}, "_o = 0")
    s = state.add_access("s")
    state.add_edge(t, "_o", s, None, dace.Memlet("s[0]"))
    assert DefaultSharedMemorySync().writes_to_smem_inside_loopregion(sdfg) is False


def test_is_shared_memory_write_predicate():
    """``is_shared_memory_write`` is True only for a GPU_Shared AccessNode with a non-empty write edge."""
    sdfg = dace.SDFG("pred")
    sdfg.add_array("s", [1], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    sdfg.add_array("g", [1], dace.float64, dace.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state()
    t = state.add_tasklet("w", {}, {"_o"}, "_o = 0")
    s = state.add_access("s")
    state.add_edge(t, "_o", s, None, dace.Memlet("s[0]"))
    g = state.add_access("g")  # GPU_Global, no write edge
    s_read = state.add_access("s")  # GPU_Shared, but no incoming edge

    assert is_shared_memory_write(s, state) is True
    assert is_shared_memory_write(g, state) is False  # wrong storage
    assert is_shared_memory_write(s_read, state) is False  # no write edge
    assert is_shared_memory_write(t, state) is False  # not an AccessNode


if __name__ == "__main__":
    pytest.main([__file__])
