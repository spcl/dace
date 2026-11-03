# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared-memory write detection of :class:`DefaultSharedMemorySync`."""

import dace
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync, is_shared_memory_write


def loopregion_with_shared_write(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("s", [1], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    loop = LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    t = body.add_tasklet("w", {}, {"_o": None}, "_o = 0")
    s = body.add_access("s")
    body.add_edge(t, "_o", s, None, dace.Memlet("s[0]"))
    return sdfg


def test_writes_to_smem_inside_loopregion_detects_write():
    """The in-edges queried are those of the write's own state, not of the enclosing loop region."""
    sdfg = loopregion_with_shared_write("smem_loopregion")
    assert DefaultSharedMemorySync().writes_to_smem_inside_loopregion(sdfg) is True


def test_writes_to_smem_inside_loopregion_absent():
    sdfg = dace.SDFG("no_smem_loop")
    sdfg.add_array("s", [1], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state()
    t = state.add_tasklet("w", {}, {"_o": None}, "_o = 0")
    s = state.add_access("s")
    state.add_edge(t, "_o", s, None, dace.Memlet("s[0]"))
    assert DefaultSharedMemorySync().writes_to_smem_inside_loopregion(sdfg) is False


def test_is_shared_memory_write_predicate():
    sdfg = dace.SDFG("pred")
    sdfg.add_array("s", [1], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    sdfg.add_array("g", [1], dace.float64, dace.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state()
    t = state.add_tasklet("w", {}, {"_o": None}, "_o = 0")
    s = state.add_access("s")
    state.add_edge(t, "_o", s, None, dace.Memlet("s[0]"))
    g = state.add_access("g")  # GPU_Global, no write edge
    s_read = state.add_access("s")  # GPU_Shared, but no incoming edge

    assert is_shared_memory_write(s, state) is True
    assert is_shared_memory_write(g, state) is False  # wrong storage
    assert is_shared_memory_write(s_read, state) is False  # no write edge
    assert is_shared_memory_write(t, state) is False  # not an AccessNode


def test_a_thread_block_map_writing_shared_memory_is_followed_by_a_block_sync():
    sdfg = dace.SDFG("smem_tb_write")
    sdfg.add_array("s", [32], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    sdfg.add_array("g", [32], dace.float64, dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    kentry, kexit = state.add_map("kernel", {"b": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    tentry, texit = state.add_map("block", {"t": "0:32"}, schedule=dace.ScheduleType.GPU_ThreadBlock)
    w = state.add_tasklet("w", {}, {"_o"}, "_o = t")
    s, c = state.add_access("s"), state.add_tasklet("c", {"_i"}, {"_o"}, "_o = _i")
    g = state.add_write("g")
    state.add_nedge(kentry, tentry, dace.Memlet())
    state.add_nedge(tentry, w, dace.Memlet())
    state.add_memlet_path(w, texit, s, src_conn="_o", memlet=dace.Memlet("s[t]"))
    state.add_edge(s, None, c, "_i", dace.Memlet("s[0]"))
    state.add_memlet_path(c, kexit, g, src_conn="_o", memlet=dace.Memlet("g[0]"))
    DefaultSharedMemorySync().apply_pass(sdfg, {})

    syncs = [n for n in state.nodes() if isinstance(n, Sync)]
    assert len(syncs) == 1 and syncs[0].scope == SyncScope.BLOCK
    assert [e.src for e in state.in_edges(syncs[0])] == [texit]
    assert s in state.successors(syncs[0])


if __name__ == "__main__":
    test_writes_to_smem_inside_loopregion_detects_write()
    test_writes_to_smem_inside_loopregion_absent()
    test_is_shared_memory_write_predicate()
    test_a_thread_block_map_writing_shared_memory_is_followed_by_a_block_sync()
