# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared-memory write detection of :class:`DefaultSharedMemorySync`."""

import dace
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
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


def lanes_publishing_global_memory(force: bool) -> dace.SDFG:
    """One block whose lanes fill a global buffer that the next statement reads whole, as a block lowering does."""
    sdfg = dace.SDFG(f"lanes_publish_{force}")
    sdfg.add_array("buf", [32], dace.float64, dace.StorageType.GPU_Global, transient=True)
    sdfg.add_array("g", [32], dace.float64, dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    kentry, kexit = state.add_map("kernel", {"b": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    tentry, texit = state.add_map("lanes", {"t": "0:32"}, schedule=dace.ScheduleType.GPU_ThreadBlock)
    tentry.map.gpu_force_syncthreads = force
    w = state.add_tasklet("w", {}, {"_o"}, "_o = t")
    buf, c = state.add_access("buf"), state.add_tasklet("c", {"_i"}, {"_o"}, "_o = _i")
    state.add_nedge(kentry, tentry, dace.Memlet())
    state.add_nedge(tentry, w, dace.Memlet())
    state.add_memlet_path(w, texit, buf, src_conn="_o", memlet=dace.Memlet("buf[t]"))
    state.add_edge(buf, None, c, "_i", dace.Memlet("buf[31]"))
    state.add_memlet_path(c, kexit, state.add_write("g"), src_conn="_o", memlet=dace.Memlet("g[0]"))
    return sdfg


def test_a_thread_block_map_that_forces_a_barrier_is_followed_by_a_block_sync():
    """The lane that reads ``buf[31]`` must wait for the lane that wrote it, though ``buf`` is global memory."""
    sdfg = lanes_publishing_global_memory(force=True)
    DefaultSharedMemorySync().apply_pass(sdfg, {})
    state = sdfg.start_state
    syncs = [n for n in state.nodes() if isinstance(n, Sync)]
    assert len(syncs) == 1 and syncs[0].scope == SyncScope.BLOCK
    assert [e.src.map.label for e in state.in_edges(syncs[0])] == ["lanes"]


def test_a_thread_block_map_writing_global_memory_gets_no_barrier_unasked():
    sdfg = lanes_publishing_global_memory(force=False)
    DefaultSharedMemorySync().apply_pass(sdfg, {})
    assert not any(isinstance(n, Sync) for n in sdfg.start_state.nodes())


def _kernel_writing_shared(name: str, inputs: set, outputs: set, out_conn: str) -> tuple[dace.SDFG, dace.SDFGState]:
    """A kernel whose tasklet (with the given connectors) writes shared memory ``s`` read by a later tasklet."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("s", [32], dace.uint64, dace.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("kernel", {"i": "0:4"}, schedule=dace.ScheduleType.GPU_Device)
    writer = state.add_tasklet("writer", inputs, outputs, "", language=dace.Language.CPP)
    state.add_nedge(entry, writer, dace.Memlet())
    s = state.add_access("s")
    state.add_edge(writer, out_conn, s, None, dace.Memlet("s[0:32]"))
    reader = state.add_tasklet("reader", {"_i"}, {}, "", language=dace.Language.CPP)
    state.add_edge(s, None, reader, "_i", dace.Memlet("s[0]"))
    state.add_nedge(reader, exit_, dace.Memlet())
    # The pass only reads the writer's connector names, so its inputs stay unconnected
    return sdfg, state


def _sync_tasklets(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, Sync) and n.scope == SyncScope.BLOCK for n, _ in sdfg.all_nodes_recursive())


def test_plain_collaborative_write_is_synchronized():
    sdfg, _ = _kernel_writing_shared("smem_plain_write", {}, {"_o"}, "_o")
    DefaultSharedMemorySync().apply_pass(sdfg, None)
    assert _sync_tasklets(sdfg) == 1


def test_memory_engine_load_is_not_synchronized():
    """A TMA load (with a ``_tma_bar`` input) completes on its mbarrier: no ``__syncthreads()`` after it."""
    sdfg = _kernel_writing_shared(
        "smem_tma_write", {"_tma_bar"}, {CopyLibraryNode.OUTPUT_CONNECTOR_NAME}, CopyLibraryNode.OUTPUT_CONNECTOR_NAME
    )[0]
    DefaultSharedMemorySync().apply_pass(sdfg, None)
    assert _sync_tasklets(sdfg) == 0


def test_mbarrier_initialization_is_not_synchronized():
    """Initializing or invalidating mbarriers (a ``_tma_bar_out`` output) synchronizes itself."""
    sdfg, _ = _kernel_writing_shared("smem_mbarrier_init", {}, {"_tma_bar_out"}, "_tma_bar_out")
    DefaultSharedMemorySync().apply_pass(sdfg, None)
    assert _sync_tasklets(sdfg) == 0


if __name__ == "__main__":
    test_writes_to_smem_inside_loopregion_detects_write()
    test_writes_to_smem_inside_loopregion_absent()
    test_is_shared_memory_write_predicate()
    test_a_thread_block_map_writing_shared_memory_is_followed_by_a_block_sync()
    test_a_thread_block_map_that_forces_a_barrier_is_followed_by_a_block_sync()
    test_a_thread_block_map_writing_global_memory_gets_no_barrier_unasked()
    test_writes_to_smem_inside_loopregion_absent()
    test_is_shared_memory_write_predicate()
    test_plain_collaborative_write_is_synchronized()
    test_memory_engine_load_is_not_synchronized()
    test_mbarrier_initialization_is_not_synchronized()
