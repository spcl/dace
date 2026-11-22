# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared-memory write detection of :class:`DefaultSharedMemorySync`."""

import dace
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync


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


def _block_syncs(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, Sync) and n.scope == SyncScope.BLOCK for n, _ in sdfg.all_nodes_recursive())


def test_plain_collaborative_write_is_synchronized():
    sdfg, _ = _kernel_writing_shared("smem_plain_write", {}, {"_o"}, "_o")
    DefaultSharedMemorySync().apply_pass(sdfg, None)
    assert _block_syncs(sdfg) == 1


def test_memory_engine_load_is_not_synchronized():
    """A TMA load (with a ``_tma_bar`` input) completes on its mbarrier: no ``__syncthreads()`` after it."""
    sdfg, _ = _kernel_writing_shared("smem_tma_write", {"_tma_bar"}, {"_cpy_out"}, "_cpy_out")
    DefaultSharedMemorySync().apply_pass(sdfg, None)
    assert _block_syncs(sdfg) == 0


def test_mbarrier_initialization_is_not_synchronized():
    """Initializing or invalidating mbarriers (a ``_tma_bar_out`` output) synchronizes itself."""
    sdfg, _ = _kernel_writing_shared("smem_mbarrier_init", {}, {"_tma_bar_out"}, "_tma_bar_out")
    DefaultSharedMemorySync().apply_pass(sdfg, None)
    assert _block_syncs(sdfg) == 0


if __name__ == "__main__":
    test_writes_to_smem_inside_loopregion_detects_write()
    test_plain_collaborative_write_is_synchronized()
    test_memory_engine_load_is_not_synchronized()
    test_mbarrier_initialization_is_not_synchronized()
