# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared-memory write detection of :class:`DefaultSharedMemorySync`."""

import dace
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


if __name__ == "__main__":
    test_writes_to_smem_inside_loopregion_detects_write()
