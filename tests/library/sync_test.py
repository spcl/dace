# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``Sync`` library node and its CUDA and HIP expansions."""

import pytest

import dace
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg.validation import InvalidSDFGNodeError


def expand(backend: str, scope: SyncScope, mask: str | None) -> dace.nodes.Tasklet:
    sdfg = dace.SDFG(f"sync_{backend}_{scope.value}_{mask is not None}")
    state = sdfg.add_state()
    sync = Sync("barrier", scope=scope, mask=mask)
    state.add_node(sync)
    sync.expand(state, backend.upper())
    (tasklet,) = state.nodes()
    return tasklet


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize(
    "scope,mask", [(SyncScope.WARP, None), (SyncScope.WARP, "0xff"), (SyncScope.BLOCK, None), (SyncScope.GRID, None)]
)
def test_a_sync_expands_to_a_barrier_with_side_effects(backend: str, scope: SyncScope, mask: str | None):
    tasklet = expand(backend, scope, mask)
    assert tasklet.side_effects
    code = tasklet.code.as_string
    if scope == SyncScope.BLOCK:
        assert code == "__syncthreads();"
    elif scope == SyncScope.GRID:
        assert code == "cooperative_groups::this_grid().sync();"
    elif backend == "hip":
        assert "__builtin_amdgcn_wave_barrier();" in code
    else:
        assert code == f"__syncwarp({mask or ''});"


def test_only_a_warp_barrier_takes_a_mask():
    sdfg = dace.SDFG("sync_block_mask")
    state = sdfg.add_state()
    state.add_node(Sync("barrier", scope=SyncScope.BLOCK, mask="0xff"))
    with pytest.raises(InvalidSDFGNodeError, match="only a warp barrier takes a mask"):
        sdfg.validate()


if __name__ == "__main__":
    for backend in ("cuda", "hip"):
        test_a_sync_expands_to_a_barrier_with_side_effects(backend, SyncScope.WARP, None)
        test_a_sync_expands_to_a_barrier_with_side_effects(backend, SyncScope.WARP, "0xff")
        test_a_sync_expands_to_a_barrier_with_side_effects(backend, SyncScope.BLOCK, None)
        test_a_sync_expands_to_a_barrier_with_side_effects(backend, SyncScope.GRID, None)
    test_only_a_warp_barrier_takes_a_mask()
