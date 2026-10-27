# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``Sync`` library node and its CUDA and HIP expansions."""

import pytest

import dace
from dace.libraries.standard.nodes.sync import GPU_STREAM, STREAM_CONNECTOR, Sync, SyncScope
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


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_device_barrier_waits_for_the_device_on_the_host(backend: str):
    tasklet = expand(backend, SyncScope.DEVICE, None)
    assert tasklet.side_effects
    assert tasklet.code.as_string == f"DACE_GPU_CHECK({backend}DeviceSynchronize());"


def stream_barrier(name: str, connect: bool) -> tuple[dace.SDFG, dace.SDFGState, Sync]:
    sdfg = dace.SDFG(name)
    sdfg.add_scalar("stream", GPU_STREAM, transient=True)
    state = sdfg.add_state()
    sync = Sync("barrier", scope=SyncScope.STREAM)
    state.add_node(sync)
    if connect:
        sync.add_in_connector(STREAM_CONNECTOR, GPU_STREAM)
        state.add_edge(state.add_read("stream"), None, sync, STREAM_CONNECTOR, dace.Memlet("stream[0]"))
    return sdfg, state, sync


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_stream_barrier_waits_for_its_connected_stream_on_the_host(backend: str):
    _, state, sync = stream_barrier(f"sync_stream_{backend}", connect=True)
    sync.expand(state, backend.upper())
    (tasklet,) = [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert tasklet.side_effects
    assert tasklet.code.as_string == f"DACE_GPU_CHECK({backend}StreamSynchronize({STREAM_CONNECTOR}));"
    (edge,) = state.in_edges(tasklet)
    assert edge.dst_conn == STREAM_CONNECTOR and edge.data.data == "stream"
    assert tasklet.in_connectors[STREAM_CONNECTOR] == GPU_STREAM


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_stream_barrier_without_a_stream_waits_for_the_default_stream(backend: str):
    sdfg, state, sync = stream_barrier(f"sync_default_stream_{backend}", connect=False)
    sdfg.validate()
    sync.expand(state, backend.upper())
    (tasklet,) = state.nodes()
    assert tasklet.code.as_string == f"DACE_GPU_CHECK({backend}StreamSynchronize(nullptr));"
    assert not tasklet.in_connectors


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
        test_a_device_barrier_waits_for_the_device_on_the_host(backend)
        test_a_stream_barrier_waits_for_its_connected_stream_on_the_host(backend)
        test_a_stream_barrier_without_a_stream_waits_for_the_default_stream(backend)
    test_only_a_warp_barrier_takes_a_mask()
