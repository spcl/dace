# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``MemoryEngineWait`` library node, which waits for asynchronous memory engine (TMA) copies."""

import pytest

import dace
from dace.libraries.standard.nodes.memory_engine_wait import BARRIER_INPUT_NAME, TOKEN_INPUT_NAME, MemoryEngineWait


def _kernel_with_wait(parity) -> tuple[dace.SDFG, dace.SDFGState, MemoryEngineWait]:
    """A kernel whose threads wait on ``bar[0]``, by ``parity`` or (if None) by a token."""
    sdfg = dace.SDFG(f"mbarrier_wait_{'token' if parity is None else 'parity'}")
    sdfg.add_array("bar", [2], dace.uint64, dace.StorageType.GPU_Shared, transient=True)
    sdfg.add_scalar("token", dace.uint64, dace.StorageType.Register, transient=True)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("kernel", {"i": "0:4"}, schedule=dace.ScheduleType.GPU_Device)
    wait = MemoryEngineWait("wait", parity=parity)
    state.add_node(wait)
    bar = state.add_access("bar")
    state.add_nedge(entry, bar, dace.Memlet())
    state.add_edge(bar, None, wait, BARRIER_INPUT_NAME, dace.Memlet("bar[1]"))
    if parity is None:
        token = state.add_access("token")
        state.add_nedge(entry, token, dace.Memlet())
        state.add_edge(token, None, wait, TOKEN_INPUT_NAME, dace.Memlet("token[0]"))
    state.add_nedge(wait, exit_, dace.Memlet())
    return sdfg, state, wait


def test_parity_expansion():
    sdfg, _, _ = _kernel_with_wait("i % 2")
    sdfg.expand_library_nodes()
    (tasklet,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert "dace::cuda::mbarrier_wait_parity(_bar, static_cast<uint32_t>((i % 2)));" in tasklet.code.as_string
    assert set(tasklet.in_connectors) == {BARRIER_INPUT_NAME}
    assert tasklet.side_effects


def test_token_expansion():
    sdfg, _, _ = _kernel_with_wait(None)
    sdfg.expand_library_nodes()
    (tasklet,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert tasklet.code.as_string == "dace::cuda::mbarrier_wait_token(_bar, _token);"
    assert set(tasklet.in_connectors) == {BARRIER_INPUT_NAME, TOKEN_INPUT_NAME}


def test_token_wait_without_token_is_rejected():
    sdfg, state, wait = _kernel_with_wait(None)
    for edge in [e for e in state.in_edges(wait) if e.dst_conn == TOKEN_INPUT_NAME]:
        state.remove_memlet_path(edge)
    with pytest.raises(ValueError, match="parity or a token"):
        sdfg.expand_library_nodes()


def test_outside_kernel_is_rejected():
    sdfg = dace.SDFG("mbarrier_wait_host")
    sdfg.add_array("bar", [1], dace.uint64, transient=True)
    state = sdfg.add_state()
    wait = MemoryEngineWait("wait", parity=0)
    state.add_edge(state.add_access("bar"), None, wait, BARRIER_INPUT_NAME, dace.Memlet("bar[0]"))
    with pytest.raises(ValueError, match="inside a GPU kernel"):
        sdfg.expand_library_nodes()


def test_free_symbols_and_serialization():
    sdfg, _, wait = _kernel_with_wait("Mod(int_floor(t, 32), 2)")
    assert wait.free_symbols == {"t"}
    restored = dace.SDFG.from_json(sdfg.to_json())
    (node,) = [n for n, _ in restored.all_nodes_recursive() if isinstance(n, MemoryEngineWait)]
    assert str(node.parity) == str(wait.parity)
    sdfg, _, wait = _kernel_with_wait(None)
    assert wait.free_symbols == set()
    restored = dace.SDFG.from_json(sdfg.to_json())
    (node,) = [n for n, _ in restored.all_nodes_recursive() if isinstance(n, MemoryEngineWait)]
    assert node.parity is None


if __name__ == "__main__":
    test_parity_expansion()
    test_token_expansion()
    test_token_wait_without_token_is_rejected()
    test_outside_kernel_is_rejected()
    test_free_symbols_and_serialization()
