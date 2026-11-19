# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``InsertAsyncWaits``: the waits that make memory engine (TMA) loads asynchronous.

Needs no GPU: ``compiler.cuda.cuda_arch`` is pinned to 90 so the memory engine is accepted.
"""

import contextlib

import pytest

import dace
from dace.libraries.standard.nodes.barrier import Barrier
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.memory_engine_wait import BARRIER_INPUT_NAME, TOKEN_INPUT_NAME, MemoryEngineWait
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.insert_async_waits import InsertAsyncWaits
from dace.transformation.passes.multiple_buffering import LowerMultipleBuffering
from tests.passes.multiple_buffering.multiple_buffering_tma_test import tma_tiled_add, tma_tiled_gemm


@contextlib.contextmanager
def hopper():
    with dace.config.set_temporary("compiler", "cuda", "cuda_arch", value="90"):
        yield


def _nodes(sdfg: dace.SDFG, cls) -> list:
    return [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, cls)]


def _edge(state: dace.SDFGState, node, conn: str, incoming: bool = True):
    edges = state.in_edges(node) if incoming else state.out_edges(node)
    (edge,) = [e for e in edges if (e.dst_conn if incoming else e.src_conn) == conn]
    return edge


def test_token_mode():
    """Outside multiple buffering the copy keeps its token, and the wait on it precedes the readers of the tile."""
    sdfg = tma_tiled_add("async_token", 0)
    with hopper():
        assert InsertAsyncWaits().apply_pass(sdfg, {}) == {"token": 1}
    sdfg.validate()
    ((copy_node, state),) = _nodes(sdfg, CopyLibraryNode)
    ((wait, wait_state),) = _nodes(sdfg, MemoryEngineWait)
    assert wait_state is state and wait.parity is None
    token = _edge(state, copy_node, "_tma_token", incoming=False).dst
    assert _edge(state, wait, TOKEN_INPUT_NAME).src is token
    # Both wait on the same barrier
    assert _edge(state, wait, BARRIER_INPUT_NAME).data.data == _edge(state, copy_node, "_tma_bar").data.data
    # The tile is read in the thread-block map, which the wait precedes
    buf = _edge(state, copy_node, "_cpy_out", incoming=False).dst
    for reader in state.successors(buf):
        assert state.edges_between(wait, reader)
    # The leading barrier still keeps the copy from overwriting the tile while it is read
    assert copy_node.sync


def test_copies_of_a_state_issue_before_any_wait():
    sdfg = tma_tiled_gemm("async_token_two", 0)
    with hopper():
        assert InsertAsyncWaits().apply_pass(sdfg, {}) == {"token": 2}
    sdfg.validate()
    copies = _nodes(sdfg, CopyLibraryNode)
    waits = _nodes(sdfg, MemoryEngineWait)
    assert len(copies) == len(waits) == 2
    for wait, state in waits:
        for copy_node, _ in copies:
            assert state.edges_between(copy_node, wait) or state.edges_between(
                _edge(state, copy_node, "_tma_token", incoming=False).dst, wait
            )


def test_synchronous_copy_is_left_alone():
    sdfg = tma_tiled_add("async_sync", 0, asynchronous=False)
    with hopper():
        assert InsertAsyncWaits().apply_pass(sdfg, {}) is None
    assert not _nodes(sdfg, MemoryEngineWait)


def test_store_is_left_alone():
    from tests.library.copy_node_test import _make_memory_engine_store_double

    sdfg = _make_memory_engine_store_double(dace.float64)
    with hopper():
        assert InsertAsyncWaits().apply_pass(sdfg, {}) is None
    assert not _nodes(sdfg, MemoryEngineWait)


def test_idempotent():
    sdfg = tma_tiled_add("async_twice", 0)
    with hopper():
        assert InsertAsyncWaits().apply_pass(sdfg, {}) == {"token": 1}
        assert InsertAsyncWaits().apply_pass(sdfg, {}) is None
    assert len(_nodes(sdfg, MemoryEngineWait)) == 1


@pytest.mark.parametrize("depth", [2, 3])
def test_parity_mode(depth: int):
    """After ``LowerMultipleBuffering``, the staged copies complete on a ring of barriers, one per slot."""
    sdfg = tma_tiled_add(f"async_parity_{depth}", depth)
    with hopper():
        results = ppl.Pipeline([LowerMultipleBuffering(), InsertAsyncWaits()]).apply_pass(sdfg, {})
    sdfg.validate()
    assert results["InsertAsyncWaits"] == {"parity": 2}
    ((lowered,),) = [results["LowerMultipleBuffering"]]
    inner = lowered.main.sdfg
    bar = inner.arrays["tiles_full_bar"]
    assert bar.shape == (depth,) and bar.storage == dace.StorageType.GPU_Shared(dynamic=False)

    # Prologue copy into slot p, main loop copy into slot (n + D-1) % D, both without their own leading barrier
    slots = []
    for copy_node, state in _nodes(sdfg, CopyLibraryNode):
        assert not copy_node.sync and copy_node.asynchronous
        assert not [e for e in state.out_edges(copy_node) if e.src_conn == "_tma_token"]
        slots.append(str(_edge(state, copy_node, "_tma_bar").data.subset))
    assert sorted(slots) == sorted([lowered.prologue.loop_variable, f"Mod(int_floor(t, 32) + {depth - 1}, {depth})"])

    # The main loop and the epilogue wait for the tile they read, before reading it
    waits = _nodes(sdfg, MemoryEngineWait)
    assert len(waits) == 2
    for wait, state in waits:
        assert str(wait.parity) == f"Mod(int_floor(int_floor(t, 32), {depth}), 2)"
        assert str(_edge(state, wait, BARRIER_INPUT_NAME).data.subset) == f"Mod(int_floor(t, 32), {depth})"
        (buf,) = [n for n in state.data_nodes() if n.data == "buf"]
        assert state.edges_between(wait, buf)
        assert state.parent_graph in {region for region, _ in lowered.consume_regions}

    # Barrier initialization before the prologue, and a block barrier and invalidation after the epilogue
    parent = lowered.prologue.parent_graph
    (init,) = parent.predecessors(lowered.prologue)
    (init_tasklet,) = [n for n in init.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert f"mbarrier_init_array(_tma_bar_out, {depth}, 1)" in init_tasklet.code.as_string
    (teardown,) = parent.successors(lowered.epilogue)
    assert any(isinstance(n, Barrier) for n in teardown.nodes())
    assert any(
        isinstance(n, dace.nodes.Tasklet) and "mbarrier_inval_array" in n.code.as_string for n in teardown.nodes()
    )


def test_parity_mode_counts_every_copy_of_a_stage():
    sdfg = tma_tiled_gemm("async_parity_gemm", 2)
    with hopper():
        results = ppl.Pipeline([LowerMultipleBuffering(), InsertAsyncWaits()]).apply_pass(sdfg, {})
    assert results["InsertAsyncWaits"] == {"parity": 4}
    inits = [
        n
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and "mbarrier_init" in n.code.as_string
    ]
    (init,) = inits
    assert "mbarrier_init_array(_tma_bar_out, 2, 2)" in init.code.as_string


def test_without_lowering_results_waits_by_token():
    """Without the results of ``LowerMultipleBuffering`` in the pipeline, the staged copies wait by token."""
    sdfg = tma_tiled_add("async_lowered_alone", 3)
    LowerMultipleBuffering().apply_pass(sdfg, {})
    with hopper():
        assert InsertAsyncWaits().apply_pass(sdfg, {}) == {"token": 2}
    sdfg.validate()


def test_plain_copies_are_not_pipelined_and_do_not_warn():
    """A pipelined loop without memory engine copies stays synchronous, without a warning."""
    import warnings

    sdfg = tma_tiled_add("async_plain", 3, implementation="MappedTasklet")
    with hopper(), warnings.catch_warnings():
        warnings.simplefilter("error")
        results = ppl.Pipeline([LowerMultipleBuffering(), InsertAsyncWaits()]).apply_pass(sdfg, {})
    assert "InsertAsyncWaits" not in results


def test_serialization():
    """The ``asynchronous`` property and the waits survive a JSON round trip."""
    sdfg = tma_tiled_add("async_serialize", 3, asynchronous=False)
    restored = dace.SDFG.from_json(sdfg.to_json())
    ((copy_node, _),) = _nodes(restored, CopyLibraryNode)
    assert copy_node.asynchronous is False

    sdfg = tma_tiled_add("async_serialize_waits", 3)
    with hopper():
        ppl.Pipeline([LowerMultipleBuffering(), InsertAsyncWaits()]).apply_pass(sdfg, {})
    restored = dace.SDFG.from_json(sdfg.to_json())
    restored.validate()
    before = sorted(str(w.parity) for w, _ in _nodes(sdfg, MemoryEngineWait))
    after = sorted(str(w.parity) for w, _ in _nodes(restored, MemoryEngineWait))
    assert before == after and len(after) == 2


if __name__ == "__main__":
    test_token_mode()
    test_copies_of_a_state_issue_before_any_wait()
    test_synchronous_copy_is_left_alone()
    test_store_is_left_alone()
    test_idempotent()
    for depth in (2, 3):
        test_parity_mode(depth)
    test_parity_mode_counts_every_copy_of_a_stage()
    test_without_lowering_results_waits_by_token()
    test_plain_copies_are_not_pipelined_and_do_not_warn()
    test_serialization()
