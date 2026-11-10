# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InsertTileSync places a barrier before the first node that reads a tile in another layout than the one that wrote
it, and before a node that overwrites a subset another thread may still read; its tokens are per subset."""

import dace
from dace import subsets, symbolic
from dace.libraries.tileops.expansions import BlockLayout
from dace.transformation.passes.tile_synchronization import InsertTileSync, Token

M, N, K = (dace.symbol(name) for name in "MNK")
BM, BN, BK = 16, 16, 8
ROW = BlockLayout(128, 1, 256, None)


@dace.program
def elementwise_chain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N:256]:
        C[i : i + 256] = dace.tile.exp(dace.tile.add(A[i : i + 256], B[i : i + 256]))


def test_an_elementwise_chain_keeps_every_element_on_its_thread():
    sdfg = elementwise_chain.to_sdfg()
    sdfg.apply_gpu_transformations()
    assert InsertTileSync().apply_pass(sdfg, {}) is None


def test_the_slots_of_a_circular_buffer_carry_their_own_tokens():
    k = symbolic.symbol("k")
    slot = lambda index: subsets.Range([(index, index, 1), (0, 255, 1)])
    current, following = Token("buf", slot(k % 2), ROW), Token("buf", slot((k + 1) % 2), None)
    assert not Token("buf", slot(0), None).waits_for(Token("buf", slot(1), ROW))
    assert following.waits_for(Token("buf", slot((k + 1) % 2), ROW))
    assert not current.waits_for(Token("buf", slot(k % 2), ROW))
    # Across the back edge a token names the previous iteration's subset
    assert Token("buf", slot(k), ROW).shifted("k", 1).subset == slot(k - 1)
    assert Token("buf", slot(k), ROW).shifted("k", None).subset is None


L = dace.symbol("L")


@dace.program
def two_row_sums(A: dace.float64[L, 64], B: dace.float64[L, 64], S: dace.float64[L], T: dace.float64[L]):
    for r in dace.map[0:L]:
        S[r : r + 1] = dace.tile.sum(dace.tile.exp(A[r, :]))
    for r in dace.map[0:L]:
        T[r : r + 1] = dace.tile.sum(dace.tile.exp(B[r, :]))


def test_a_barrier_orders_only_the_nodes_of_its_own_kernel():
    """Two kernels in one state: each barrier connects only to nodes inside the kernel it synchronizes."""
    sdfg = two_row_sums.to_sdfg()
    sdfg.apply_gpu_transformations()
    sync = InsertTileSync()
    assert sync.apply_pass(sdfg, {}) == 4
    for state, barrier in sync.inserted:
        kernel = state.entry_node(barrier)
        neighbours = [e.src for e in state.in_edges(barrier)] + [e.dst for e in state.out_edges(barrier)]
        assert all(n is kernel or state.entry_node(n) is kernel for n in neighbours)
    sdfg.validate()


if __name__ == "__main__":
    test_an_elementwise_chain_keeps_every_element_on_its_thread()
    test_the_slots_of_a_circular_buffer_carry_their_own_tokens()
    test_a_barrier_orders_only_the_nodes_of_its_own_kernel()
