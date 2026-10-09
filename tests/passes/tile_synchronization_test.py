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
def blocked_gemm(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M:BM, 0:N:BN]:
        acc = dace.define_local([BM, BN], dace.float64, storage=dace.StorageType.Register)
        dace.tile.fill(acc, 0.0)
        for k in range(0, K, BK):
            dace.tile.mma(A[i : i + BM, k : k + BK], B[k : k + BK, j : j + BN], acc)
        dace.tile.store(C[i : i + BM, j : j + BN], acc)


@dace.program
def elementwise_chain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N:256]:
        C[i : i + 256] = dace.tile.exp(dace.tile.add(A[i : i + 256], B[i : i + 256]))


def barriers(sdfg: dace.SDFG) -> dict[str, list[str]]:
    """The nodes each barrier comes right before, by label."""
    return {
        f"{node.label}{index}": sorted(edge.dst.label for edge in state.out_edges(node))
        for index, (node, state) in enumerate(
            (node, state) for node, state in sdfg.all_nodes_recursive() if node.label == "tile_sync"
        )
    }


def test_a_gemm_waits_for_its_loads_and_for_the_last_iteration_reads():
    sdfg = blocked_gemm.to_sdfg()
    sdfg.apply_gpu_transformations()
    assert InsertTileSync().apply_pass(sdfg, {}) == 2
    # The loads, before the mma reads them across lanes; and the loads of the next iteration, before they overwrite
    # what the mma of this one reads. The accumulator stays on its threads from the fill to the store.
    assert sorted(barriers(sdfg).values()) == [["copy_A_tile", "copy_B_tile", "mma"], ["mma"]]


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


if __name__ == "__main__":
    test_a_gemm_waits_for_its_loads_and_for_the_last_iteration_reads()
    test_an_elementwise_chain_keeps_every_element_on_its_thread()
    test_the_slots_of_a_circular_buffer_carry_their_own_tokens()
