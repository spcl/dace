# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scatter whose slot takes more than one index is guarded on the PAIR, not on each index.

``out[xi[j], yi[j]] = ...`` collides when two iterations pick the same element of ``out``. Asking
``xi`` and ``yi`` each to be a permutation -- what one guard per index array amounts to -- is a
strictly stronger demand, and the shape this exists for fails it: npbench ``mandelbrot2`` scatters
through ``Xiv``/``Yiv``, the flattened mgrid coordinates, where ``Xiv`` holds ``XN`` distinct values
across ``XN*YN`` entries while every ``(Xiv[j], Yiv[j])`` pair is distinct. Per-array guards abort a
correct program there.

So the write is keyed across all of its dimensions -- ``xi[j]*stride0 + yi[j]*stride1``, the flat
slot -- and the existing one-dimensional conflict check runs on that key. A conditional scatter
sends its masked-off iterations past the end of the target instead, one slot each, since an
iteration that writes nothing may not be counted as colliding with one that does.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes import scatter_to_guarded_maps as sgm
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map

M, N, K = (dace.symbol(s, dtype=dace.int64) for s in ('M', 'N', 'K'))


@dace.program
def pair_scatter(out: dace.int64[M, N], xi: dace.int64[K], yi: dace.int64[K], val: dace.int64[K]):
    for j in range(K):
        out[xi[j], yi[j]] = val[j]


@dace.program
def masked_pair_scatter(out: dace.int64[M, N], xi: dace.int64[K], yi: dace.int64[K], val: dace.int64[K],
                        keep: dace.bool[K]):
    for j in range(K):
        if keep[j]:
            out[xi[j], yi[j]] = val[j]


@dace.program
def else_branch_pair_scatter(out: dace.int64[M, N], xi: dace.int64[K], yi: dace.int64[K], val: dace.int64[K],
                             keep: dace.bool[K]):
    # The else branch runs when NO listed condition held, which the keyer does not reconstruct --
    # so this loop must be declined outright rather than keyed on a mask it cannot read.
    for j in range(K):
        if keep[j]:
            out[xi[j], yi[j]] = val[j]
        else:
            out[xi[j], yi[j]] = -val[j]


@dace.program
def row_pair_scatter(out: dace.int64[M, N], xi: dace.int64[2, K], yi: dace.int64[2, K], val: dace.int64[K]):
    # Rank-2 index arrays, read at a fixed row: the key fill has to address them at BOTH of their
    # dimensions.
    for j in range(K):
        out[xi[1, j], yi[1, j]] = val[j]


def loop_labels(sdfg: dace.SDFG):
    return [c.label for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


def has_map(sdfg: dace.SDFG) -> bool:
    return any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


def built(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.simplify()
    return sdfg


def grid_indices(m: int, n: int, seed: int):
    """A permutation of the ``m x n`` slots, split into its two coordinates.

    Both coordinates repeat -- ``xi`` takes ``m`` distinct values over ``m*n`` entries -- while the
    pairs are all distinct. That gap is the whole subject of this file.
    """
    perm = np.random.default_rng(seed).permutation(m * n)
    return (perm // n).astype(np.int64), (perm % n).astype(np.int64)


def test_a_pair_indexed_write_is_keyed_across_both_dimensions():
    """Detection reports ONE joint write naming both indices, and leaves ``idx_arrays`` empty --
    an entry there would mean a per-array guard was still going to be emitted alongside."""
    _loops, idx_arrays, sliced, joint = sgm.detect_scatter_loops_and_idx_arrays(built(pair_scatter))
    assert idx_arrays == set() and sliced == []
    assert [w.dim_exprs for _loop, w in joint] == [('xi[j]', 'yi[j]')]
    assert [w.target for _loop, w in joint] == ['out']
    assert [w.mask_expr for _loop, w in joint] == [None]


def test_a_masked_write_carries_its_mask():
    """The guard needs the mask itself, not just the knowledge that one exists: keying a
    masked-off iteration onto the slot it would have written invents a collision."""
    _loops, _idx, _sliced, joint = sgm.detect_scatter_loops_and_idx_arrays(built(masked_pair_scatter))
    assert [w.mask_expr for _loop, w in joint] == ['(keep[j])']


def test_an_unreadable_mask_declines_the_loop():
    """Declining means the loop stays sequential. The failure this guards against is not "missed
    optimization" but the fallback: per-array guards on a write the keyer gave up on."""
    sdfg = built(else_branch_pair_scatter)
    loops, idx_arrays, sliced, joint = sgm.detect_scatter_loops_and_idx_arrays(sdfg)
    assert (loops, idx_arrays, sliced, joint) == ([], set(), [], [])
    sgm.ScatterToGuardedMaps().apply_pass(sdfg, {})
    assert loop_labels(sdfg) != []


def test_the_pair_scatter_parallelizes():
    sdfg = built(pair_scatter)
    assert sgm.ScatterToGuardedMaps().apply_pass(sdfg, {}) == 1
    assert loop_labels(sdfg) == []
    assert has_map(sdfg)
    sdfg.validate()


def test_repeated_coordinates_are_not_a_collision():
    """Every ``xi`` value recurs ``n`` times and every ``yi`` value ``m`` times; the pairs are a
    permutation. A per-array guard aborts here, so reaching the assertion at all is the result --
    the trap is ``std::abort()`` and would keep the test process with it."""
    m, n = 8, 6
    xi, yi = grid_indices(m, n, seed=0)
    val = np.random.default_rng(1).integers(1, 100, size=m * n).astype(np.int64)
    ref = np.zeros((m, n), dtype=np.int64)
    for j in range(m * n):
        ref[xi[j], yi[j]] = val[j]

    sdfg = built(pair_scatter)
    sgm.ScatterToGuardedMaps().apply_pass(sdfg, {})
    got = np.zeros((m, n), dtype=np.int64)
    sdfg(out=got, xi=xi, yi=yi, val=val, M=m, N=n, K=m * n)
    assert np.array_equal(got, ref)


def test_a_masked_scatter_keeps_its_values():
    """The masked-off half repeats the written half's slots EXACTLY, so the mask is the only thing
    separating them: key a skipped iteration by the slot it would have written and every slot has
    two claimants. Run in trap mode -- if the mask were dropped the guard would ``std::abort()``
    and take the test process with it, which is the observation this test is making."""
    m, n = 4, 4
    px, py = grid_indices(m, n, seed=2)
    xi = np.concatenate([px, px])
    yi = np.concatenate([py, py])
    keep = np.concatenate([np.ones(m * n, np.bool_), np.zeros(m * n, np.bool_)])
    val = np.random.default_rng(3).integers(1, 100, size=2 * m * n).astype(np.int64)
    ref = np.zeros((m, n), dtype=np.int64)
    for j in range(2 * m * n):
        if keep[j]:
            ref[xi[j], yi[j]] = val[j]

    sdfg = built(masked_pair_scatter)
    sgm.ScatterToGuardedMaps().apply_pass(sdfg, {})
    assert loop_labels(sdfg) == []
    got = np.zeros((m, n), dtype=np.int64)
    sdfg(out=got, xi=xi, yi=yi, val=val, keep=keep, M=m, N=n, K=2 * m * n)
    assert np.array_equal(got, ref)


def test_a_repeated_pair_routes_to_the_sequential_branch():
    """The other half of the contract: a genuinely colliding pair must be CAUGHT. Run in dispatch
    mode so the collision is observable in-process -- the trap mode answers the same question by
    aborting, which a test cannot survive."""
    m, n = 4, 4
    xi = np.array([0, 1, 2, 0], dtype=np.int64)
    yi = np.array([0, 1, 2, 0], dtype=np.int64)  # iterations 0 and 3 write the same slot
    val = np.array([5, 6, 7, 8], dtype=np.int64)
    ref = np.zeros((m, n), dtype=np.int64)
    for j in range(4):
        ref[xi[j], yi[j]] = val[j]

    sdfg = built(pair_scatter)
    sgm.ScatterToGuardedMaps(emit_unparallelized_else_branch=True).apply_pass(sdfg, {})
    # Structural first: four iterations over four threads can land on the right answer by luck, so
    # the dispatcher's existence is what says the collision was detected rather than survived.
    dispatch = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    assert len(dispatch) == 1, 'the guard must emit exactly one parallel-vs-sequential dispatcher'
    assert '_scatter_joint_key' in dispatch[0].branches[0][0].as_string
    got = np.zeros((m, n), dtype=np.int64)
    sdfg(out=got, xi=xi, yi=yi, val=val, M=m, N=n, K=4)
    assert np.array_equal(got, ref), 'the colliding run must fall back to the sequential branch'


def test_an_index_array_is_read_at_its_own_rank():
    """``xi[1, j]`` is one element of a rank-2 array, so the key fill reads it through a rank-2
    memlet. Flattening the subscript into a single expression string instead makes
    ``pystr_to_symbolic`` hand back a LIST for ``"1, j"`` and takes the whole guard down with
    ``'list' object has no attribute 'subs'``."""
    m, n = 3, 4
    xi2, yi2 = grid_indices(m, n, seed=11)
    xi = np.stack([np.zeros_like(xi2), xi2])
    yi = np.stack([np.zeros_like(yi2), yi2])
    val = np.arange(1, m * n + 1, dtype=np.int64)

    sdfg = built(row_pair_scatter)
    assert sgm.ScatterToGuardedMaps().apply_pass(sdfg, {}) == 1
    assert loop_labels(sdfg) == [] and has_map(sdfg)
    sdfg.validate()

    ref = np.zeros((m, n), dtype=np.int64)
    ref[xi[1], yi[1]] = val
    got = np.zeros((m, n), dtype=np.int64)
    sdfg(out=got, xi=xi, yi=yi, val=val, M=m, N=n, K=m * n)
    assert np.array_equal(got, ref)


def test_the_key_fill_map_is_not_a_vectorizer_candidate():
    """The fill is program-level scaffolding: flat, sized by the loop's trip count, and of a rank a
    K-dim tiling has nothing to say about. The guard's other pieces are already invisible to the
    vectorizer -- the check is a library node, the trap a C++ tasklet -- and MarkTileDims raises
    outright on a candidate map with fewer params than K, so this one has to opt out by name."""
    sdfg = built(pair_scatter)
    sgm.ScatterToGuardedMaps().apply_pass(sdfg, {})
    fills = [(n, g) for n, g in sdfg.all_nodes_recursive()
             if isinstance(n, nodes.MapEntry) and n.map.label.startswith('scatter_joint_key')]
    assert len(fills) == 1, f'expected the one key-fill map; got {[n.map.label for n, _ in fills]}'
    entry, state = fills[0]
    assert not is_vectorizable_map(state, entry, 2)


if __name__ == '__main__':
    test_a_pair_indexed_write_is_keyed_across_both_dimensions()
    test_a_masked_write_carries_its_mask()
    test_an_unreadable_mask_declines_the_loop()
    test_the_pair_scatter_parallelizes()
    test_repeated_coordinates_are_not_a_collision()
    test_a_masked_scatter_keeps_its_values()
    test_a_repeated_pair_routes_to_the_sequential_branch()
    test_an_index_array_is_read_at_its_own_rank()
    test_the_key_fill_map_is_not_a_vectorizer_candidate()
