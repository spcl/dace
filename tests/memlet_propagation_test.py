# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlets_state
from dace.sdfg.propagation import propagate_memlet, propagate_memlets_sdfg
from dace.symbolic import same_value


def test_conditional():

    @dace.program
    def conditional(in1, out):
        for i in dace.map[0:10]:
            if i >= 1:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    inp = np.random.rand(10)
    outp = np.zeros((10, ))
    conditional(inp, outp)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_conditional_nested():

    @dace.program
    def conditional(in1, out):
        for i in dace.map[0:10]:
            if i >= 1:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    @dace.program
    def nconditional(in1, out):
        conditional(in1, out)

    inp = np.random.rand(10)
    outp = np.zeros((10, ))
    nconditional(inp, outp)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_runtime_conditional():

    @dace.program
    def rconditional(in1, out, mask):
        for i in dace.map[0:10]:
            if mask[i] > 0:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    inp = np.random.rand(10)
    mask = np.ones((10, ))
    mask[0] = 0
    outp = np.zeros((10, ))
    rconditional(inp, outp, mask)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


S = dace.symbol("S")
S1 = dace.symbol("S1")
S2 = dace.symbol("S2")


@dace.program
def tasklet_in_nested_sdfg(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    offset1: dace.int64,
    offset2: dace.int64,
):
    for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
        a[i + offset1, j + offset2] = ((1.5 * b[i + offset1, j + offset2]) + (2.0 * a[i + offset1, j + offset2])) / 3.5


def test_nsdfg_memlet_propagation():
    sdfg = tasklet_in_nested_sdfg.to_sdfg(simplify=False)
    propagate_memlets_sdfg(sdfg)

    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.SDFGState):
            propagate_memlets_state(n.sdfg, n)


def test_nsdfg_memlet_propagation_with_one_sparse_dimension():
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def sparse(A: dace.float32[M, N], ind: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, ind[i, j]] += 1

    sdfg = sparse.to_sdfg(simplify=False)
    propagate_memlets_sdfg(sdfg)

    # Verify all memlet subsets and volumes in the main state of the program, i.e. around the NSDFG.
    map_state = sdfg.states()[1]
    i = dace.symbol('i')
    j = dace.symbol('j')

    outer_in = map_state.edges()[0].data
    if outer_in.volume != M * N:
        raise RuntimeError('Expected a volume of M*N on the outer input memlet')
    if not same_value(outer_in.subset[0], (0, M - 1, 1)) or not same_value(outer_in.subset[1], (0, N - 1, 1)):
        raise RuntimeError('Expected subset of outer in memlet to be [0:M, 0:N], found ' + str(outer_in.subset))

    # Symbols minted here carry the default dtype, which is part of a symbol's identity; the SDFG's
    # map parameters carry the inferred one. Only the values they stand for are being compared.
    inner_in = map_state.edges()[1].data
    if inner_in.volume != 1:
        raise RuntimeError('Expected a volume of 1 on the inner input memlet')
    if not same_value(inner_in.subset[0], (i, i, 1)) or not same_value(inner_in.subset[1], (j, j, 1)):
        raise RuntimeError('Expected subset of inner in memlet to be [i, j], found ' + str(inner_in.subset))

    inner_out = map_state.edges()[2].data
    if inner_out.volume != 1:
        raise RuntimeError('Expected a volume of 1 on the inner output memlet')
    if not same_value(inner_out.subset[0], (0, i, 1)) or not same_value(inner_out.subset[1], (0, N - 1, 1)):
        raise RuntimeError('Expected subset of inner out memlet to be [0:i+1, 0:N], found ' + str(inner_out.subset))

    outer_out = map_state.edges()[3].data
    if outer_out.volume != M * N:
        raise RuntimeError('Expected a volume of M*N on the outer output memlet')
    if not same_value(outer_out.subset[0], (0, M - 1, 1)) or not same_value(outer_out.subset[1], (0, N - 1, 1)):
        raise RuntimeError('Expected subset of outer out memlet to be [0:M, 0:N], found ' + str(outer_out.subset))


def test_strided_write_keeps_the_multiplier():
    """``C[2 * i]`` covers every second element, not the first ``N``.

    A single-element access has ``re - rb + 1 == 1``, which equals the stride of a unit-stride map
    range, and ``2 * i`` at a zero map begin starts where the map range starts. Both halves of the
    ``i:i+stride`` special case in :class:`~dace.sdfg.propagation.AffineSMemlet` therefore hold for
    an access it was never meant to cover, and returning the map range verbatim drops the
    multiplier -- an under-approximated write set, which is unsound.
    """
    N = dace.symbol('N')

    @dace.program
    def strided_write(A: dace.float64[2 * N], C: dace.float64[2 * N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[2 * i]
                c >> C[2 * i]
                c = a

    sdfg = strided_write.to_sdfg(simplify=False)
    propagate_memlets_sdfg(sdfg)

    state = next(s for s in sdfg.states() if any(isinstance(n, dace.sdfg.nodes.MapExit) for n in s.nodes()))
    out = next(e.data for e in state.edges() if isinstance(e.src, dace.sdfg.nodes.MapExit)
               and isinstance(e.dst, dace.sdfg.nodes.AccessNode) and e.dst.data == 'C')

    assert out.subset.ranges == [(0, 2 * N - 2, 2)], out.subset
    assert out.subset.num_elements() == N, out.subset.num_elements()
    # The written elements must be inside the propagated set; the bug put 2*N-2 outside it.
    assert out.subset.covers(dace.subsets.Range([(2 * N - 2, 2 * N - 2, 1)]))


def test_a_supplied_symbol_table_propagates_what_the_derived_one_does():
    """``propagate_memlet`` derives the scope's symbols itself, which walks every descriptor in the
    SDFG. Callers propagating several memlets through ONE scope node hand it the table instead, so
    the two paths have to agree -- a table that differed would silently widen or narrow the outer
    subset of every edge such a caller builds.
    """
    N = dace.symbol('N')

    @dace.program
    def scaled(a: dace.float64[N], out: dace.float64[N]):
        for i in dace.map[0:N]:
            out[i] = a[i] * 2.0

    sdfg = scaled.to_sdfg(simplify=False)
    state = next(s for s in sdfg.states() if any(isinstance(n, dace.nodes.MapEntry) for n in s.nodes()))
    entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    edge = next(e for e in state.out_edges(entry) if not e.data.is_empty())

    derived = propagate_memlet(state, edge.data, entry, True)
    supplied = propagate_memlet(state,
                                edge.data,
                                entry,
                                True,
                                defined_variables=state.symbols_defined_at(entry).keys()
                                | sdfg.constants.keys())
    assert derived.subset == supplied.subset, (derived, supplied)
    assert derived.volume == supplied.volume, (derived, supplied)


def test_widening_a_subset_whose_rank_does_not_match_its_array():
    """A border memlet can name an array whose rank its subset does not share.

    ``fft_3d`` reaches propagation with a rank-3 ``0:1024, r_index, s_index`` on the rank-1
    ``__gather0_o``. Indexing the per-dimension fallback by the subset's own dimension then walks
    off its end with an IndexError, so a mismatched rank widens to the whole array instead -- the
    same over-approximation this fallback already makes, and the only sound answer when the
    dimensions cannot be matched up.
    """
    from dace.sdfg.propagation import widen_inner_symbol_dims

    desc = dace.data.Array(dace.float64, [1024])
    inner_only = dace.subsets.Range.from_string('0:1024, r_index, s_index')
    widened = widen_inner_symbol_dims(inner_only, desc, {'N': dace.int64}, '__gather0_o')
    assert len(widened) == 1, f'a mismatched rank must widen to the array, got {widened}'
    assert str(widened) == '0:1024', str(widened)


def test_a_mismatched_rank_with_only_outer_symbols_is_left_alone():
    """Rank disagreement alone is not a reason to widen: the fallback exists for symbols that do
    not survive outside the nested SDFG, and a subset naming none of them still describes what it
    describes."""
    from dace.sdfg.propagation import widen_inner_symbol_dims

    desc = dace.data.Array(dace.float64, [1024])
    outer_only = dace.subsets.Range.from_string('0:1024, 0:N')
    kept = widen_inner_symbol_dims(outer_only, desc, {'N': dace.int64}, '__gather0_o')
    assert str(kept) == '0:1024, 0:N', str(kept)


def test_matching_rank_widens_only_the_inner_symbol_dimension():
    """The per-dimension path is unchanged: only the dim naming an inside-only symbol widens."""
    from dace.sdfg.propagation import widen_inner_symbol_dims

    desc = dace.data.Array(dace.float64, [8, 16])
    mixed = dace.subsets.Range.from_string('0:N, r_index')
    widened = widen_inner_symbol_dims(mixed, desc, {'N': dace.int64}, 'arr')
    assert str(widened) == '0:N, 0:16', str(widened)


if __name__ == '__main__':
    test_conditional()
    test_conditional_nested()
    test_runtime_conditional()
    test_nsdfg_memlet_propagation_with_one_sparse_dimension()
    test_strided_write_keeps_the_multiplier()
    test_a_supplied_symbol_table_propagates_what_the_derived_one_does()
    test_widening_a_subset_whose_rank_does_not_match_its_array()
    test_a_mismatched_rank_with_only_outer_symbols_is_left_alone()
    test_matching_rank_widens_only_the_inner_symbol_dimension()
