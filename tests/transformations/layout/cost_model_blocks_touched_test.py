# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Average new blocks touched per iteration -- the layout-sensitive term the cost model pays for.

The invariant these pin: the metric must SEE a layout change. A contiguous-inner layout must cost
less than a strided one (Permute), and a contiguous tile must cost less than a scattered one even
when the reuse is on a non-innermost loop (Block/AoSoA). The second is the one the old
innermost-only fraction could not express."""
import itertools

import dace
import pytest
import sympy as sp

from dace.symbolic import int_floor, pystr_to_symbolic
from dace.transformation.layout.cost_model.blocks_touched import average_blocks_touched
from dace.transformation.layout.cost_model.access_subsets import get_access_subsets

N = dace.symbol("N")
T = dace.symbol("T")


def _brute_force_avg_blocks(extents, strides, block_size):
    """Ground truth: enumerate the nest in traversal order and count block-index CHANGES between
    consecutive iterations (+1 for the first) -- one new block message per change under the
    streaming/coalescing model."""
    total = 1
    count = 0
    prev = None
    for idx in itertools.product(*[range(e) for e in extents]):
        block = sum(i * s for i, s in zip(idx, strides)) // block_size
        if prev is not None and block != prev:
            total += 1
        prev = block
        count += 1
    return total / count


def _formula_avg_blocks(extents, strides, block_size):
    sdfg = dace.SDFG("bt_oracle")
    shape = [dace.symbol(f"E{i}") for i in range(len(extents))]
    sdfg.add_array("A", shape, dace.float64, strides=strides)
    sdfg.add_array("B", shape, dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {f"i{d}": f"0:{extents[d]}" for d in range(len(extents))})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    sub = ",".join(f"i{d}" for d in range(len(extents)))
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet(f"A[{sub}]"))
    st.add_memlet_path(t, mx, st.add_write("B"), src_conn="b", memlet=dace.Memlet(f"B[{sub}]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    return float(sp.simplify(average_blocks_touched(st, lr, get_access_subsets(st, me), block_size)["A"]))


def _blocks_2d(strides, block_size=8):
    sdfg = dace.SDFG("bt2")
    sdfg.add_array("A", [N, N], dace.float64, strides=strides)
    sdfg.add_array("B", [N, N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N", "j": "0:N"})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet("A[i,j]"))
    st.add_memlet_path(t, mx, st.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i,j]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    return sp.simplify(average_blocks_touched(st, lr, get_access_subsets(st, me), block_size)["A"])


def _blocks_tiled(ii_stride, block_size=8):
    """4D tiled C[I,J,ii,jj] over loops (I,J,ii,jj). ii is a NON-innermost loop; a contiguous tile
    gives it a sub-block stride, so its reuse is only visible if the fraction is per-dimension."""
    sdfg = dace.SDFG("bt4")
    sdfg.add_array("C", [T, T, 4, 4], dace.float64, strides=(T * 16, 16, ii_stride, 1))
    sdfg.add_array("D", [T, T, 4, 4], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"I": "0:T", "J": "0:T", "ii": "0:4", "jj": "0:4"})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("C"), me, t, dst_conn="a", memlet=dace.Memlet("C[I,J,ii,jj]"))
    st.add_memlet_path(t, mx, st.add_write("D"), src_conn="b", memlet=dace.Memlet("D[I,J,ii,jj]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    return float(sp.simplify(average_blocks_touched(st, lr, get_access_subsets(st, me), block_size)["C"]).subs(T, 64))


def nest_1d(index_expr, loop_range="0:N"):
    """1-D nest reading ``A[index_expr]`` over ``loop_range``; returns (state, loop_ranges, access_subsets)."""
    sdfg = dace.SDFG("bt_1d")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": loop_range})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet(f"A[{index_expr}]"))
    st.add_memlet_path(t, mx, st.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    return st, [{p: r for p, r in zip(me.map.params, me.map.range)}], get_access_subsets(st, me)


def test_contiguous_inner_reaches_one_over_block_size():
    """Contiguous innermost (stride 1): 8 elements share a 64B line, so ~1/8 new block per iter."""
    value = float(_blocks_2d((N, 1)).subs(N, 4096))
    assert abs(value - 1.0 / 8.0) < 0.02


def test_permute_is_creditable_in_2d():
    """A transpose (contiguous outer instead of inner) costs strictly more -- the model sees Permute."""
    row = float(_blocks_2d((N, 1)).subs(N, 4096))  # contiguous inner
    col = float(_blocks_2d((1, N)).subs(N, 4096))  # contiguous outer
    assert col > 6 * row  # ~1.0 vs ~1/8


def test_block_size_scales_the_reuse():
    """A smaller block (GPU 32B sector = 4 fp64 elements) shares fewer elements per line, so the
    contiguous-inner cost is higher than with a 64B line (8 elements)."""
    cpu_line = float(_blocks_2d((N, 1), block_size=8).subs(N, 4096))  # 64B / 8B
    gpu_sector = float(_blocks_2d((N, 1), block_size=4).subs(N, 4096))  # 32B / 8B
    assert abs(gpu_sector - 1.0 / 4.0) < 0.02
    assert gpu_sector > cpu_line


def test_outer_tile_reuse_is_credited():
    """The fix: a contiguous tile puts a sub-block stride on the ii loop, which is NOT innermost.
    Crediting reuse only on the innermost loop (the old behaviour) would make these two equal; the
    contiguous tile must come out cheaper."""
    contiguous_tile = _blocks_tiled(4)  # ii stride 4 < block 8 -> reuse on the ii loop
    scattered_tile = _blocks_tiled(4096)  # ii stride huge -> no reuse
    assert contiguous_tile < scattered_tile


def test_extent_uses_integer_floor_not_c_division():
    """A non-unit loop step must give the right iteration count: floor((end-begin)/step)+1 via
    int_floor, not a truncating C division. Range 0:N:2 is int_floor(N,2)+1 iterations."""
    sdfg = dace.SDFG("bt_step")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N:2"})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet("A[i]"))
    st.add_memlet_path(t, mx, st.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    value = average_blocks_touched(st, lr, get_access_subsets(st, me), 8)["A"]
    # stride-2 access of a unit-stride array: 2 elements per step, so ~1/4 of a line per iteration.
    assert abs(float(sp.simplify(value).subs(N, 4096)) - 1.0 / 4.0) < 0.02


def test_formula_ranks_layouts_the_same_as_the_brute_force_oracle():
    """The load-bearing property: the formula must ORDER layouts the same way the exact traversal
    count does. Absolute values need not match (the continuous fraction overcounts sub-block tiles),
    but a layout the oracle calls cheaper must never come out more expensive."""
    cases = [
        ((16, 16), (16, 1)),  # 2D row-major
        ((16, 16), (1, 16)),  # 2D col-major
        ((8, 8, 4, 4), (128, 16, 4, 1)),  # 4D contiguous tile
        ((8, 8, 4, 4), (128, 16, 64, 1)),  # 4D scattered tile
        ((32, ), (1, )),  # 1D contiguous
        ((32, ), (3, )),  # 1D strided
    ]
    block = 8
    brute = [_brute_force_avg_blocks(e, s, block) for e, s in cases]
    formula = [_formula_avg_blocks(e, s, block) for e, s in cases]
    # Every ordered pair must agree in direction (allowing ties within a small tolerance).
    for a in range(len(cases)):
        for b in range(len(cases)):
            if brute[a] < brute[b] - 1e-6:
                assert formula[a] <= formula[b] + 1e-6, (cases[a], cases[b], brute, formula)


def test_formula_converges_to_the_oracle_for_large_extents():
    """The continuous fraction is asymptotically exact: as extents grow the ceil() rounding it omits
    becomes negligible, so the 2D row-major formula approaches the true 1/8."""
    formula = float(_blocks_2d((N, 1)).subs(N, 8192))
    brute_limit = 1.0 / 8.0
    assert abs(formula - brute_limit) < 0.005


def test_formula_overcounts_small_tiles_ranking_still_holds():
    """The known, documented inaccuracy: a contiguous 4x4 tile is 16 contiguous elements = 2 blocks
    (oracle 0.125), but the per-dimension fraction reports ~0.25. Pinned so the limitation is not
    mistaken for exactness -- and so a future exact rewrite has a target."""
    extents, strides = (8, 8, 4, 4), (128, 16, 4, 1)
    assert abs(_brute_force_avg_blocks(extents, strides, 8) - 0.125) < 1e-6  # the truth
    assert _formula_avg_blocks(extents, strides, 8) > 0.2  # the overcount
    # but still below the scattered tile -- ranking intact
    assert _formula_avg_blocks(extents, strides, 8) < _formula_avg_blocks((8, 8, 4, 4), (128, 16, 64, 1), 8)


def test_non_affine_index_is_refused():
    """``A[i//2]`` and ``A[i%4]`` have a per-step address delta that still depends on i, so no
    constant per-iteration block count exists. It must be refused where it is detected: the loop
    symbol used to leak into the returned expression and only surface at the caller's float(), as a
    conversion error naming neither the array nor the index that caused it."""
    for index_expr in ("i//2", "i%4"):
        state, loop_ranges, subsets = nest_1d(index_expr)
        with pytest.raises(ValueError, match="not affine"):
            average_blocks_touched(state, loop_ranges, subsets, 8)


def test_empty_loop_range_is_refused():
    """A zero-iteration level makes total_iters 0, so the per-iteration average divided by it into
    sympy.zoo -- a poison value that compares as neither better nor worse than any real layout, and
    so silently corrupts a ranking. dace Range ends are INCLUSIVE, so "5:5" is (5, 4, 1): the end
    lies BELOW the begin, which is the zero-extent case."""
    state, loop_ranges, subsets = nest_1d("i", loop_range="5:5")
    begin, end, step = loop_ranges[0]["i"]
    extent = int_floor(pystr_to_symbolic(end) - pystr_to_symbolic(begin), pystr_to_symbolic(step)) + 1
    assert int(extent) == 0  # the nest really is empty, not merely short
    with pytest.raises(ValueError, match="empty nest"):
        average_blocks_touched(state, loop_ranges, subsets, 8)


def test_affine_result_never_contains_the_loop_parameter():
    """The property the non-affine bug violated, stated positively: a per-ITERATION average must not
    depend on the iteration. Only the shape symbol may survive, and substituting it must yield a
    plain float -- that float() is what every caller does, and what a leaked loop symbol broke."""
    state, loop_ranges, subsets = nest_1d("i")
    result = average_blocks_touched(state, loop_ranges, subsets, 8)["A"]
    assert pystr_to_symbolic("i") not in result.free_symbols
    assert float(result.subs({N: 4096})) == pytest.approx(1.0 / 8, rel=0.01)


if __name__ == "__main__":
    test_contiguous_inner_reaches_one_over_block_size()
    test_permute_is_creditable_in_2d()
    test_block_size_scales_the_reuse()
    test_outer_tile_reuse_is_credited()
    test_extent_uses_integer_floor_not_c_division()
    test_formula_ranks_layouts_the_same_as_the_brute_force_oracle()
    test_formula_converges_to_the_oracle_for_large_extents()
    test_formula_overcounts_small_tiles_ranking_still_holds()
    test_non_affine_index_is_refused()
    test_empty_loop_range_is_refused()
    test_affine_result_never_contains_the_loop_parameter()
    print("blocks_touched tests PASS")


def test_replayed_blocks_bounds_an_indirect_access():
    """A[idx[i]] has no affine subset -- the block count comes from REPLAYING the materialized index
    array (the static-indirection case). Two bounds: streaming (no reuse, block changes between
    consecutive accesses) and distinct (infinite cache). streaming >= distinct always; for a fully
    scattered index they coincide, so the answer is exact exactly where layout matters most."""
    import numpy
    import pytest
    from dace.transformation.layout.cost_model.blocks_touched import replayed_blocks_touched

    rng = numpy.random.default_rng(0)
    n, elems_per_block = 4096, 8

    # contiguous: both bounds agree at 1/8 -- replay reproduces the affine answer
    streaming, distinct = replayed_blocks_touched(numpy.arange(n), elems_per_block)
    assert streaming == pytest.approx(1.0 / 8, rel=0.01)
    assert distinct == pytest.approx(1.0 / 8, rel=0.01)

    # fully scattered: both bounds ~1 -- exact, no cache model needed
    scattered = rng.choice(10**7, size=n, replace=False)
    streaming, distinct = replayed_blocks_touched(scattered, elems_per_block)
    assert streaming == pytest.approx(1.0, rel=0.02)
    assert distinct == pytest.approx(streaming, rel=0.02)

    # clustered but shuffled: the bounds DIVERGE -- distinct says 1/8 (fits cache), streaming ~1
    # (no-reuse); the truth depends on the cache and an honest model reports the pair
    clustered = rng.permutation(n)
    streaming, distinct = replayed_blocks_touched(clustered, elems_per_block)
    assert distinct == pytest.approx(1.0 / 8, rel=0.01)
    assert streaming > 6 * distinct
    assert streaming >= distinct

    # degenerate inputs
    assert replayed_blocks_touched(numpy.array([], dtype=int), 8) == (0.0, 0.0)
    with pytest.raises(ValueError):
        replayed_blocks_touched(numpy.arange(4), 0)


def test_cross_parameter_nonaffine_index_is_refused():
    """``A[i*j]`` is affine in i for fixed j and vice versa, so the per-step guard clears it twice.
    The average still carries both parameters, which would crash the caller's float()."""
    sdfg = dace.SDFG("cross_param")
    sdfg.add_array("A", [64], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    loop_ranges = [{"i": (0, 7, 1)}, {"j": (0, 7, 1)}]
    ij = pystr_to_symbolic("i*j")
    with pytest.raises(ValueError, match="not affine in"):
        average_blocks_touched(state, loop_ranges, {"A": dace.subsets.Range([(ij, ij, 1)])}, 8)
