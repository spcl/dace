# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What the LogP model says about strided vs contiguous, and about unstructured access.

Two families, both reduced to one number -- cache efficiency ``eps = distinct useful bytes / bytes
moved`` -- which is what the bandwidth term of the LogP model divides by:

* STRIDED: eps degrades as 1/stride, but only until the stride reaches one block. Past that every
  access already owns a block and the model SATURATES: stride 8 and stride 800 cost the same.
* UNSTRUCTURED: the model does not care that an access is "random", only how many blocks it lands
  in. Random-but-clustered is cheap; random-and-scattered is the 1/8 floor. Same access count.

fp64 on 64-byte blocks throughout, so a block holds 8 elements and eps ranges over [1/8, 1] for
reads (see the relayout tests for the [1/16, 1] write range).
"""
import numpy
import pytest
import sympy as sp

import dace
from dace.transformation.layout.cost_model.access_subsets import get_access_subsets
from dace.transformation.layout.cost_model.blocks_touched import average_blocks_touched
from dace.transformation.layout.cost_model.loggp import LogGP
from dace.transformation.layout.cost_model.relayout import (break_even_passes, nest_time_by_efficiency,
                                                            relayout_pays_by_efficiency,
                                                            single_pass_efficiency_threshold)

N = dace.symbol("N")
ELEMS_PER_BLOCK = 8  # 64-byte block / fp64
CPU = LogGP(L=95e-9, o=0.0, g=4e-9, G=1.0 / 100e9, line_bytes=64, bw_saturated=100e9, bw_core=20e9)


def blocks_per_iter(stride: int, n: int = 4096) -> float:
    """New blocks touched per iteration of ``for i: A[i]`` when A has element stride ``stride``.

    Concrete extent, not a symbol: the metric is an average over the nest, so a symbolic bound
    leaves an unevaluated expression rather than the number we want to compare.
    """
    sdfg = dace.SDFG(f"stride_{stride}")
    sdfg.add_array("A", [n], dace.float64, strides=[stride])
    sdfg.add_array("B", [n], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": f"0:{n}"})
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet("A[i]"))
    st.add_memlet_path(t, mx, st.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    lr = [{p: r for p, r in zip(me.map.params, me.map.range)}]
    return float(sp.simplify(average_blocks_touched(st, lr, get_access_subsets(st, me), ELEMS_PER_BLOCK)["A"]))


def efficiency_of(indices) -> float:
    """Measured eps for a gather at ``indices``: distinct useful bytes / bytes actually moved.

    Counts DISTINCT elements, not accesses -- otherwise re-reading one element would push eps above
    1 and the number would be measuring reuse (temporal locality) rather than how much of each
    fetched block is used (spatial locality), which is the layout question.
    """
    idx = numpy.asarray(indices)
    distinct_elements = len(numpy.unique(idx))
    blocks = len(numpy.unique(idx // ELEMS_PER_BLOCK))
    return (distinct_elements * 8) / (blocks * 64)


# --------------------------------------------------------------------------------------------- #
#  Strided vs contiguous
# --------------------------------------------------------------------------------------------- #
def test_contiguous_touches_one_block_per_eight_iterations():
    """Stride 1: 8 fp64 elements share a block, so a block message every 8 iterations. eps = 1."""
    assert blocks_per_iter(1) == pytest.approx(1.0 / 8, rel=0.01)


@pytest.mark.parametrize("stride,expected", [(1, 1 / 8), (2, 1 / 4), (4, 1 / 2), (8, 1.0)])
def test_stride_degrades_the_block_count_linearly_until_one_block_per_access(stride, expected):
    assert blocks_per_iter(stride) == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("stride", [8, 9, 16, 64, 800])
def test_stride_saturates_once_every_access_owns_a_block(stride):
    """The ceiling that bounds every layout claim in this suite: once the stride reaches one block,
    each access already pays a whole block and a WORSE stride costs nothing more. This is why 8x is
    the hard bound for fp64 -- a kernel reporting more than 8x from a Permute is confounded."""
    assert blocks_per_iter(stride) == pytest.approx(1.0, rel=0.01)


def test_the_whole_strided_span_is_exactly_the_eight_times_bound():
    """Best over worst = 8 = block bytes / element bytes. Nothing about layout can exceed it."""
    assert blocks_per_iter(8) / blocks_per_iter(1) == pytest.approx(8.0, rel=0.01)


def test_strided_access_is_bandwidth_bound_here_and_contiguous_is_more_so():
    """Both regimes come from the SAME nest by changing only the layout: the strided version moves
    8x the bytes for the same useful work, so its bandwidth term is 8x."""
    useful = 8 * 1024  # 1024 fp64 elements actually wanted
    t_contig = nest_time_by_efficiency(useful, 1.0, 1024 / 8, CPU, concurrency=148.0)
    t_strided = nest_time_by_efficiency(useful, 1.0 / 8, 1024, CPU, concurrency=148.0)
    assert float(t_strided) / float(t_contig) == pytest.approx(8.0, rel=0.05)


# --------------------------------------------------------------------------------------------- #
#  Unstructured: random-and-scattered vs random-and-close
# --------------------------------------------------------------------------------------------- #
def test_scattered_random_access_hits_the_one_eighth_floor():
    """Indices spread over a huge range almost never share a block -> one block per element."""
    rng = numpy.random.default_rng(0)
    idx = rng.choice(10**7, size=4096, replace=False)
    assert efficiency_of(idx) == pytest.approx(1.0 / 8, rel=0.02)


def test_clustered_random_access_is_far_cheaper_than_scattered():
    """SAME number of random accesses, 100x the efficiency -- the model keys on blocks landed in,
    not on whether an index is 'random'. 'Unstructured' is not the cost; SPREAD is the cost."""
    rng = numpy.random.default_rng(0)
    n = 4096
    scattered = rng.choice(10**7, size=n, replace=False)
    clustered = rng.choice(n, size=n, replace=False)  # same n accesses, packed into n/8 blocks
    assert efficiency_of(clustered) == pytest.approx(1.0)  # every element of every block is used
    assert efficiency_of(clustered) > 7 * efficiency_of(scattered)


def test_efficiency_degrades_monotonically_as_the_random_span_widens():
    """Sweep the spread with the access count fixed: eps falls from 1 to the 1/8 floor and stops."""
    rng = numpy.random.default_rng(0)
    n = 2048
    eps = [efficiency_of(rng.choice(span, size=n, replace=False)) for span in (n, 4 * n, 32 * n, 10**7)]
    assert eps == sorted(eps, reverse=True)  # monotone decreasing
    assert eps[0] == pytest.approx(1.0)
    assert eps[-1] == pytest.approx(1.0 / 8, rel=0.02)


def test_a_scattered_gather_is_below_the_single_pass_relayout_threshold():
    """Joins the two halves: eps=1/8 is under the 1/3 threshold, so relaying out a scattered gather
    pays for itself in ONE pass -- while a clustered one has nothing to win."""
    rng = numpy.random.default_rng(0)
    scattered = efficiency_of(rng.choice(10**7, size=4096, replace=False))
    clustered = efficiency_of(rng.choice(4096, size=4096, replace=False))
    assert scattered < single_pass_efficiency_threshold(1.0) < clustered
    assert relayout_pays_by_efficiency(scattered, 1.0, passes=1)
    assert not relayout_pays_by_efficiency(clustered, 1.0, passes=1)


# --------------------------------------------------------------------------------------------- #
#  Static replace, and inspector-executor as a THREAT TO VALIDITY
# --------------------------------------------------------------------------------------------- #
#  Static replace of a static indirection A[sigma[i]] reorders A once, at overhead_passes = 0.
#
#  We are NOT pursuing inspector-executor. These tests exist to keep the threat honest: I/E removes
#  the SAME indirection by the SAME reordering and differs only in paying to learn sigma at runtime,
#  so static replace is its overhead -> 0 limit and the distinction is narrow. The tests pin what we
#  may and may not claim -- in particular that at a badly scattered gather I/E's first pass wins too,
#  so a speed claim over I/E would be an overclaim. See report1 "Threats to state".
SCATTERED, MILD, PERFECT = 1.0 / 8, 1.0 / 2, 1.0
INSPECTOR = 3.0  # read the index array + bucket it: ~3 array passes of traffic


def test_static_replace_beats_inspector_executor_at_equal_efficiency():
    """Same reordering, same resulting layout -- static replace simply does not pay the inspector.
    It is strictly better wherever sigma is known without running the program."""
    assert (break_even_passes(SCATTERED, PERFECT, overhead_passes=0.0) <=
            break_even_passes(SCATTERED, PERFECT, overhead_passes=INSPECTOR))


def test_static_replace_of_a_scattered_gather_pays_immediately():
    """eps=1/8 -> gain 7 per pass against a cost of 2. One pass, no reuse argument needed."""
    assert break_even_passes(SCATTERED, PERFECT, overhead_passes=0.0) == 1
    assert relayout_pays_by_efficiency(SCATTERED, PERFECT, passes=1, overhead_passes=0.0)


def test_inspector_executor_does_get_fast_when_the_access_is_badly_scattered():
    """The answer to 'do they ever get fast?': YES, and immediately when the gather is bad enough.
    gain 7 per pass vs a cost of 2 + 3 = 5 -> the very first executor pass already wins."""
    assert relayout_pays_by_efficiency(SCATTERED, PERFECT, passes=1, overhead_passes=INSPECTOR)
    assert break_even_passes(SCATTERED, PERFECT, overhead_passes=INSPECTOR) == 1


def test_inspector_executor_needs_reuse_when_the_gather_is_only_mildly_scattered():
    """And the flip side: at eps=1/2 the gain is only 1 per pass, so the inspector needs FIVE
    executor passes. This is exactly why the I/E literature amortizes over timesteps -- the
    ordering is computed once and reused while the mesh connectivity stays fixed."""
    assert break_even_passes(MILD, PERFECT, overhead_passes=INSPECTOR) == 5
    assert not relayout_pays_by_efficiency(MILD, PERFECT, passes=4, overhead_passes=INSPECTOR)
    assert relayout_pays_by_efficiency(MILD, PERFECT, passes=5, overhead_passes=INSPECTOR)
    # Static replace needs only two, because it never pays the inspector.
    assert break_even_passes(MILD, PERFECT, overhead_passes=0.0) == 2


def test_a_costlier_inspector_needs_more_passes():
    """Monotone in the inspector cost -- the knob a real I/E implementation actually moves."""
    passes = [break_even_passes(MILD, PERFECT, overhead_passes=c) for c in (0.0, 3.0, 10.0)]
    assert passes == sorted(passes)
    assert passes[0] < passes[-1]


def test_neither_helps_when_the_gather_is_already_contiguous():
    """No reordering, static or inspected, redeems an access that is already perfect."""
    assert break_even_passes(PERFECT, PERFECT, overhead_passes=0.0) is None
    assert not relayout_pays_by_efficiency(PERFECT, PERFECT, passes=10**6, overhead_passes=0.0)


def test_a_negative_inspector_cost_is_rejected():
    with pytest.raises(ValueError):
        relayout_pays_by_efficiency(SCATTERED, PERFECT, overhead_passes=-1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and not hasattr(fn, "pytestmark"):
            fn()
    print("access-pattern cost-model tests PASS (parametrized ones via pytest)")
