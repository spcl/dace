# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TIER 0 of the cost model: structural block counts + the dominance lemma. No measured parameters.

The claim tier 0 carries: if layout A touches no more request-blocks AND moves no more bytes than
layout B, then A is at least as fast for EVERY (L, G, C) -- so a layout sweep can drop B before
compiling or timing anything. The claim it refuses: ranking layouts whose counts DISAGREE (fewer
requests vs fewer bytes) -- that ranking is genuinely concurrency-dependent (C_flip), tier 2's job.
"""
import dace
import pytest
import sympy as sp

from dace.transformation.layout.cost_model.logp_analysis import (ArrayLogP, NestCounts, count_loop_nest,
                                                                 dominance_verdict, pareto_front)
from dace.transformation.layout.cost_model.loggp import LogGP, gap_from_bandwidth, nest_memory_time

N = dace.symbol("N")


def _nest(a_strides):
    sdfg = dace.SDFG(f"t0_{abs(hash(a_strides)) % 10**8}")
    sdfg.add_array("A", [N, N], dace.float64, strides=a_strides)
    sdfg.add_array("B", [N, N], dace.float64)
    sdfg.add_array("C", [N, N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N", "j": "0:N"})
    t = st.add_tasklet("t", {"a", "b"}, {"c"}, "c = a + b")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet("A[i,j]"))
    st.add_memlet_path(st.add_read("B"), me, t, dst_conn="b", memlet=dace.Memlet("B[i,j]"))
    st.add_memlet_path(t, mx, st.add_write("C"), src_conn="c", memlet=dace.Memlet("C[i,j]"))
    return count_loop_nest(st, me)


def _hand_counts(messages, bytes_moved):
    """A NestCounts with given totals: one synthetic array, total_iters = 1."""
    arr = ArrayLogP("x", False, sp.sympify(messages), sp.sympify(bytes_moved) / 64, sp.sympify(bytes_moved))
    return NestCounts(total_iters=sp.Integer(1), arrays={"x": arr}, line_bytes=64, sector_bytes=64)


def test_counts_need_no_parameters():
    """The tier-0 entry takes no LogGP: granularities are device-class constants."""
    counts = _nest((N, 1))
    assert counts.line_bytes == counts.sector_bytes == 64
    # 3 arrays, contiguous: 3/8 blocks/iter asymptotically (the continuous fraction carries a
    # (1/N)-order edge term, so use a size where it is small); C written -> (8 + 8 + 16) B/iter
    n = {N: 512}
    assert float(counts.messages().subs(n)) == pytest.approx(3 / 8 * 512 * 512, rel=0.05)
    assert float(counts.bytes_moved().subs(n)) == pytest.approx(32 * 512 * 512, rel=0.05)


def test_contiguous_dominates_transposed():
    """Row-major vs transposed A: fewer messages AND fewer bytes. The blocks_touched fractions carry
    int_floor, whose symbolic sign sympy will not resolve -- so tier 0 honestly says 'undecided'
    without sizes, and decides with a concrete N. Sizes are still ZERO measured parameters."""
    row, col = _nest((N, 1)), _nest((1, N))
    assert dominance_verdict(row, col) == "undecided"  # honest: int_floor sign is open symbolically
    assert dominance_verdict(row, col, subs={N: 256}) == "first"
    assert dominance_verdict(col, row, subs={N: 256}) == "second"


def test_simple_symbolic_counts_decide_without_sizes():
    """Where the expressions are floor-free, the nonnegative-symbol assumptions DO decide the sign
    with no sizes at all: k vs 2k."""
    k = sp.Symbol("k", nonnegative=True)
    assert dominance_verdict(_hand_counts(k, 64 * k), _hand_counts(2 * k, 128 * k)) == "first"


def test_equal_layouts_tie():
    assert dominance_verdict(_nest((N, 1)), _nest((N, 1))) == "tie"  # identical exprs simplify to 0


def test_disagreeing_counts_are_refused_not_guessed():
    """Fewer requests vs fewer bytes: tier 0 must say 'undecided', never pick -- the true ranking
    flips at C_flip = M2*L/(B1*G), which needs tier-2 parameters."""
    fewer_requests = _hand_counts(messages=1_000, bytes_moved=1_024_000)
    fewer_bytes = _hand_counts(messages=16_000, bytes_moved=512_000)
    assert dominance_verdict(fewer_requests, fewer_bytes) == "undecided"
    assert dominance_verdict(fewer_bytes, fewer_requests) == "undecided"


def test_verdict_first_implies_tier2_wins_for_every_parameter_set():
    """The dominance lemma, executed across tiers: a tier-0 'first' must mean the tier-2 time is <=
    for EVERY (L, G, C) -- sampled across three decades of each."""
    row, col = _nest((N, 1)), _nest((1, N))
    n = {N: 256}
    assert dominance_verdict(row, col, subs=n) == "first"
    m_r, b_r = float(row.messages().subs(n)), float(row.bytes_moved().subs(n))
    m_c, b_c = float(col.messages().subs(n)), float(col.bytes_moved().subs(n))
    for L in (10e-9, 95e-9, 500e-9):
        for bw in (10e9, 100e9, 1000e9):
            p = LogGP(L=L, o=0.0, g=4e-9, G=gap_from_bandwidth(bw), line_bytes=64, bw_saturated=bw,
                      bw_core=bw / 2)
            for C in (1.0, 8.0, 148.0, float("inf")):
                assert nest_memory_time(p, b_r, m_r, C) <= nest_memory_time(p, b_c, m_c, C)


def test_pareto_front_prunes_only_dominated_layouts():
    """The sweep integration: dominated layouts are dropped BEFORE compiling/timing; disagreeing
    ones both survive (each is optimal for some device)."""
    front = pareto_front({
        "contig": _hand_counts(1_000, 512_000),
        "blocked": _hand_counts(2_000, 768_000),  # worse on both -> pruned
        "fewer_bytes": _hand_counts(16_000, 256_000),  # disagrees with contig -> kept
    })
    assert front == ["contig", "fewer_bytes"]


def test_pareto_front_keeps_first_of_a_tie():
    front = pareto_front({"a": _hand_counts(100, 6_400), "b": _hand_counts(100, 6_400)})
    assert front == ["a"]


def test_undecidable_symbolic_sign_is_undecided_not_wrong():
    """A sign that genuinely depends on an unconstrained symbol must come back 'undecided'; with
    concrete sizes substituted it resolves."""
    k = sp.Symbol("k")  # NO nonnegativity assumption
    a = _hand_counts(k, 64 * k)
    b = _hand_counts(100, 6_400)
    assert dominance_verdict(a, b) == "undecided"
    assert dominance_verdict(a, b, subs={k: 50}) == "first"
    assert dominance_verdict(a, b, subs={k: 200}) == "second"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("tier-0 tests PASS")
