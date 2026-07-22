# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`SymbolDedup` -- CSE on duplicate interstate-edge symbols.

Positive cases assert that provably-equal interstate symbols are merged into one
while the SDFG still validates and runs bit-identically. Negative cases assert
that symbols which are NOT provably equal (different RHS, or a partial /
mismatched def-site set) are left untouched -- the merge must stay
value-preserving.
"""
import numpy as np

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup

N = dace.symbol('N')


def _all_assignments(sdfg):
    """Every interstate-edge assignment dict across the whole SDFG (recursive)."""
    out = []
    for nested in sdfg.all_sdfgs_recursive():
        for edge in nested.all_interstate_edges():
            if edge.data.assignments:
                out.append(dict(edge.data.assignments))
    return out


def _assigned_symbols(sdfg):
    """Set of every symbol assigned on any interstate edge (recursive)."""
    syms = set()
    for assign in _all_assignments(sdfg):
        syms |= set(assign.keys())
    return syms


def _run_dedup(sdfg):
    return Pipeline([SymbolDedup()]).apply_pass(sdfg, {})


# ----------------------------------------------------------------------
# Positive: hand-built duplicate-gather SDFG (direct, runnable)
# ----------------------------------------------------------------------


def _make_duplicate_gather_sdfg(name='dedup_positive'):
    """2-state SDFG: one interstate edge assigns two symbols to the SAME array
    read ``idx[0]``; the compute state uses each to gather-index a different
    array. This is the fused-gather duplicate distilled to its core."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_array('e', [N], dace.float64)
    sdfg.add_array('out', [2], dace.float64)
    sdfg.add_symbol('s1', dace.int64)
    sdfg.add_symbol('s2', dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    comp = sdfg.add_state('comp')
    sdfg.add_edge(init, comp, dace.InterstateEdge(assignments={'s1': 'idx[0]', 's2': 'idx[0]'}))

    rb = comp.add_read('b')
    re = comp.add_read('e')
    w = comp.add_write('out')
    t1 = comp.add_tasklet('t1', {'inp'}, {'o'}, 'o = inp')
    t2 = comp.add_tasklet('t2', {'inp'}, {'o'}, 'o = inp')
    comp.add_edge(rb, None, t1, 'inp', dace.Memlet('b[s1]'))
    comp.add_edge(t1, 'o', w, None, dace.Memlet('out[0]'))
    comp.add_edge(re, None, t2, 'inp', dace.Memlet('e[s2]'))
    comp.add_edge(t2, 'o', w, None, dace.Memlet('out[1]'))
    sdfg.validate()
    return sdfg


def test_symbol_dedup_positive_handbuilt():
    sdfg = _make_duplicate_gather_sdfg()

    # Sanity: both symbols present and duplicated on the edge.
    assert {'s1', 's2'} <= sdfg.symbols.keys()
    assert _all_assignments(sdfg) == [{'s1': 'idx[0]', 's2': 'idx[0]'}]

    n = 8
    rng = np.random.default_rng(0)
    b = rng.random(n)
    e = rng.random(n)
    idx = rng.integers(0, n, n).astype(np.int64)

    # Reference run BEFORE dedup.
    out_ref = np.zeros(2)
    sdfg(idx=idx, b=b, e=e, out=out_ref, N=n)

    ret = _run_dedup(sdfg)
    assert ret and ret['SymbolDedup'] == 1

    # Exactly one symbol survives; keeper is the shorter/lexicographically-smaller
    # name ('s1'). The dropped symbol is gone from symbols AND every assignment.
    assert 's1' in sdfg.symbols
    assert 's2' not in sdfg.symbols
    assert 's2' not in _assigned_symbols(sdfg)
    assert _all_assignments(sdfg) == [{'s1': 'idx[0]'}]

    sdfg.validate()

    # Post-dedup run must be bit-identical to the pre-dedup reference.
    out_new = np.zeros(2)
    sdfg(idx=idx, b=b, e=e, out=out_new, N=n)
    assert np.array_equal(out_ref, out_new)
    # And matches the plain-numpy gather.
    assert out_new[0] == b[idx[0]]
    assert out_new[1] == e[idx[0]]


# ----------------------------------------------------------------------
# Positive: the real gather-stencil through canonicalize (motivating case)
# ----------------------------------------------------------------------


@dace.program
def gather_stencil(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N],
                   f: dace.float64[N], idx: dace.int64[N]):
    for i in range(N):
        a[i] = b[idx[i]] + c[i]
        d[i] = e[idx[i]] + f[i]


def test_symbol_dedup_gather_stencil_bit_exact():
    sdfg = canonicalize(gather_stencil.to_sdfg(simplify=True))

    # Canonicalization leaves duplicate gather-index assignments on the fused
    # map body's interstate edges: {'idx_index': 'idx[i]', 'idx_index_0': 'idx[i]'}.
    dup_edges = [a for a in _all_assignments(sdfg) if len(a) >= 2 and len(set(a.values())) < len(a)]
    assert dup_edges, 'expected duplicate interstate assignments after canonicalize'
    n_syms_before = len(_assigned_symbols(sdfg))

    n = 16
    rng = np.random.default_rng(1)
    b = rng.random(n)
    c = rng.random(n)
    e = rng.random(n)
    f = rng.random(n)
    idx = rng.integers(0, n, n).astype(np.int64)

    a_ref = np.zeros(n)
    d_ref = np.zeros(n)
    sdfg(a=a_ref, b=b, c=c, d=d_ref, e=e, f=f, idx=idx, N=n)

    ret = _run_dedup(sdfg)
    assert ret and ret['SymbolDedup'] >= 1

    # Fewer assigned symbols, and no edge assigns two symbols the same RHS anymore.
    assert len(_assigned_symbols(sdfg)) < n_syms_before
    for assign in _all_assignments(sdfg):
        assert len(set(assign.values())) == len(assign), f'still-duplicate assignment {assign}'

    sdfg.validate()

    a_new = np.zeros(n)
    d_new = np.zeros(n)
    sdfg(a=a_new, b=b, c=c, d=d_new, e=e, f=f, idx=idx, N=n)

    assert np.array_equal(a_ref, a_new)
    assert np.array_equal(d_ref, d_new)
    assert np.allclose(a_new, b[idx] + c)
    assert np.allclose(d_new, e[idx] + f)


# ----------------------------------------------------------------------
# Negative: different RHS must NOT merge
# ----------------------------------------------------------------------


def test_symbol_dedup_different_rhs_not_merged():
    sdfg = dace.SDFG('dedup_neg_diff_rhs')
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_array('out', [2], dace.float64)
    sdfg.add_symbol('s1', dace.int64)
    sdfg.add_symbol('s2', dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    comp = sdfg.add_state('comp')
    # Same name pattern, DIFFERENT RHS: idx[0] vs idx[1].
    sdfg.add_edge(init, comp, dace.InterstateEdge(assignments={'s1': 'idx[0]', 's2': 'idx[1]'}))

    rb = comp.add_read('b')
    w = comp.add_write('out')
    t1 = comp.add_tasklet('t1', {'inp'}, {'o'}, 'o = inp')
    t2 = comp.add_tasklet('t2', {'inp'}, {'o'}, 'o = inp')
    comp.add_edge(rb, None, t1, 'inp', dace.Memlet('b[s1]'))
    comp.add_edge(t1, 'o', w, None, dace.Memlet('out[0]'))
    comp.add_edge(rb, None, t2, 'inp', dace.Memlet('b[s2]'))
    comp.add_edge(t2, 'o', w, None, dace.Memlet('out[1]'))
    sdfg.validate()

    _run_dedup(sdfg)

    # Neither symbol removed; both assignments intact.
    assert {'s1', 's2'} <= sdfg.symbols.keys()
    assert _all_assignments(sdfg) == [{'s1': 'idx[0]', 's2': 'idx[1]'}]
    sdfg.validate()


# ----------------------------------------------------------------------
# Negative: partial (mismatched def-site sets) must NOT merge
# ----------------------------------------------------------------------


def test_symbol_dedup_partial_def_sites_not_merged():
    sdfg = dace.SDFG('dedup_neg_partial')
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_symbol('s1', dace.int64)
    sdfg.add_symbol('s2', dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    mid = sdfg.add_state('mid')
    comp = sdfg.add_state('comp')
    # s1 and s2 both defined (equal RHS) on the first edge ...
    sdfg.add_edge(init, mid, dace.InterstateEdge(assignments={'s1': 'idx[0]', 's2': 'idx[0]'}))
    # ... but s2 is ALSO (re)defined on a second edge where s1 is not. The
    # def-site sets differ, so they are not provably equal everywhere.
    sdfg.add_edge(mid, comp, dace.InterstateEdge(assignments={'s2': 'idx[0]'}))
    sdfg.validate()

    _run_dedup(sdfg)

    # Both symbols survive; both edges keep their assignments.
    assert {'s1', 's2'} <= sdfg.symbols.keys()
    assert {'s1', 's2'} <= _assigned_symbols(sdfg)
    assignments = _all_assignments(sdfg)
    assert {'s1': 'idx[0]', 's2': 'idx[0]'} in assignments
    assert {'s2': 'idx[0]'} in assignments
    sdfg.validate()
