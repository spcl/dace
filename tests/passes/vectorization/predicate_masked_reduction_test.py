# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.vectorization.predicate_masked_reduction.PredicateMaskedReduction`,
focused on the fault gate: predicating a masked reduction dissolves the guard and
runs the addend on EVERY lane, so a data-dependent (gather) addend -- which would
fault on the mask-false lanes -- must be refused, while a structured affine addend
(always in-bounds) is predicated as before."""
import sympy

import dace
from dace import subsets
from dace.memlet import Memlet
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.properties import CodeBlock
from dace.transformation.passes.vectorization.predicate_masked_reduction import (PredicateMaskedReduction,
                                                                                 _body_has_data_dependent_read)

N = dace.symbol('N')


def _make_masked_reduction(gather: bool) -> dace.SDFG:
    """``if c: acc += a[<idx>]`` -- a materialized-scalar-mask masked SUM reduction.
    ``gather=True`` reads ``a[idx]`` (data-dependent index), else ``a[j]`` (affine)."""
    sdfg = dace.SDFG('gather_masked' if gather else 'affine_masked')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_scalar('c', dace.int64)  # the materialized per-lane mask
    sdfg.add_scalar('acc', dace.float64, transient=True)

    cb = ConditionalBlock('mask_if')
    body = ControlFlowRegion('mask_true', sdfg=sdfg)
    cb.add_branch(CodeBlock('c'), body)
    st = body.add_state('reduce', is_start_block=True)

    an_a = st.add_access('a')
    tk = st.add_tasklet('addend', {'_i'}, {'_o'}, '_o = _i')
    an_acc = st.add_access('acc')
    j = dace.symbol('j')
    read_sub = subsets.Range([(sympy.Symbol('idx'), sympy.Symbol('idx'), 1)]) if gather \
        else subsets.Range([(j, j, 1)])
    st.add_edge(an_a, None, tk, '_i', Memlet(data='a', subset=read_sub))
    st.add_edge(tk, '_o', an_acc, None, Memlet(data='acc', subset=subsets.Range([(0, 0, 1)]),
                                               wcr='lambda x, y: x + y'))

    init = sdfg.add_state('init', is_start_block=True)
    sdfg.add_node(cb)
    sdfg.add_edge(init, cb, dace.InterstateEdge())
    return sdfg


def _num_conditionals(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock))


def test_gate_helper_affine_vs_gather():
    """The gate helper flags a data-dependent (gather) read subset and clears an affine one."""
    aff = _make_masked_reduction(gather=False)
    gat = _make_masked_reduction(gather=True)
    aff_body = next(n for n, _ in aff.all_nodes_recursive()
                    if isinstance(n, dace.SDFGState) and n.label == 'reduce')
    gat_body = next(n for n, _ in gat.all_nodes_recursive()
                    if isinstance(n, dace.SDFGState) and n.label == 'reduce')
    assert _body_has_data_dependent_read(aff, aff_body) is False
    assert _body_has_data_dependent_read(gat, gat_body) is True


def test_affine_masked_reduction_is_predicated():
    """Structured affine addend ``a[j]`` -- the mask-false lane load is in-bounds,
    so the ConditionalBlock is dissolved (predicated) as before."""
    sdfg = _make_masked_reduction(gather=False)
    assert _num_conditionals(sdfg) == 1
    applied = PredicateMaskedReduction().apply_pass(sdfg, {})
    assert applied == 1, 'affine masked reduction should be predicated'
    assert _num_conditionals(sdfg) == 0, 'the guard must be dissolved'


def test_gather_masked_reduction_is_refused():
    """Data-dependent addend ``a[idx[j]]`` -- predicating would eagerly gather on the
    mask-false lanes (fault). The pass must refuse and keep the guard intact."""
    sdfg = _make_masked_reduction(gather=True)
    assert _num_conditionals(sdfg) == 1
    applied = PredicateMaskedReduction().apply_pass(sdfg, {})
    assert applied is None, 'gather masked reduction must NOT be predicated'
    assert _num_conditionals(sdfg) == 1, 'the guard must be preserved (branch kept)'


if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main([__file__, '-v']))
