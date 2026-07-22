# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vectorizer only tiles a map whose body is a LEAF: no inner map and no loop.

The tile emitter lane-widens a body by rewriting its dataflow into width-W tile ops. A
``LoopRegion`` is not dataflow -- it is a sequential trip counter over a carried state, and the
tile pipeline has no unroller to flatten one away, so tiling it would give W lanes one shared
counter and one shared carry. That is polybench adi (per-row tridiagonal sweeps), deriche (IIR
filter passes) and lu (triangular update): genuine carried recurrences, not lane-parallel at any
width. ``map_body_has_inner_loop`` refuses them so those kernels stay scalar and correct.

A CONDITIONAL is a different story and is deliberately still allowed: a branch lowers by MASKING
(evaluate both sides, select per lane), which is what the tile mask machinery is for. Refusing
conditionals here would cost 18 TSVC kernels (s271, s441, vif, ...) their vectorization for no
soundness gain -- the genuinely unmaskable ones are refused precisely by
``map_body_has_tiled_param_dependent_branch``.
"""
import copy

import dace
import pytest
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.utils.map_predicates import (is_vectorizable_map, map_body_has_inner_loop)

N = 16


def _map_over_body(inner: dace.SDFG, arrays):
    """Wrap ``inner`` in a ``row`` map reading and writing each of ``arrays``."""
    sdfg = dace.SDFG('wrapper')
    for a in arrays:
        sdfg.add_array(a, [N, N], dace.float64)
    st = sdfg.add_state('main', is_start_block=True)
    me, mx = st.add_map('row', dict(i=f'0:{N}'))
    ns = st.add_nested_sdfg(inner, set(arrays), set(arrays), symbol_mapping={'i': 'i'})
    for a in arrays:
        st.add_memlet_path(st.add_access(a), me, ns, dst_conn=a, memlet=dace.Memlet(f'{a}[0:{N}, 0:{N}]'))
        st.add_memlet_path(ns, mx, st.add_access(a), src_conn=a, memlet=dace.Memlet(f'{a}[0:{N}, 0:{N}]'))
    return sdfg, st, me


def _elementwise_state(container, arr: str, label: str, index: str):
    """``arr[i, <index>] = arr[i, <index>] * 2`` as one state of ``container``."""
    body = container.add_state(label, is_start_block=True)
    t = body.add_tasklet(f'{label}_t', {'cur'}, {'out'}, 'out = cur * 2')
    body.add_edge(body.add_access(arr), None, t, 'cur', dace.Memlet(f'{arr}[i, {index}]'))
    body.add_edge(t, 'out', body.add_access(arr), None, dace.Memlet(f'{arr}[i, {index}]'))
    return body


def _flat_body():
    """A body NSDFG that is pure dataflow -- the tileable shape."""
    inner = dace.SDFG('flat')
    inner.add_symbol('i', dace.int64)
    inner.add_array('p', [N, N], dace.float64)
    _elementwise_state(inner, 'p', 'body', '0')
    return inner


def _loop_body(arrays):
    """A body NSDFG that sweeps each of ``arrays`` along j in a sequential loop."""
    inner = dace.SDFG('loops')
    inner.add_symbol('i', dace.int64)
    for a in arrays:
        inner.add_array(a, [N, N], dace.float64)
    prev = None
    for a in arrays:
        loop = LoopRegion(f'sweep_{a}', f'j < {N}', 'j', 'j = 1', 'j = j + 1', sdfg=inner)
        inner.add_node(loop)
        body = loop.add_state('body', is_start_block=True)
        r, w = body.add_access(a), body.add_access(a)
        t = body.add_tasklet(f'{a}_t', {'cur', 'prev'}, {'out'}, 'out = cur + prev')
        body.add_edge(r, None, t, 'cur', dace.Memlet(f'{a}[i, j]'))
        body.add_edge(r, None, t, 'prev', dace.Memlet(f'{a}[i, j - 1]'))
        body.add_edge(t, 'out', w, None, dace.Memlet(f'{a}[i, j]'))
        if prev is None:
            inner.start_block = inner.node_id(loop)
        else:
            inner.add_edge(prev, loop, dace.InterstateEdge())
        prev = loop
    return inner


def _conditional_body():
    """A body NSDFG whose work sits under a ``ConditionalBlock`` guard."""
    inner = dace.SDFG('guarded')
    inner.add_symbol('i', dace.int64)
    inner.add_array('p', [N, N], dace.float64)
    cond = ConditionalBlock('guard', sdfg=inner)
    inner.add_node(cond, is_start_block=True)
    branch = ControlFlowRegion('then', sdfg=inner)
    cond.add_branch(dace.properties.CodeBlock('i > 0'), branch)
    _elementwise_state(branch, 'p', 'then_body', '0')
    return inner


def test_flat_body_is_accepted():
    """The gate is narrow: a pure-dataflow body stays vectorizable."""
    _sdfg, state, map_entry = _map_over_body(_flat_body(), ['p'])
    assert map_body_has_inner_loop(state, map_entry) is False
    assert is_vectorizable_map(state, map_entry, 1) is True


@pytest.mark.parametrize('arrays', [['p'], ['p', 'q'], ['p', 'q', 'v']])
def test_loop_body_is_refused(arrays):
    """ANY sequential inner loop is refused -- one carried sweep as much as three (adi)."""
    _sdfg, state, map_entry = _map_over_body(_loop_body(arrays), arrays)
    assert map_body_has_inner_loop(state, map_entry) is True
    assert is_vectorizable_map(state, map_entry, 1) is False


def test_conditional_body_is_still_allowed():
    """A lane-uniform guard is MASKABLE, so this gate must let it through -- refusing every
    conditional would strip 18 TSVC kernels of their vectorization for no soundness gain."""
    _sdfg, state, map_entry = _map_over_body(_conditional_body(), ['p'])
    assert map_body_has_inner_loop(state, map_entry) is False
    assert is_vectorizable_map(state, map_entry, 1) is True


@pytest.mark.parametrize('name', ['adi', 'deriche', 'lu'])
def test_real_kernels_are_refused_and_correct(name):
    """End-to-end: the three polybench kernels this gate catches stay value-correct.

    Skipped if the corpus harness is unavailable in this environment.
    """
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    base, checker = mp.CORPORA['poly'][1](name)
    sd = copy.deepcopy(base)
    mp.apply_config(sd, 'canon+vec', mp.cpu_params(4))
    fin = finalize_for_target(copy.deepcopy(sd), 'cpu')
    fin.name = f'{name}_inner_loop_gate_test'
    assert bool(checker(fin)), f'{name} must be value-correct after the refusal'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
