# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitStatements`` on a STRAIGHT-LINE multi-output loop body.

The pass used to require a ``ConditionalBlock`` or an interstate-edge assignment before it would
replicate a ``NestedSDFG`` per output group. A guard and a ``sym = idx[i]`` assignment are
DUPLICABLE -- ``_split`` deep-copies them into every clone -- so they are a shape the split
handles, not a precondition for it. The kernel this pins is::

    for i in range(1, N):
        s1 = x[i]
        a[i] = a[i - 1] + s1   # sequential carry on a
        b[i] = y[i] * 2.0      # independent, parallel

``a`` is an in-place read-modify-write, so the body's ``a`` is both an input and an output
connector. The blanket RMW refusal that used to sit in front of the split is narrowed here to the
rule that keeps the carry intact: an RMW array is fine while the group that WRITES it is the only
group that READS it. TSVC s2710 -- where a branch guard reads the array a sibling arm updates --
stays refused, and its own regression lives in
``tests/passes/canonicalize/split_statements_in_place_rmw_test.py``.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars

N = dace.symbol('N')


def _outer(name):
    """``for i in range(1, N)`` over the four global arrays, with an empty body state."""
    sdfg = dace.SDFG(name)
    for nm in ('a', 'b', 'x', 'y'):
        sdfg.add_array(nm, [N], dace.float64)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    return sdfg, loop.add_state('body', is_start_block=True)


def _attach(body, inner, inputs, outputs):
    """Wire ``inner`` into ``body`` as a whole-array NestedSDFG with the given connectors."""
    node = body.add_nested_sdfg(inner,
                                inputs=dict.fromkeys(inputs),
                                outputs=dict.fromkeys(outputs),
                                symbol_mapping={
                                    'i': 'i',
                                    'N': 'N'
                                })
    for nm in inputs:
        body.add_edge(body.add_read(nm), None, node, nm, dace.Memlet(f'{nm}[0:N]'))
    for nm in outputs:
        body.add_edge(node, nm, body.add_write(nm), None, dace.Memlet(f'{nm}[0:N]'))
    return node


def _inner(name):
    """A body SDFG holding the four global arrays plus one straight-line state."""
    inner = dace.SDFG(name)
    for nm in ('a', 'b', 'x', 'y'):
        inner.add_array(nm, [N], dace.float64)
    return inner, inner.add_state('st', is_start_block=True)


def carry_plus_independent(name, ordering_edge=False, private_temp=True):
    """The user's kernel as a NestedSDFG body: ``a[i] = a[i-1] + x[i]`` and ``b[i] = y[i] * 2``.

    :param name: SDFG name.
    :param ordering_edge: Also add the empty (ordering-only) memlet the frontend leaves between the
        two chains when they reuse one temporary name.
    :param private_temp: Feed only the carry from the temp; ``False`` shares it with ``b`` too.
    """
    sdfg, body = _outer(name)
    inner, st = _inner(name + '_body')
    inner.add_scalar('s1', dace.float64, transient=True)

    t0 = st.add_tasklet('t0', {'inp'}, {'out'}, 'out = inp')
    ws1 = st.add_access('s1')
    st.add_edge(st.add_read('x'), None, t0, 'inp', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', ws1, None, dace.Memlet('s1'))

    tadd = st.add_tasklet('_Add_', {'prev', 'sv'}, {'out'}, 'out = prev + sv')
    st.add_edge(st.add_read('a'), None, tadd, 'prev', dace.Memlet('a[i - 1]'))
    st.add_edge(ws1, None, tadd, 'sv', dace.Memlet('s1'))
    st.add_edge(tadd, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))

    ry = st.add_read('y')
    if private_temp:
        tmul = st.add_tasklet('_Mult_', {'yy'}, {'out'}, 'out = yy * 2.0')
        st.add_edge(ry, None, tmul, 'yy', dace.Memlet('y[i]'))
    else:
        tmul = st.add_tasklet('_Mult_', {'yy', 'sv'}, {'out'}, 'out = yy * sv')
        st.add_edge(ry, None, tmul, 'yy', dace.Memlet('y[i]'))
        st.add_edge(ws1, None, tmul, 'sv', dace.Memlet('s1'))
    st.add_edge(tmul, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))
    if ordering_edge:
        st.add_nedge(tadd, ry, dace.Memlet())

    return sdfg, body, _attach(body, inner, ('a', 'x', 'y'), ('a', 'b'))


def _nsdfgs(state):
    return [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]


def _run(sdfg, n=32):
    """Compile and run over fixed inputs; returns the two written arrays."""
    rng = np.random.default_rng(7)
    args = {'a': rng.random(n), 'b': np.zeros(n), 'x': rng.random(n), 'y': rng.random(n)}
    sdfg.compile()(**args, N=n)
    return args['a'], args['b']


# ===========================================================================
# The kernel splits.
# ===========================================================================
def test_straight_line_body_splits_by_output():
    """Two independent outputs, one of them an in-place carry -> one clone each."""
    sdfg, body, node = carry_plus_independent('split_carry')
    groups = SplitStatements._independent_output_groups(body, node)
    assert groups == [{'a': None}, {'b': None}]

    assert SplitStatements().apply_pass(sdfg, {}) == 1
    clones = _nsdfgs(body)
    assert len(clones) == 2
    by_output = {tuple(sorted(c.out_connectors)): c for c in clones}
    assert set(by_output) == {('a', ), ('b', )}
    # The clone that no longer writes the carried array must not read it either -- the connector
    # is dropped, so nothing reads ``a`` unordered against the sibling clone's store.
    assert 'a' not in by_output[('b', )].in_connectors
    assert 'a' in by_output[('a', )].in_connectors
    sdfg.validate()


def test_split_preserves_values():
    """The real gate: the split body computes exactly what the unsplit body computes."""
    ref_a, ref_b = _run(carry_plus_independent('split_carry_ref')[0])
    sdfg = carry_plus_independent('split_carry_cand')[0]
    assert SplitStatements().apply_pass(sdfg, {}) == 1
    got_a, got_b = _run(sdfg)
    assert np.allclose(got_a, ref_a)
    assert np.allclose(got_b, ref_b)


def test_split_is_idempotent():
    """Every clone is single-output, so a second run has nothing left to partition."""
    sdfg, body, _node = carry_plus_independent('split_carry_idem')
    assert SplitStatements().apply_pass(sdfg, {}) == 1
    before = sdfg.hash_sdfg()
    assert SplitStatements().apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before
    sdfg.validate()


def test_empty_memlet_ordering_edge_does_not_merge_groups():
    """An empty memlet constrains execution ORDER, not dataflow.

    The frontend leaves one between the two chains of ``s = x[i]; a[i] = a[i-1] + s; s = y[i];
    b[i] = s * 2.0`` (the WAR on the reused name). Walking it as if it carried a value unions the
    two independent outputs into one group, and the split then silently never fires -- which reads
    as "the pass did not match" rather than as a bug.
    """
    sdfg, body, node = carry_plus_independent('split_carry_order', ordering_edge=True)
    assert SplitStatements._independent_output_groups(body, node) == [{'a': None}, {'b': None}]
    assert SplitStatements().apply_pass(sdfg, {}) == 1
    assert len(_nsdfgs(body)) == 2
    sdfg.validate()


def test_ordering_edge_split_preserves_values():
    ref_a, ref_b = _run(carry_plus_independent('split_order_ref', ordering_edge=True)[0])
    sdfg = carry_plus_independent('split_order_cand', ordering_edge=True)[0]
    assert SplitStatements().apply_pass(sdfg, {}) == 1
    got_a, got_b = _run(sdfg)
    assert np.allclose(got_a, ref_a)
    assert np.allclose(got_b, ref_b)


# ===========================================================================
# A scalar temp both statements read is RECOMPUTED, never materialized.
# ===========================================================================
def test_shared_scalar_temp_is_recomputed_per_clone():
    """``s1 = x[i]`` feeding BOTH statements does not block the split.

    ``_split`` clones the whole body and lets ``SimplifyPass`` prune what each clone no longer
    needs, so a shared PURE local is simply recomputed in both -- the same treatment
    ``_split_one_map`` gives a shared temp on the map path, and sound here because
    :func:`has_opaque_code` has already barred any producer whose effect is not its out-memlets.
    """
    sdfg, body, node = carry_plus_independent('split_shared_temp', private_temp=False)
    assert SplitStatements._independent_output_groups(body, node) == [{'a': None}, {'b': None}]
    assert SplitStatements().apply_pass(sdfg, {}) == 1
    b_clone = next(c for c in _nsdfgs(body) if tuple(c.out_connectors) == ('b', ))
    # Recomputed, not materialized: the clone still derives the temp from ``x`` itself.
    assert 'x' in b_clone.in_connectors
    assert any(n.data == 's1' for s in b_clone.sdfg.all_states() for n in s.data_nodes())
    sdfg.validate()


def test_shared_scalar_temp_preserves_values():
    ref_a, ref_b = _run(carry_plus_independent('split_shared_ref', private_temp=False)[0])
    sdfg = carry_plus_independent('split_shared_cand', private_temp=False)[0]
    assert SplitStatements().apply_pass(sdfg, {}) == 1
    got_a, got_b = _run(sdfg)
    assert np.allclose(got_a, ref_a)
    assert np.allclose(got_b, ref_b)


# ===========================================================================
# Read-modify-write refusals.
# ===========================================================================
def cross_group_rmw(name):
    """``a[i] = a[i-1] + x[i]; b[i] = a[i-1] * 2`` -- ``b``'s group also READS the carried ``a``."""
    sdfg, body = _outer(name)
    inner, st = _inner(name + '_body')
    ra = st.add_read('a')
    tadd = st.add_tasklet('_Add_', {'prev', 'xx'}, {'out'}, 'out = prev + xx')
    st.add_edge(ra, None, tadd, 'prev', dace.Memlet('a[i - 1]'))
    st.add_edge(st.add_read('x'), None, tadd, 'xx', dace.Memlet('x[i]'))
    st.add_edge(tadd, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    tmul = st.add_tasklet('_Mult_', {'av'}, {'out'}, 'out = av * 2.0')
    st.add_edge(ra, None, tmul, 'av', dace.Memlet('a[i - 1]'))
    st.add_edge(tmul, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))
    return sdfg, body, _attach(body, inner, ('a', 'x'), ('a', 'b'))


def test_rmw_read_by_another_group_is_refused():
    """Cloning would leave ``b``'s read of ``a`` unordered against ``a``'s clone storing to it."""
    sdfg, body, node = cross_group_rmw('split_cross_rmw')
    assert SplitStatements._independent_output_groups(body, node) is None
    before = sdfg.hash_sdfg()
    assert SplitStatements().apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before


def guarded_rmw(name):
    """s2710's shape: a guard reading BOTH in-place arrays selects which one an arm updates."""
    sdfg, body = _outer(name)
    inner = dace.SDFG(name + '_body')
    for nm in ('a', 'b', 'x'):
        inner.add_array(nm, [N], dace.float64)
    inner.add_scalar('g', dace.float64, transient=True)

    pre = inner.add_state('guard', is_start_block=True)
    tg = pre.add_tasklet('g', {'av', 'bv'}, {'out'}, 'out = 1.0 if av > bv else 0.0')
    pre.add_edge(pre.add_read('a'), None, tg, 'av', dace.Memlet('a[i]'))
    pre.add_edge(pre.add_read('b'), None, tg, 'bv', dace.Memlet('b[i]'))
    pre.add_edge(tg, 'out', pre.add_write('g'), None, dace.Memlet('g'))

    cond = ConditionalBlock('arms')
    inner.add_node(cond)
    inner.add_edge(pre, cond, dace.InterstateEdge())
    then_body = ControlFlowRegion('then_body', sdfg=inner)
    st_then = then_body.add_state('t', is_start_block=True)
    ta = st_then.add_tasklet('ta', {'av', 'xv'}, {'out'}, 'out = av + xv')
    st_then.add_edge(st_then.add_read('a'), None, ta, 'av', dace.Memlet('a[i]'))
    st_then.add_edge(st_then.add_read('x'), None, ta, 'xv', dace.Memlet('x[i]'))
    st_then.add_edge(ta, 'out', st_then.add_write('a'), None, dace.Memlet('a[i]'))
    cond.add_branch(dace.properties.CodeBlock('g > 0.0'), then_body)
    else_body = ControlFlowRegion('else_body', sdfg=inner)
    st_else = else_body.add_state('e', is_start_block=True)
    tb = st_else.add_tasklet('tb', {'bv'}, {'out'}, 'out = bv * 2.0')
    st_else.add_edge(st_else.add_read('b'), None, tb, 'bv', dace.Memlet('b[i]'))
    st_else.add_edge(tb, 'out', st_else.add_write('b'), None, dace.Memlet('b[i]'))
    cond.add_branch(None, else_body)

    return sdfg, body, _attach(body, inner, ('a', 'b', 'x'), ('a', 'b'))


def test_s2710_shaped_guarded_rmw_stays_refused():
    """The guard is duplicated into EVERY clone and re-evaluated against already-updated data.

    Measured on TSVC s2710: the if-arm's update to ``a`` flips ``a[i] > b[i]`` and the else-arm's
    store to ``b`` then fires on if-arm lanes. The guard is not dataflow into either output cone,
    so the per-output read walk cannot see that both arms read both arrays -- a body with a
    conditional and an in-place output is refused outright.
    """
    sdfg, body, node = guarded_rmw('split_guarded_rmw')
    assert SplitStatements._independent_output_groups(body, node) is None
    before = sdfg.hash_sdfg()
    assert SplitStatements().apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before


# ===========================================================================
# Pipeline wiring.
# ===========================================================================
def test_length_one_scalarization_runs_before_the_split():
    """A ``(1,)`` transient Array is the frontend's spelling of a scalar temporary; normalize it
    to a real ``Scalar`` before anything downstream keys on the descriptor."""
    stages = [(lbl, type(p).__name__) for lbl, p in _build_stages()]
    convert = [i for i, (lbl, nm) in enumerate(stages) if nm == ConvertLengthOneArraysToScalars.__name__]
    split = [i for i, (lbl, nm) in enumerate(stages) if nm == SplitStatements.__name__]
    assert convert and split, stages
    assert stages[convert[0]][0] == 'prep'
    assert convert[0] < split[0]


def test_length_one_scalarization_keeps_the_signature():
    """Wired with the default knobs, so a signature-level length-1 array is NOT rewritten -- the
    caller still passes a 1-element buffer."""
    stages = _build_stages()
    convert = next(p for _lbl, p in stages if isinstance(p, ConvertLengthOneArraysToScalars))
    assert convert.preserve_abi is False


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
