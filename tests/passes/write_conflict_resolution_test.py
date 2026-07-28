# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for :class:`~dace.transformation.passes.write_conflict_resolution.ResolveWriteConflicts`.

The pass decides, from the dataflow graph alone, whether concurrent iterations
of a parallel scope can write the same element, resolves the order-independent
read-modify-writes among them into conflict-resolution edges, and reports the
rest.

Two groups of tests, deliberately:

- **hand-built SDFGs**, where no frontend has pre-applied conflict resolution,
  which is what proves the pass can own the policy on its own. Checked
  structurally AND by execution, since a race is only probabilistically
  observable and a numerical check alone proves nothing about synchronization.
- **frontend-produced SDFGs**, the acceptance corpus recorded in ``plan.md``,
  which pins the classification of the shapes a syntactic marker cannot see.
"""
import warnings

import numpy as np
import pytest

import dace
from dace.memlet import Memlet
from dace.transformation.passes.write_conflict_resolution import ResolveWriteConflicts

N = 16


def _accumulating_map(code: str) -> dace.SDFG:
    """``for i in map: b[0] = <code over __acc = b[0] and __in = a[i]>``, with
    the self-read spelled out and NO conflict resolution on the write."""
    sdfg = dace.SDFG('accumulating_map')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('update', {'__acc', '__in'}, {'__out'}, code)
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='__in', memlet=Memlet('a[i]'))
    state.add_memlet_path(state.add_read('b'), entry, tasklet, dst_conn='__acc', memlet=Memlet('b[0]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='__out', memlet=Memlet('b[0]'))
    return sdfg


def _partitioned_map() -> dace.SDFG:
    """``for i in map: b[i] = a[i] + 1`` — every iteration writes its own
    element, so there is no conflict to find."""
    sdfg = dace.SDFG('partitioned_map')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('update', {'__in'}, {'__out'}, '__out = __in + 1')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='__in', memlet=Memlet('a[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='__out', memlet=Memlet('b[i]'))
    return sdfg


def _apply(sdfg: dace.SDFG):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = ResolveWriteConflicts().apply_pass(sdfg, {})
    warned = [w for w in caught if 'write conflict' in str(w.message) or 'Order-dependent' in str(w.message)]
    return result or {'resolved': [], 'unresolved': []}, warned


def _wcrs(sdfg: dace.SDFG):
    return {edge.data.wcr for state in sdfg.all_states() for edge in state.edges() if edge.data.wcr}


@pytest.mark.parametrize('code, combiner, reference', [
    ('__out = __acc + __in', 'lambda x, y: x + y', lambda a: a.sum()),
    ('__out = __in + __acc', 'lambda x, y: x + y', lambda a: a.sum()),
    ('__out = max(__acc, __in)', 'lambda x, y: max(x, y)', lambda a: max(a.max(), 0.0)),
    ('__out = min(__in, __acc)', 'lambda x, y: min(x, y)', lambda a: min(a.min(), 0.0)),
])
def test_resolves_order_independent_accumulation(code, combiner, reference):
    """An order-independent read-modify-write becomes a conflict-resolved
    write: the self-read is dropped and the combiner moves onto the edge."""
    sdfg = _accumulating_map(code)
    result, warned = _apply(sdfg)

    assert len(result['resolved']) == 1 and not result['unresolved'] and not warned
    assert _wcrs(sdfg) == {combiner}
    sdfg.validate()

    rng = np.random.default_rng(0)
    a = rng.random(N)
    b = np.zeros(1)
    sdfg(a=a, b=b)
    assert np.allclose(b[0], reference(a))


@pytest.mark.parametrize('code, combiner', [
    ('__out = __in - __acc', 'lambda x, y: y - x'),
    ('__out = __acc % __in', 'lambda x, y: x % y'),
])
def test_resolves_but_reports_order_dependent_update(code, combiner):
    """A self-referential update whose combiner does not commute is still made
    race-free — an atomic update beats a racing one — but the result depends on
    thread order, which is a property of the PROGRAM that no lowering can fix,
    so it must be reported rather than left looking settled."""
    sdfg = _accumulating_map(code)
    result, warned = _apply(sdfg)

    assert len(result['resolved']) == 1 and not result['unresolved']
    assert not result['resolved'][0].order_independent
    assert _wcrs(sdfg) == {combiner}
    assert any('order-dependent' in str(w.message).lower() for w in warned)


def test_modulo_is_not_treated_as_order_independent():
    """``%`` is in the classic frontend's conflict-resolution table, which is
    why that frontend miscompiles it: ``(30 % 17) % 27 == 13`` while
    ``(30 % 27) % 17 == 3``. It may be applied atomically, but never silently."""
    assert (30 % 17) % 27 != (30 % 27) % 17
    result, warned = _apply(_accumulating_map('__out = __acc % __in'))
    assert not result['resolved'][0].order_independent
    assert warned


def test_reports_unresolvable_accumulation():
    """An update that does not reduce to a combiner over the element and ONE
    contribution: conflict resolution passes exactly two values, so two
    independent per-iteration inputs cannot both survive."""
    sdfg = dace.SDFG('two_contributions')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('c', [N], dace.float64)
    sdfg.add_array('b', [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('update', {'__acc', '__in', '__in2'}, {'__out'}, '__out = __acc * __in + __in2')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='__in', memlet=Memlet('a[i]'))
    state.add_memlet_path(state.add_read('c'), entry, tasklet, dst_conn='__in2', memlet=Memlet('c[i]'))
    state.add_memlet_path(state.add_read('b'), entry, tasklet, dst_conn='__acc', memlet=Memlet('b[0]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='__out', memlet=Memlet('b[0]'))

    result, warned = _apply(sdfg)

    assert not result['resolved']
    assert len(result['unresolved']) == 1
    assert warned
    assert not _wcrs(sdfg)


def test_resolves_a_nonlinear_but_factorable_update():
    """``b = b + a * b`` reads the element twice, but still factors into a
    combiner over the element and one contribution (``x + y * x``), so it is
    resolved — order-dependently."""
    sdfg = _accumulating_map('__out = __acc + __in * __acc')
    result, _ = _apply(sdfg)
    assert len(result['resolved']) == 1
    assert not result['resolved'][0].order_independent
    assert _wcrs(sdfg) == {'lambda x, y: x + y * x'}


def test_partitioned_write_is_not_a_conflict():
    sdfg = _partitioned_map()
    result, warned = _apply(sdfg)
    assert not result['resolved'] and not result['unresolved'] and not warned


def test_existing_conflict_resolution_is_left_alone():
    """A write that already carries conflict resolution, with no other read of
    the element feeding it, is sound and must not be touched or reported."""
    sdfg = dace.SDFG('already_resolved')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('contribute', {'__in'}, {'__out'}, '__out = __in')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='__in', memlet=Memlet('a[i]'))
    memlet = Memlet('b[0]')
    memlet.wcr = 'lambda x, y: x + y'
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='__out', memlet=memlet)

    result, warned = _apply(sdfg)
    assert not result['resolved'] and not result['unresolved'] and not warned


def test_folds_an_existing_conflict_resolution_over_a_self_read():
    """The shape a syntactic marker cannot see: the write IS conflict-resolved,
    but the value it contributes was computed from a separate read of the same
    element, so the combiner folds a stale value. Composing the existing
    conflict resolution with the region gives the true update, ``x + (x + y)``.
    """
    sdfg = dace.SDFG('stale_self_read')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('update', {'__acc', '__in'}, {'__out'}, '__out = __acc + __in')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='__in', memlet=Memlet('a[i]'))
    state.add_memlet_path(state.add_read('b'), entry, tasklet, dst_conn='__acc', memlet=Memlet('b[0]'))
    memlet = Memlet('b[0]')
    memlet.wcr = 'lambda x, y: x + y'
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='__out', memlet=memlet)

    result, warned = _apply(sdfg)
    assert len(result['resolved']) == 1 and not result['unresolved']
    assert _wcrs(sdfg) == {'lambda x, y: x + (x + y)'}
    assert not result['resolved'][0].order_independent
    assert warned
    sdfg.validate()


def test_reports_conflicting_overwrite():
    """Concurrent iterations writing the same element with values that do not
    depend on it: no conflict resolution can express it, so it is reported."""
    sdfg = dace.SDFG('overwrite')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('overwrite', {'__in'}, {'__out'}, '__out = __in')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='__in', memlet=Memlet('a[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='__out', memlet=Memlet('b[0]'))

    result, warned = _apply(sdfg)
    assert not result['resolved']
    assert len(result['unresolved']) == 1
    assert 'overwrite' in result['unresolved'][0].reason
    assert warned


def test_per_iteration_transient_is_not_a_conflict():
    """A transient used only inside the scope is allocated per scope instance
    (``AllocationLifetime.Scope``), so concurrent writes to it cannot collide.
    Reporting these would bury the real conflicts in noise."""
    sdfg = dace.SDFG('scope_local_transient')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    stage = state.add_tasklet('stage', {'__in'}, {'__out'}, '__out = __in * 2')
    use = state.add_tasklet('use', {'__in'}, {'__out'}, '__out = __in + 1')
    tmp = state.add_access('tmp')
    state.add_memlet_path(state.add_read('a'), entry, stage, dst_conn='__in', memlet=Memlet('a[i]'))
    state.add_edge(stage, '__out', tmp, None, Memlet('tmp[0]'))
    state.add_edge(tmp, None, use, '__in', Memlet('tmp[0]'))
    state.add_memlet_path(use, exit_node, state.add_write('b'), src_conn='__out', memlet=Memlet('b[i]'))

    result, warned = _apply(sdfg)
    assert not result['resolved'] and not result['unresolved'] and not warned


# --- The acceptance corpus, as produced by the Python frontend


def _frontend_sdfg(program, *arguments) -> dace.SDFG:
    from dace.frontend.python import nextgen
    from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import from_schedule_tree
    return from_schedule_tree(nextgen.parse_program(program, *arguments))


M = 12
_matrix = (np.zeros((M, M)), np.zeros((M, M)))


@dace.program
def _augmented(a: dace.float64[M, M], b: dace.float64[M, M]):
    for i in dace.map[0:M]:
        for j in dace.map[0:M]:
            b[i, 0] += a[i, j]


@dace.program
def _spelled_out(a: dace.float64[M, M], b: dace.float64[M, M]):
    for i in dace.map[0:M]:
        for j in dace.map[0:M]:
            b[i, 0] = b[i, 0] + a[i, j]


@dace.program
def _combiner_call(a: dace.float64[M, M], b: dace.float64[M, M]):
    for i in dace.map[0:M]:
        for j in dace.map[0:M]:
            b[i, 0] = max(b[i, 0], a[i, j])


@dace.program
def _chained_self_reference(a: dace.float64[M, M], b: dace.float64[M, M]):
    for i in dace.map[0:M]:
        for j in dace.map[0:M]:
            b[i, 0] += b[i, 0] + a[i, j]


@dace.program
def _partitioned_replacement(A: dace.float64[4, 8], out: dace.float64[4]):
    for i in dace.map[0:4]:
        out[i] = np.mean(A[i])


@dace.program
def _map_invariant_replacement(A: dace.float64[4, 8], out: dace.float64[1]):
    for i in dace.map[0:4]:
        out[0] = np.mean(A[i])


@pytest.mark.parametrize('program, arguments', [
    (_augmented, _matrix),
    (_spelled_out, _matrix),
    (_combiner_call, _matrix),
    (_partitioned_replacement, (np.zeros((4, 8)), np.zeros(4))),
])
def test_corpus_sound_programs_are_clean(program, arguments):
    """Shapes the frontend already resolves (or that never conflict) must come
    out of the pass with nothing to say -- no double resolution, no noise."""
    result, warned = _apply(_frontend_sdfg(program, *arguments))
    assert not result['resolved'] and not result['unresolved'] and not warned


def test_corpus_chained_self_reference_is_resolved():
    """``b += b + a``: ANF hoists the extra self-read into its own statement, so
    the frontend's marker sees a plain accumulation and emits a conflict
    resolution that folds a stale value — measured at ``769756.8`` where the
    serial answer is ~``10``. Composing the region recovers the true update,
    ``x + (x + y)``.

    Verified by EXECUTION under a sequential schedule: the update does not
    commute, so under a parallel schedule there is no single right answer to
    compare against, but with one iteration order there is exactly one, and the
    composed atomic update must reproduce it.
    """
    from dace import dtypes
    from dace.sdfg import nodes as sdfg_nodes

    sdfg = _frontend_sdfg(_chained_self_reference, *_matrix)
    result, warned = _apply(sdfg)
    assert len(result['resolved']) == 1 and not result['unresolved']
    assert not result['resolved'][0].order_independent
    assert warned
    sdfg.validate()

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, sdfg_nodes.MapEntry):
                node.map.schedule = dtypes.ScheduleType.Sequential

    rng = np.random.default_rng(0)
    a = rng.random((M, M))
    b = rng.random((M, M))
    expected = b.copy()
    for i in range(M):
        for j in range(M):
            expected[i, 0] = expected[i, 0] + (expected[i, 0] + a[i, j])

    result_b = b.copy()
    sdfg(a=a, b=result_b)
    assert np.allclose(result_b[:, 0], expected[:, 0])


def test_corpus_replacement_overwrite_is_reported():
    """A registry replacement expanded inside a map writing a map-invariant
    element: there is no Python statement to mark, because the writes come from
    the replacement's own subgraph."""
    result, warned = _apply(_frontend_sdfg(_map_invariant_replacement, np.zeros((4, 8)), np.zeros(1)))
    assert len(result['unresolved']) == 1
    assert 'overwrite' in result['unresolved'][0].reason
    assert warned


if __name__ == '__main__':
    for _code, _combiner, _reference in [
        ('__out = __acc + __in', 'lambda x, y: x + y', lambda a: a.sum()),
        ('__out = __in + __acc', 'lambda x, y: x + y', lambda a: a.sum()),
        ('__out = max(__acc, __in)', 'lambda x, y: max(x, y)', lambda a: max(a.max(), 0.0)),
        ('__out = min(__in, __acc)', 'lambda x, y: min(x, y)', lambda a: min(a.min(), 0.0))
    ]:
        test_resolves_order_independent_accumulation(_code, _combiner, _reference)
    for _code, _combiner in [('__out = __in - __acc', 'lambda x, y: y - x'),
                             ('__out = __acc % __in', 'lambda x, y: x % y')]:
        test_resolves_but_reports_order_dependent_update(_code, _combiner)
    test_reports_unresolvable_accumulation()
    test_resolves_a_nonlinear_but_factorable_update()
    test_modulo_is_not_treated_as_order_independent()
    test_partitioned_write_is_not_a_conflict()
    test_existing_conflict_resolution_is_left_alone()
    test_folds_an_existing_conflict_resolution_over_a_self_read()
    test_reports_conflicting_overwrite()
    test_per_iteration_transient_is_not_a_conflict()
    for _program, _arguments in [(_augmented, _matrix), (_spelled_out, _matrix), (_combiner_call, _matrix),
                                 (_partitioned_replacement, (np.zeros((4, 8)), np.zeros(4)))]:
        test_corpus_sound_programs_are_clean(_program, _arguments)
    test_corpus_chained_self_reference_is_resolved()
    test_corpus_replacement_overwrite_is_reported()
