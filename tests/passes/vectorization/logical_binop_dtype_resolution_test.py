# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``a and b`` is a binop too, and the single-dtype rule applies to it.

``ResolveMixedDtypeBinops`` unifies the operand dtypes of every tasklet the tile converter will
turn into a ``TileBinop``, because that converter locks one dtype per lib node (design 6.2) and
raises ``NotImplementedError`` -- not a graceful refusal -- when it sees two. Its detector matched
``ast.BinOp`` and ``ast.Compare`` only, so Python's ``and`` / ``or`` (``ast.BoolOp``) walked past
it unresolved.

CloudSC is the case: a Fortran ``LOGICAL`` reaches DaCe as an int array, and combining it with a
comparison's ``bool`` gives ``__t1 = _in_ldcum_0 and __t0`` over ``{int, bool}``. That raised out
of the whole ``VectorizeCPUMultiDim`` call, so a single tasklet cost the SDFG all of its tiling.
"""

import ast

import numpy as np

import dace
from dace import nodes
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.resolve_mixed_dtype_binops import (ResolveMixedDtypeBinops,
                                                                                 _binop_operands, _is_logical)

N = 8


def logical_and_sdfg(flag_dtype) -> dace.SDFG:
    """``out[i] = flags[i] and (vals[i] > 0)``, with ``flags`` typed by the caller."""
    sdfg = dace.SDFG(f'logical_and_{flag_dtype.to_string()}')
    sdfg.add_array('flags', (N, ), flag_dtype)
    sdfg.add_array('vals', (N, ), dace.float64)
    sdfg.add_array('out', (N, ), dace.bool_)
    sdfg.add_scalar('_cmp', dace.bool_, transient=True)
    state = sdfg.add_state('body', is_start_block=True)

    compare = state.add_tasklet('cmp', {'_in'}, {'_out'}, '_out = _in > 0.0')
    state.add_edge(state.add_access('vals'), None, compare, '_in', dace.Memlet('vals[0]'))
    cmp_access = state.add_access('_cmp')
    state.add_edge(compare, '_out', cmp_access, None, dace.Memlet('_cmp[0]'))

    conj = state.add_tasklet('conj', {'_a', '_b'}, {'_o'}, '_o = _a and _b')
    state.add_edge(state.add_access('flags'), None, conj, '_a', dace.Memlet('flags[0]'))
    state.add_edge(cmp_access, None, conj, '_b', dace.Memlet('_cmp[0]'))
    state.add_edge(conj, '_o', state.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg


def test_a_logical_conjunction_is_detected_as_a_binop():
    tasklet = nodes.Tasklet('conj', {'_a', '_b'}, {'_o'}, '_o = _a and _b')
    detected = _binop_operands(tasklet)
    assert detected is not None, '``and`` was not recognised as a two-operand binop'
    out_conn, a_conn, b_conn, is_cmp = detected
    assert (out_conn, {a_conn, b_conn}) == ('_o', {'_a', '_b'})
    assert is_cmp is False, 'the converter checks the output dtype of ``and``, so it is not a comparison'


def test_a_three_operand_conjunction_is_left_alone():
    """``a and b and c`` has three values and no two-operand lib node to lower to."""
    assert _binop_operands(nodes.Tasklet('conj3', {'_a', '_b', '_c'}, {'_o'}, '_o = _a and _b and _c')) is None


def test_mixed_int_and_bool_operands_get_a_cast():
    sdfg = logical_and_sdfg(dace.int32)
    assert ResolveMixedDtypeBinops().apply_pass(sdfg, {}) is not None, 'the mixed conjunction was not resolved'

    state = next(iter(sdfg.all_states()))
    conj = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == 'conj')
    operand_dtypes = {sdfg.arrays[e.data.data].dtype for e in state.in_edges(conj) if e.data and e.data.data}
    # BOOL specifically, not merely "the same": numpy promotion answers ``int`` for int + bool, and
    # ``ConvertTaskletsToTileOps`` asserts that a ``&&`` / ``||`` TileBinop has bool inputs -- an
    # int operand fails that invariant, which aborts the whole SDFG's vectorization.
    assert operand_dtypes == {dace.bool_}, f'logical operands must unify at bool, got {operand_dtypes}'
    sdfg.validate()


def test_uniform_operands_are_not_touched():
    """The control: bool ``and`` bool needs no cast, so the pass must report no change for it."""
    assert ResolveMixedDtypeBinops().apply_pass(logical_and_sdfg(dace.bool_), {}) is None


def test_the_resolved_conjunction_still_computes_the_same_values():
    """Executable: the cast must not change which lanes come out true."""
    flags = np.array([0, 1, 0, 1, 1, 0, 1, 1], dtype=np.int32)
    vals = np.array([1.0, -1.0, 2.0, 3.0, -2.0, 0.0, 0.5, 4.0], dtype=np.float64)
    expected = np.array([bool(flags[0]) and vals[0] > 0.0] + [False] * (N - 1))

    plain = logical_and_sdfg(dace.int32)
    got = np.zeros(N, dtype=np.bool_)
    plain(flags=flags, vals=vals, out=got)
    assert got[0] == expected[0]

    resolved = logical_and_sdfg(dace.int32)
    ResolveMixedDtypeBinops().apply_pass(resolved, {})
    resolved.name = 'logical_and_resolved'
    after = np.zeros(N, dtype=np.bool_)
    resolved(flags=flags, vals=vals, out=after)
    assert np.array_equal(after, got)


def int_condition_ite_sdfg() -> dace.SDFG:
    """``out[i] = ITE(flags[i], hot[i], cold[i])`` with an INT condition -- a Fortran LOGICAL."""
    sdfg = dace.SDFG('int_condition_ite')
    sdfg.add_array('flags', (N, ), dace.int32)
    sdfg.add_array('hot', (N, ), dace.float64)
    sdfg.add_array('cold', (N, ), dace.float64)
    sdfg.add_array('out', (N, ), dace.float64)
    state = sdfg.add_state('body', is_start_block=True)

    blend = state.add_tasklet('blend', {'_c', '_t', '_e'}, {'_o'}, '_o = _t if _c else _e')
    state.add_edge(state.add_access('flags'), None, blend, '_c', dace.Memlet('flags[0]'))
    state.add_edge(state.add_access('hot'), None, blend, '_t', dace.Memlet('hot[0]'))
    state.add_edge(state.add_access('cold'), None, blend, '_e', dace.Memlet('cold[0]'))
    state.add_edge(blend, '_o', state.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg


def test_an_int_ite_condition_is_cast_to_bool():
    """``TileITE``'s ``_mask`` is bool by contract, and the converter asserts it.

    A condition lifted from a bare value keeps that value's dtype, so CloudSC's ``if ldcum[jl]``
    over an int LOGICAL reached the mask as ``int`` and failed the invariant -- aborting the whole
    SDFG's vectorization, not just that one kernel.
    """
    sdfg = int_condition_ite_sdfg()
    assert ResolveMixedDtypeBinops().apply_pass(sdfg, {}) is not None

    state = next(iter(sdfg.all_states()))
    blend = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == 'blend')
    cond_edge = next(e for e in state.in_edges(blend) if e.dst_conn == '_c')
    assert sdfg.arrays[cond_edge.data.data].dtype == dace.bool_, 'the int condition was left un-cast'
    sdfg.validate()


def test_a_bool_ite_condition_is_left_alone():
    """The control: a bool condition already satisfies the contract, so nothing is inserted."""
    sdfg = int_condition_ite_sdfg()
    sdfg.arrays['flags'].dtype = dace.bool_
    assert ResolveMixedDtypeBinops().apply_pass(sdfg, {}) is None


def test_the_cast_condition_selects_the_same_lanes():
    """Executable: casting the condition must not change which arm each lane takes."""
    flags = np.array([0, 2, 0, 1, 5, 0, 1, 0], dtype=np.int32)
    hot = np.full(N, 1.0)
    cold = np.full(N, -1.0)

    plain = int_condition_ite_sdfg()
    before = np.zeros(N)
    plain(flags=flags, hot=hot, cold=cold, out=before)

    resolved = int_condition_ite_sdfg()
    ResolveMixedDtypeBinops().apply_pass(resolved, {})
    resolved.name = 'int_condition_ite_resolved'
    after = np.zeros(N)
    resolved(flags=flags, hot=hot, cold=cold, out=after)

    assert before[0] == cold[0], 'the fixture stopped selecting on flags[0] == 0'
    assert np.array_equal(after, before)


def three_operand_conjunction_sdfg(flag_dtype) -> dace.SDFG:
    """``out[i] = flags[i] and p[i] and q[i]`` -- ONE tasklet holding a three-value ``BoolOp``.

    The shape CloudSC builds: ``SameWriteSetIfElseToITECFG`` lifts a whole guard into a single
    ``lift_cond_expr`` tasklet, so a three-term Fortran condition arrives as one ``a and b and c``.
    """
    sdfg = dace.SDFG(f'conj3_{flag_dtype.to_string()}')
    sdfg.add_array('flags', (N, ), flag_dtype)
    sdfg.add_array('p', (N, ), dace.bool_)
    sdfg.add_array('q', (N, ), dace.bool_)
    sdfg.add_array('out', (N, ), dace.bool_)
    state = sdfg.add_state('body', is_start_block=True)

    conj = state.add_tasklet('conj3', {'_a', '_b', '_c'}, {'_o'}, '_o = _a and _b and _c')
    state.add_edge(state.add_access('flags'), None, conj, '_a', dace.Memlet('flags[0]'))
    state.add_edge(state.add_access('p'), None, conj, '_b', dace.Memlet('p[0]'))
    state.add_edge(state.add_access('q'), None, conj, '_c', dace.Memlet('q[0]'))
    state.add_edge(conj, '_o', state.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg


def logical_tasklets(sdfg: dace.SDFG):
    """Every tasklet whose body is an ``and`` / ``or``, paired with its owning state."""
    found = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for n in state.nodes():
                if isinstance(n, nodes.Tasklet) and _is_logical(n):
                    found.append((sd, state, n))
    return found


def test_a_three_operand_conjunction_splits_into_two_operand_ops():
    """``SplitTasklets``' contract is ONE primitive op per tasklet, and ``a and b and c`` is two.

    Emitted as a single statement it stays a three-value ``BoolOp``, which
    ``_binop_operands`` declines (see ``test_a_three_operand_conjunction_is_left_alone``) -- so it
    walks past the bool cast and reaches the converter as a ``&&`` ``TileBinop`` holding the raw
    int operand, failing the ``logical_binops_are_bool`` invariant and aborting the whole SDFG.
    """
    sdfg = three_operand_conjunction_sdfg(dace.int32)
    SplitTasklets().apply_pass(sdfg, {})

    for _, _, tasklet in logical_tasklets(sdfg):
        tree = ast.parse(tasklet.code.as_string.strip())
        rhs = tree.body[0].value
        assert len(rhs.values) == 2, (f'{tasklet.label} still holds a {len(rhs.values)}-value BoolOp; '
                                      'the split must leave two operands per logical tasklet')


def test_every_logical_operand_of_a_three_term_conjunction_unifies_at_bool():
    """The whole point: after split + resolve, no logical op is left holding an int operand."""
    sdfg = three_operand_conjunction_sdfg(dace.int32)
    SplitTasklets().apply_pass(sdfg, {})
    ResolveMixedDtypeBinops().apply_pass(sdfg, {})

    logical = logical_tasklets(sdfg)
    assert logical, 'the fixture stopped producing a logical tasklet'
    for sd, state, tasklet in logical:
        dtypes_in = {sd.arrays[e.data.data].dtype for e in state.in_edges(tasklet) if e.data and e.data.data}
        assert dtypes_in == {dace.bool_}, (f'{tasklet.label} operands must unify at bool, got {dtypes_in}')
    sdfg.validate()


def test_the_split_three_term_conjunction_computes_the_same_values():
    """Executable: splitting and casting must not change which lanes come out true."""
    flags = np.array([0, 2, 0, 1, 5, 0, 1, 0], dtype=np.int32)
    p = np.array([True, True, False, True, True, False, True, False])
    q = np.array([True, True, True, False, True, True, False, False])

    plain = three_operand_conjunction_sdfg(dace.int32)
    before = np.zeros(N, dtype=np.bool_)
    plain(flags=flags, p=p, q=q, out=before)
    assert before[0] == (bool(flags[0]) and p[0] and q[0]), 'the fixture stopped selecting on lane 0'

    split = three_operand_conjunction_sdfg(dace.int32)
    SplitTasklets().apply_pass(split, {})
    ResolveMixedDtypeBinops().apply_pass(split, {})
    split.name = 'conj3_resolved'
    after = np.zeros(N, dtype=np.bool_)
    split(flags=flags, p=p, q=q, out=after)
    assert np.array_equal(after, before)


def literal_operand_conjunction_sdfg(flag_dtype) -> dace.SDFG:
    """``out[i] = flags[i] and True`` -- a logical op with only ONE data connector.

    The other operand is a literal, so the two-operand detector declines it: it requires exactly
    two input connectors. The int operand then reaches the ``&&`` lib node un-cast.
    """
    sdfg = dace.SDFG(f'conj_literal_{flag_dtype.to_string()}')
    sdfg.add_array('flags', (N, ), flag_dtype)
    sdfg.add_array('out', (N, ), dace.bool_)
    state = sdfg.add_state('body', is_start_block=True)

    conj = state.add_tasklet('conj_lit', {'_a'}, {'_o'}, '_o = _a and True')
    state.add_edge(state.add_access('flags'), None, conj, '_a', dace.Memlet('flags[0]'))
    state.add_edge(conj, '_o', state.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg


def test_a_one_connector_logical_is_declined_by_the_two_operand_detector():
    """Pins WHY the broader branch is needed, not just that it works."""
    tasklet = nodes.Tasklet('conj_lit', {'_a'}, {'_o'}, '_o = _a and True')
    assert _binop_operands(tasklet) is None, 'the two-operand detector must not claim a one-connector logical'
    assert _is_logical(tasklet), 'it is still an ``and``, so the logical contract still applies to it'


def test_an_int_operand_of_a_literal_conjunction_is_cast_to_bool():
    sdfg = literal_operand_conjunction_sdfg(dace.int32)
    assert ResolveMixedDtypeBinops().apply_pass(sdfg, {}) is not None, 'the int operand was left un-cast'

    state = next(iter(sdfg.all_states()))
    conj = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == 'conj_lit')
    operand_dtypes = {sdfg.arrays[e.data.data].dtype for e in state.in_edges(conj) if e.data and e.data.data}
    assert operand_dtypes == {dace.bool_}, f'logical operands must unify at bool, got {operand_dtypes}'
    sdfg.validate()


def test_a_bool_literal_conjunction_is_left_alone():
    """The control: a bool operand already satisfies the contract, so nothing is inserted."""
    assert ResolveMixedDtypeBinops().apply_pass(literal_operand_conjunction_sdfg(dace.bool_), {}) is None


def test_the_cast_literal_conjunction_computes_the_same_values():
    """Executable, and it is NOT a formality: ``&&`` lowers to a bitwise ``&``, which disagrees
    with logical ``and`` on any operand that is truthy but not 0/1 -- ``2 & 1 == 0``."""
    flags = np.array([0, 2, 0, 1, 5, 0, 1, 0], dtype=np.int32)

    plain = literal_operand_conjunction_sdfg(dace.int32)
    before = np.zeros(N, dtype=np.bool_)
    plain(flags=flags, out=before)

    resolved = literal_operand_conjunction_sdfg(dace.int32)
    ResolveMixedDtypeBinops().apply_pass(resolved, {})
    resolved.name = 'conj_literal_resolved'
    after = np.zeros(N, dtype=np.bool_)
    resolved(flags=flags, out=after)

    assert before[0] == bool(flags[0]), 'the fixture stopped selecting on flags[0]'
    assert np.array_equal(after, before)
