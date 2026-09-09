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

import numpy as np

import dace
from dace import nodes
from dace.transformation.passes.vectorization.resolve_mixed_dtype_binops import (ResolveMixedDtypeBinops,
                                                                                 _binop_operands)

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
    assert len(operand_dtypes) == 1, f'operands still disagree: {operand_dtypes}'
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
