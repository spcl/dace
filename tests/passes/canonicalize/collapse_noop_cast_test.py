# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`CollapseNoOpCast`.

The pass rewrites a no-op cast tasklet ``x = cast(y)`` into the plain assignment
``x = y`` when -- and only when -- the cast is genuine noise: ``y`` is a data/symbol
reference (not a constant literal) and source, destination and cast-target dtype all
coincide. A real conversion (differing dtypes) and a typed constant are left alone.

Every hand-built case is mirrored by an equivalent ``@dace.program`` fixture, and the
final case proves the intended interaction with ``TrivialTaskletElimination``: a
collapsed no-op cast becomes trivial and is eliminated, while a genuine cast survives
BOTH passes.
"""
import numpy as np

import dace
from dace import subsets, symbolic
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.collapse_noop_cast import CollapseNoOpCast
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination


def build_cast_sdfg(src_ty, dst_ty, body):
    """AccessNode(src_ty) -> tasklet(body) -> AccessNode(dst_ty), a single scalar copy."""
    sdfg = dace.SDFG('cast')
    sdfg.add_scalar('a', src_ty, transient=True)
    sdfg.add_scalar('b', dst_ty, transient=True)
    st = sdfg.add_state()
    a, b = st.add_access('a'), st.add_access('b')
    tasklet = st.add_tasklet('cast', {'inp'}, {'out'}, body)
    st.add_edge(a, None, tasklet, 'inp', dace.Memlet('a[0]'))
    st.add_edge(tasklet, 'out', b, None, dace.Memlet('b[0]'))
    return sdfg, st, tasklet


def only_tasklet(sdfg):
    tasklets = [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 1
    return tasklets[0]


def test_noop_cast_collapsed():
    # dace.<type>(...) spelling that _datatype_converter emits, source == dest == target.
    sdfg, _, tasklet = build_cast_sdfg(dace.float64, dace.float64, 'out = dace.float64(inp)')
    assert tasklet.code.as_string == 'out = dace.float64(inp)'  # pre-state: the cast is present (RED baseline)
    count = CollapseNoOpCast().apply_pass(sdfg, {})
    assert count == 1
    assert tasklet.code.as_string == 'out = inp'


def test_bare_spelling_collapsed():
    # The bare ``int64(...)`` spelling the Fortran frontend / sympy printer produce.
    sdfg, _, tasklet = build_cast_sdfg(dace.int64, dace.int64, 'out = int64(inp)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) == 1
    assert tasklet.code.as_string == 'out = inp'


def test_subscript_argument_collapsed():
    sdfg = dace.SDFG('cast_sub')
    sdfg.add_array('a', (4, ), dace.float32, transient=True)
    sdfg.add_scalar('b', dace.float32, transient=True)
    st = sdfg.add_state()
    a, b = st.add_access('a'), st.add_access('b')
    tasklet = st.add_tasklet('cast', {'inp'}, {'out'}, 'out = dace.float32(inp)')
    st.add_edge(a, None, tasklet, 'inp', dace.Memlet('a[1]'))
    st.add_edge(tasklet, 'out', b, None, dace.Memlet('b[0]'))
    assert CollapseNoOpCast().apply_pass(sdfg, {}) == 1
    assert tasklet.code.as_string == 'out = inp'


def test_genuine_cast_kept():
    # float32 -> float64 is a real conversion: source dtype != target, so it must survive.
    sdfg, _, tasklet = build_cast_sdfg(dace.float32, dace.float64, 'out = dace.float64(inp)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert tasklet.code.as_string == 'out = dace.float64(inp)'


def test_typed_constant_kept():
    # ``float64(2.0)`` is a typed constant, not a no-op cast of a variable -- leave it.
    sdfg, _, tasklet = build_cast_sdfg(dace.float64, dace.float64, 'out = dace.float64(2.0)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert tasklet.code.as_string == 'out = dace.float64(2.0)'


def test_non_cast_tasklet_untouched():
    sdfg, _, tasklet = build_cast_sdfg(dace.float64, dace.float64, 'out = inp * 2.0')
    before = tasklet.code.as_string
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert tasklet.code.as_string == before


# ----------------------------------------------------------------------------------
# @dace.program fixtures -- same shapes, driven through the frontend.
# ----------------------------------------------------------------------------------


@dace.program
def noop_astype_prog(a: dace.float64[8], b: dace.float64[8]):
    b[:] = a.astype(dace.float64)


@dace.program
def genuine_astype_prog(a: dace.float32[8], b: dace.float64[8]):
    b[:] = a.astype(dace.float64)


def cast_tasklet_bodies(sdfg):
    return [
        n.code.as_string for st in sdfg.all_states() for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and 'float' in n.code.as_string
    ]


def test_program_noop_astype_collapsed_and_correct():
    sdfg = noop_astype_prog.to_sdfg(simplify=False)
    # The frontend emits a no-op cast tasklet ``__out = dace.float64(__inp)``.
    assert any('dace.float64' in body for body in cast_tasklet_bodies(sdfg))
    assert CollapseNoOpCast().apply_pass(sdfg, {}) == 1
    assert not any('dace.float64' in body for body in cast_tasklet_bodies(sdfg))

    a = np.random.rand(8).astype(np.float64)
    b = np.zeros(8, dtype=np.float64)
    sdfg(a=a, b=b)
    assert np.allclose(b, a)


def test_program_genuine_astype_kept_and_correct():
    sdfg = genuine_astype_prog.to_sdfg(simplify=False)
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert any('dace.float64' in body for body in cast_tasklet_bodies(sdfg))

    a = (np.random.rand(8) * 1e8).astype(np.float32)
    b = np.zeros(8, dtype=np.float64)
    sdfg(a=a, b=b)
    # The narrowing to float32 then widening must be preserved -- b equals the float32 view.
    assert np.allclose(b, a.astype(np.float64))


# ----------------------------------------------------------------------------------
# Interaction with TrivialTaskletElimination.
# ----------------------------------------------------------------------------------


def test_collapsed_noop_then_eliminated():
    """A no-op cast collapses to ``out = inp`` and is then legitimately eliminated."""
    sdfg, _, _ = build_cast_sdfg(dace.float64, dace.float64, 'out = dace.float64(inp)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) == 1
    assert sdfg.apply_transformations_repeated(TrivialTaskletElimination) == 1
    assert not any(isinstance(n, nodes.Tasklet) for st in sdfg.all_states() for n in st.nodes())


def test_genuine_cast_survives_both_passes():
    """A real float32 -> float64 cast is refused by both passes -- never dropped."""
    sdfg, _, tasklet = build_cast_sdfg(dace.float32, dace.float64, 'out = dace.float64(inp)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert sdfg.apply_transformations_repeated(TrivialTaskletElimination) == 0
    assert tasklet.code.as_string == 'out = dace.float64(inp)'


def test_idempotent():
    """Re-running must be a no-op: the pass leads the canonicalize ``clean`` block, which is
    expected to reach a fixed point, so the rewritten ``out = inp`` must not re-match."""
    sdfg, _, tasklet = build_cast_sdfg(dace.float64, dace.float64, 'out = dace.float64(inp)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) == 1
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert tasklet.code.as_string == 'out = inp'


# ----------------------------------------------------------------------------------
# The cast argument is a SCOPED symbol -- a map parameter, which no declaration table holds.
#
# ``MapEntry.new_symbols`` types a parameter as ``result_type_of`` over the range's begin and
# END, and ``Range`` stores the end as ``N - 1``, whose integer literal infers as int64. So the
# sugared ``'0:N'`` reports int64 whatever ``N`` is declared, while an explicit ``Range`` over
# two pure int32 bounds reports int32. Both cases have to be READ; a guessed int64 collapses the
# genuine widening in the int32 case and drops a real conversion.
# ----------------------------------------------------------------------------------


def build_map_param_cast_sdfg(ndrange, out_ty, body):
    """map(ndrange) -> tasklet(body) -> AccessNode(out_ty), the cast argument being the parameter."""
    sdfg = dace.SDFG('param_cast')
    sdfg.add_symbol('M', dace.int32)
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('b', (16, ), out_ty, transient=True)
    st = sdfg.add_state()
    entry, exit_node = st.add_map('m', ndrange)
    tasklet = st.add_tasklet('cast', {}, {'out'}, body)
    write = st.add_write('b')
    exit_node.add_in_connector('IN_b')
    exit_node.add_out_connector('OUT_b')
    st.add_edge(entry, None, tasklet, None, dace.Memlet())
    st.add_edge(tasklet, 'out', exit_node, 'IN_b', dace.Memlet('b[0]'))
    st.add_edge(exit_node, 'OUT_b', write, None, dace.Memlet('b[0:16]'))
    return sdfg, only_tasklet(sdfg)


def test_cast_of_a_sugared_map_parameter_to_its_own_width_is_collapsed():
    """``for i in 0:N`` carries an int64 parameter, so ``int64(i)`` into an int64 slot is noise."""
    sdfg, tasklet = build_map_param_cast_sdfg({'i': '0:N'}, dace.int64, 'out = int64(i)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) == 1
    assert tasklet.code.as_string == 'out = i'


def test_cast_of_a_32_bit_map_parameter_to_64_bits_is_a_real_widening_and_is_kept():
    """An explicit ``M:N`` range over two int32 bounds carries an int32 parameter: ``int64(i)``
    widens it, and dropping the cast would store 32 bits where 64 are read."""
    sdfg, tasklet = build_map_param_cast_sdfg(
        {'i': subsets.Range([(symbolic.pystr_to_symbolic('M'), symbolic.pystr_to_symbolic('N'), 1)])}, dace.int64,
        'out = int64(i)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert tasklet.code.as_string == 'out = int64(i)'


def test_cast_of_a_32_bit_map_parameter_to_its_own_width_is_collapsed():
    """The same int32 parameter cast to int32 into an int32 slot IS noise -- the pass must
    distinguish the two by reading the parameter's width, not by refusing every symbol."""
    sdfg, tasklet = build_map_param_cast_sdfg(
        {'i': subsets.Range([(symbolic.pystr_to_symbolic('M'), symbolic.pystr_to_symbolic('N'), 1)])}, dace.int32,
        'out = int32(i)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) == 1
    assert tasklet.code.as_string == 'out = i'


def test_a_cast_of_an_undeclared_name_is_left_alone():
    """Nothing declares ``ghost``, and the destination and cast target are both int64 -- so any
    default width for the source would make this look like a no-op and collapse it. An
    undeterminable width is reported by refusing to rewrite, never by defaulting to one."""
    sdfg, _, tasklet = build_cast_sdfg(dace.int64, dace.int64, 'out = int64(ghost)')
    assert CollapseNoOpCast().apply_pass(sdfg, {}) is None
    assert tasklet.code.as_string == 'out = int64(ghost)'


if __name__ == '__main__':
    test_idempotent()
    test_noop_cast_collapsed()
    test_bare_spelling_collapsed()
    test_subscript_argument_collapsed()
    test_genuine_cast_kept()
    test_typed_constant_kept()
    test_non_cast_tasklet_untouched()
    test_program_noop_astype_collapsed_and_correct()
    test_program_genuine_astype_kept_and_correct()
    test_collapsed_noop_then_eliminated()
    test_genuine_cast_survives_both_passes()
    test_cast_of_a_sugared_map_parameter_to_its_own_width_is_collapsed()
    test_cast_of_a_32_bit_map_parameter_to_64_bits_is_a_real_widening_and_is_kept()
    test_cast_of_a_32_bit_map_parameter_to_its_own_width_is_collapsed()
    test_a_cast_of_an_undeclared_name_is_left_alone()
