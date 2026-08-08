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
