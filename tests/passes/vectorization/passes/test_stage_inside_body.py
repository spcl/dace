# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`stage_constant_access` (design section 3.1 + section 3.6)."""
import dace
from dace.transformation.passes.vectorization.stage_inside_body import (
    StageInsideBody,
    stage_constant_access,
)


def _build_state_with_an(shape=(16, ), dtype=dace.float64, name="A"):
    sdfg = dace.SDFG("stage_const_fixture")
    sdfg.add_array(name, shape, dtype, transient=False)
    state = sdfg.add_state("s")
    an = state.add_access(name)
    return sdfg, state, an


def test_helper_mints_scalar_transient_and_an_to_an_edge():
    """The helper adds a Scalar transient + a direct AN -> AN edge."""
    sdfg, state, an = _build_state_with_an()
    name = stage_constant_access(state, an, name_hint="bridge")
    desc = sdfg.arrays[name]
    assert isinstance(desc, dace.data.Scalar)
    assert desc.transient
    assert desc.dtype == dace.float64
    bridge_ans = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == name]
    assert len(bridge_ans) == 1
    bridge_an = bridge_ans[0]
    # The edge is direct AN(an) -> AN(bridge); no tasklet between.
    edges = [e for e in state.edges() if e.src is an and e.dst is bridge_an]
    assert len(edges) == 1
    assert edges[0].data is not None


def test_helper_unique_names_when_called_twice():
    """Two calls with the same name_hint produce distinct transients."""
    sdfg, state, an = _build_state_with_an()
    name_a = stage_constant_access(state, an, name_hint="bridge")
    name_b = stage_constant_access(state, an, name_hint="bridge")
    assert name_a != name_b
    assert name_a in sdfg.arrays and name_b in sdfg.arrays


def test_helper_preserves_source_dtype():
    """The Scalar transient's dtype matches the source array's element dtype."""
    sdfg, state, an = _build_state_with_an(dtype=dace.int32)
    name = stage_constant_access(state, an)
    assert sdfg.arrays[name].dtype == dace.int32


def test_pass_stub_is_noop():
    """The Pass class itself is a stub returning None."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    assert StageInsideBody().apply_pass(sdfg, {}) is None
