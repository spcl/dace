# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize must not hand ``LoopToMap`` a scope summary that its own inlining made stale.

A map entry/exit edge summarizes the body it encloses, so it is derived data. ``InlineSDFG``
replaces a NestedSDFG with its body and rewrites the body memlets into outer coordinates, but the
enclosing scope keeps the whole-array approximation the opaque nested node had forced. Nothing is
miscompiled -- the summary is only ever wider than the truth -- but ``LoopToMap`` reads it as the
loop's write set, and a whole-array write is refused unconditionally.

Polybench ``covariance`` is where the two meet: ``cov[i, i:M] = data[:, i] @ data[:, i:M]`` lowers
to a ``MatMul`` inside a map whose exit edge claimed ``cov[0:M, 0:M]`` long after the inline had
exposed the exact ``cov[i, i:M]``. Both halves have to hold for the loop to lift -- the summary
must be rebuilt (``PropagateMemlets``), and the mirrored ragged writes must then be certified
disjoint -- so the assertion here is on the pipeline, not on either piece.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import smt_dependence
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.propagate_memlets import PropagateMemlets

M = dace.symbol('M')
N = dace.symbol('N')


@dace.program
def covariance_columns(data: dace.float64[N, M], cov: dace.float64[M, M]):
    # The trailing division is load-bearing, not decoration: it is what puts the row store inside a
    # NestedSDFG, and the inline of that nest is what strands the enclosing map's summary. Without
    # it the loop lifts on the ragged-write certificate alone and this file would test half the fix.
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (np.float64(N) - 1.0)
        cov[i:M, i] = cov[i, i:M]


def residual_loops(sdfg: dace.SDFG):
    return [c.label for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


def scope_summaries(sdfg: dace.SDFG):
    """Every map-exit edge's subset, keyed by where it sits, i.e. the derived scope summaries."""
    return {
        (st.label, st.edge_id(e)): str(e.data.subset)
        for st in sdfg.all_states()
        for e in st.edges() if isinstance(e.src, nodes.MapExit)
    }


@pytest.mark.skipif(not smt_dependence.has_z3(), reason='the ragged-write certificate is SMT-backed')
def test_the_mirrored_column_loop_parallelizes():
    sdfg = covariance_columns.to_sdfg(simplify=True)
    sdfg.simplify()
    canonicalize(sdfg, validate_all=False)
    assert residual_loops(sdfg) == []


def test_canonicalize_leaves_the_scope_summaries_at_a_fixpoint():
    """The invariant behind the lift, and the one that holds whatever canonicalize decides to
    parallelize: propagation must have nothing left to tighten. Comparing against a re-propagated
    copy states it directly -- a hardcoded subset would only describe today's pipeline, and a
    stale summary is invisible to every other assertion because it is merely too WIDE, never
    wrong."""
    sdfg = covariance_columns.to_sdfg(simplify=True)
    sdfg.simplify()
    canonicalize(sdfg, validate_all=False)
    before = scope_summaries(sdfg)
    PropagateMemlets().apply_pass(sdfg, {})
    assert scope_summaries(sdfg) == before


def test_propagating_is_value_preserving():
    """``PropagateMemlets`` rewrites derived edges only, so running it must not change results."""
    rng = np.random.default_rng(0)
    data = rng.random((24, 16))
    ref = np.zeros((16, 16))
    for i in range(16):
        ref[i, i:16] = data[:, i] @ data[:, i:16] / (24.0 - 1.0)
        ref[i:16, i] = ref[i, i:16]

    sdfg = covariance_columns.to_sdfg(simplify=True)
    sdfg.simplify()
    PropagateMemlets().apply_pass(sdfg, {})
    got = np.zeros((16, 16))
    sdfg(data=data, cov=got, M=16, N=24)
    assert np.allclose(got, ref)


if __name__ == '__main__':
    test_the_mirrored_column_loop_parallelizes()
    test_canonicalize_leaves_the_scope_summaries_at_a_fixpoint()
    test_propagating_is_value_preserving()
