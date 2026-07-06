# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`NormalizeMapBody`.

The pass consolidates a map body that mixes control flow with siblings into a
single NestedSDFG (sequencing the siblings), so downstream ``ConditionFusion``
can fold same-condition guards that MapFusion left in separate nested SDFGs.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody

N = dace.symbol('N')


@dace.program
def _two_guarded_loops_idx(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        if i % 2 == 0:
            b[i] = a[i] + 1.0
    for i in range(N):
        if i % 2 == 0:
            c[i] = a[i] * 2.0


@dace.program
def _two_plain_loops(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        b[i] = a[i] + 1.0
    for i in range(N):
        c[i] = a[i] * 2.0


def _body_nsdfg_count(sdfg: dace.SDFG) -> int:
    """Max number of NestedSDFG nodes inside any single top-level map body."""
    counts = [0]
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None:
                counts.append(
                    sum(1 for x in st.all_nodes_between(n, st.exit_node(n)) if isinstance(x, nodes.NestedSDFG)))
    return max(counts)


def _num_condblocks(sdfg: dace.SDFG) -> int:
    return sum(1 for cb in sdfg.all_control_flow_regions(recursive=True) if isinstance(cb, ConditionalBlock))


def test_merges_two_sibling_nsdfgs_into_one():
    """A fused map body with two guarded nested SDFGs -> one nested SDFG whose
    CFG sequences both; ConditionFusion then folds the two guards; valid + exact."""
    # Canonicalize up to (but not through) the map-body normalization by using
    # the standalone pass on the already-fused two-guarded-map form.
    sdfg = _two_guarded_loops_idx.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    # The pipeline now already normalizes+merges, so the end state has one guard;
    # re-running the pass must be a safe no-op, and the merged form must be valid.
    assert _num_condblocks(sdfg) == 1, "canon must leave a single merged in-map guard"
    assert _body_nsdfg_count(sdfg) <= 1, "the map body must be a single nested SDFG"
    assert NormalizeMapBody().apply_pass(sdfg, {}) is None, "re-run must be a no-op on the normalized form"
    sdfg.validate()

    n = 16
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b, c = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=b, c=c, N=n)
    even = np.arange(n) % 2 == 0
    assert np.allclose(b, np.where(even, a + 1.0, 0.0))
    assert np.allclose(c, np.where(even, a * 2.0, 0.0))


def test_all_tasklet_body_untouched():
    """A map body with no nested SDFG (all tasklets) is left unchanged."""
    sdfg = _two_plain_loops.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    before = _body_nsdfg_count(sdfg)
    assert NormalizeMapBody().apply_pass(sdfg, {}) is None, "no control flow -> no-op"
    assert _body_nsdfg_count(sdfg) == before
    sdfg.validate()

    n = 16
    rng = np.random.default_rng(1)
    a = rng.random(n)
    b, c = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=b, c=c, N=n)
    assert np.allclose(b, a + 1.0)
    assert np.allclose(c, a * 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
