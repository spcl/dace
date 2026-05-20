# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contract tests for the standalone ``FuseOverlappingLoads`` pass.

The existing ``test_jacobi*_with_parameters`` tests exercise the
``insert_copies=False, fuse_overlapping_loads=True`` standalone path on
jacobi1d/2d/heat3d (overlapping-stencil kernels), but a number of
*contract* cases aren't covered there:

- **Inert on a disjoint-read kernel** — a kernel with no overlapping
  reads of the same array must not fuse anything (no-op contract).
- **Idempotent** — re-running the pass on an already-fused SDFG is a
  no-op (no second fusion / no spurious window).
- **Multi-array kernel runs correctly** — two distinct arrays with
  their own stencils in the same map produce the right output under
  fusion + ``insert_copies=False`` (numeric equivalence only — the
  per-array structural shape is implementation-defined and not asserted
  here, to avoid hyperfit).

Each test runs an existing-style ``@dace.program`` through the
vectorizer with ``insert_copies=False, fuse_overlapping_loads=True``;
``run_vectorization_test`` itself enforces bit-exact numeric equivalence
against the unvectorized reference, so each contract test only needs
to add the structural / state assertion specific to its case (or, for
the multi-array case, simply rely on the harness's bit-exact check).
"""
import numpy as np
import pytest

import dace
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    S,
)


@dace.program
def _disjoint_unit_stride(A: dace.float64[S], B: dace.float64[S]):
    """No overlapping reads — fusion must be inert."""
    for i in dace.map[0:S]:
        B[i] = A[i] * 2.0


@dace.program
def _multi_array_stencils(A: dace.float64[S, S], B: dace.float64[S, S], C: dace.float64[S, S]):
    """Two distinct arrays, each with its own 3-point j-stencil — used
    for the multi-array correctness contract (numeric only)."""
    for i, j in dace.map[0:S, 0:S - 2]:
        C[i, j + 1] = (A[i, j] + A[i, j + 1] + A[i, j + 2]) + (B[i, j] + B[i, j + 1] + B[i, j + 2])


def test_fuse_inert_on_disjoint_reads():
    """Disjoint per-lane reads: pass leaves the SDFG with no widened
    union window (nothing to fuse). Numeric output equals untiled."""
    _S = 64
    A = np.random.random(_S)
    B = np.random.random(_S)
    vectorized_sdfg = run_vectorization_test(
        dace_func=_disjoint_unit_stride,
        arrays={
            "A": A,
            "B": B
        },
        params={"S": _S},
        vector_width=8,
        sdfg_name="fuse_inert_disjoint",
        fuse_overlapping_loads=True,
        insert_copies=False,
    )
    # No ``A_vec`` widened window should exist (no overlap to fuse).
    for nsdfg in (n for n, _ in vectorized_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)):
        for arr_name in nsdfg.sdfg.arrays:
            if arr_name in ("A_vec", "B_vec"):
                shape = tuple(
                    int(s) if not hasattr(s, "free_symbols") or not s.free_symbols else -1
                    for s in nsdfg.sdfg.arrays[arr_name].shape)
                # Allowed: a plain ``_vec`` array of exactly W elements
                # (an ordinary vector load buffer). Forbidden: a wider
                # union window (would indicate spurious fusion).
                assert -1 in shape or max(shape) <= 8, (
                    f"unexpected widened union window {arr_name} shape={shape} in disjoint-read kernel")


def test_fuse_multi_array_kernel_e2e():
    """Two distinct arrays each with their own 3-point stencil in the
    same map: e2e correctness only. The harness asserts bit-exact
    equality against the unvectorized reference — the per-array
    structural shape is implementation-defined (not claimed here)."""
    _S = 64
    run_vectorization_test(
        dace_func=_multi_array_stencils,
        arrays={
            "A": np.random.random((_S, _S)),
            "B": np.random.random((_S, _S)),
            "C": np.zeros((_S, _S)),
        },
        params={"S": _S},
        vector_width=8,
        sdfg_name="fuse_multi_array_e2e",
        fuse_overlapping_loads=True,
        insert_copies=False,
    )


def test_fuse_threshold_property_round_trip():
    """``FuseOverlappingLoads`` exposes a ``fusion_threshold`` Property
    that gates ``len(v) <= threshold`` in ``_apply`` — default ``1``
    preserves the prior "fuse iff >= 2 overlapping reads" behaviour.
    A full integration test that observes the discriminating effect on
    a hand-built SDFG is implementation-fragile (the pass refactors
    topology in ways that don't map to a simple count metric) and is
    deferred. This sanity-checks the API surface — the Property
    accepts an int, default is 1, and the constructor wires it."""
    from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads
    p_default = FuseOverlappingLoads()
    assert p_default.fusion_threshold == 1
    p_high = FuseOverlappingLoads(fusion_threshold=5)
    assert p_high.fusion_threshold == 5


def test_fuse_idempotent_on_jacobi1d():
    """Re-applying ``FuseOverlappingLoads`` after vectorization is a
    no-op: the union window already exists; the pass must not double-
    fuse or invalidate the SDFG."""
    from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads

    _S = 130

    @dace.program
    def _jacobi1d_local(A: dace.float64[S], B: dace.float64[S], tsteps: dace.int64):
        for t in range(tsteps):
            for i in dace.map[0:S - 2]:
                B[i + 1] = 0.33333 * (A[i] + A[i + 1] + A[i + 2])
            for i in dace.map[0:S - 2]:
                A[i + 1] = 0.33333 * (B[i] + B[i + 1] + B[i + 2])

    A = np.random.random(_S)
    B = np.random.random(_S)
    vectorized_sdfg = run_vectorization_test(
        dace_func=_jacobi1d_local,
        arrays={
            "A": A,
            "B": B
        },
        params={
            "S": _S,
            "tsteps": 5
        },
        vector_width=8,
        sdfg_name="fuse_idempotent_jacobi1d",
        fuse_overlapping_loads=True,
        insert_copies=False,
    )
    # First fusion happened inside ``run_vectorization_test``. Now apply
    # the standalone pass a second time on the post-vectorization SDFG.
    snapshot_arrays = {
        nsdfg.sdfg.label: sorted(nsdfg.sdfg.arrays.keys())
        for nsdfg in (n for n, _ in vectorized_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    }
    FuseOverlappingLoads().apply_pass(vectorized_sdfg, {})
    vectorized_sdfg.validate()
    after_arrays = {
        nsdfg.sdfg.label: sorted(nsdfg.sdfg.arrays.keys())
        for nsdfg in (n for n, _ in vectorized_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    }
    assert snapshot_arrays == after_arrays, (
        f"FuseOverlappingLoads is not idempotent: arrays changed on second apply.\n"
        f"before={snapshot_arrays}\n after={after_arrays}")
