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
from tests.passes.vectorization.helpers.harness import (
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
    Sanity-checks the API surface: Property accepts an int, default
    is 1, constructor wires it, instrumentation counters initialize."""
    from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads
    p_default = FuseOverlappingLoads()
    assert p_default.fusion_threshold == 1
    assert p_default._last_groups_gated == 0
    assert p_default._last_groups_fused == 0
    p_high = FuseOverlappingLoads(fusion_threshold=5)
    assert p_high.fusion_threshold == 5


def _build_fuse_input_sdfg(num_inner: int):
    """Hand-build the exact shape ``FuseOverlappingLoads._apply``
    scans for: an inner MapEntry whose scope parent is also a MapEntry,
    with ``num_inner`` distinct AccessNodes of the same array fanning
    into it (each a single source-to-inner load)."""
    sdfg = dace.SDFG(f"fuse_input_{num_inner}")
    sdfg.add_array("A", [64], dace.float64)
    sdfg.add_array("OUT", [32], dace.float64)
    st = sdfg.add_state(is_start_block=True)
    outer_src = st.add_access("A")
    outer_dst = st.add_access("OUT")
    ome, omx = st.add_map("outer", {"o": "0:1"})
    ime, imx = st.add_map("body", {"i": "0:32"})
    t = st.add_tasklet("nop", {f"_in{k}"
                               for k in range(num_inner)}, {"_out"},
                       "_out = " + " + ".join(f"_in{k}" for k in range(num_inner)))
    inner_a_root = st.add_access("A")
    st.add_memlet_path(outer_src, ome, inner_a_root, memlet=dace.Memlet("A[0:64]"))
    for k in range(num_inner):
        an = st.add_access("A")
        st.add_edge(inner_a_root, None, an, None, dace.Memlet("A[0:64]"))
        st.add_memlet_path(an, ime, t, dst_conn=f"_in{k}", memlet=dace.Memlet(f"A[i + {k}]"))
    st.add_memlet_path(t, imx, omx, outer_dst, src_conn="_out", memlet=dace.Memlet("OUT[i]"))
    return sdfg


@pytest.mark.parametrize(
    "num_inner,threshold,expect_fused,expect_gated",
    [
        # Strictly-above-threshold groups are fused.
        (3, 1, 1, 0),
        (3, 2, 1, 0),
        (5, 4, 1, 0),
        # At-or-below the per-group count: gated, not fused.
        (3, 3, 0, 1),
        (3, 5, 0, 1),
        (5, 5, 0, 1),
    ])
def test_fuse_threshold_gates_via_instrumentation(num_inner, threshold, expect_fused, expect_gated):
    """``FuseOverlappingLoads`` records in ``_last_groups_fused`` /
    ``_last_groups_gated`` how its ``len(v) <= threshold`` gate
    decided. Hand-built input matches the pass's exact predicate
    (nested MapEntry + ``num_inner`` distinct fan-in AccessNodes of A)
    so the gate sees exactly one group of size ``num_inner``. The
    counters expose the decision directly — non-fragile, no post-hoc
    topology probe."""
    from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads
    sdfg = _build_fuse_input_sdfg(num_inner)
    p = FuseOverlappingLoads(fusion_threshold=threshold)
    p.apply_pass(sdfg, {})
    assert p._last_groups_fused == expect_fused, \
        (f"num_inner={num_inner} threshold={threshold}: fused={p._last_groups_fused} "
         f"expected {expect_fused}")
    assert p._last_groups_gated == expect_gated, \
        (f"num_inner={num_inner} threshold={threshold}: gated={p._last_groups_gated} "
         f"expected {expect_gated}")


def test_fuse_idempotent_on_jacobi1d():
    """Re-applying ``FuseOverlappingLoads`` after vectorization is a
    no-op: the union window already exists; the pass must not double-
    fuse or invalidate the SDFG."""
    from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads
    # Reuse the canonical jacobi1d kernel (dedup) instead of a local copy.
    from tests.passes.vectorization.kernels.test_jacobi import jacobi1d

    _S = 130

    A = np.random.random(_S)
    B = np.random.random(_S)
    vectorized_sdfg = run_vectorization_test(
        dace_func=jacobi1d,
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
