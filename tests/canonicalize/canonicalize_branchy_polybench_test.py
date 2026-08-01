# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize behaviour on branchy npbench/polybench-derived kernels.

The pure-stencil kernels (``jacobi_2d``, ``heat_3d``, ``seidel_2d``,
``fdtd_2d``, ``gemm`` ...) are slice-based and therefore branch-free; the
branchy polybench kernels are concentrated in:

* **nussinov** -- boundary guards ``if j - 1 >= 0`` (loop-variant) and
  ``if i + 1 < N`` (invariant over the inner ``j`` loop -- a textbook
  ``MoveLoopInvariantIfUp`` candidate);
* **floyd_warshall** -- ``if path[i, k] + path[k, j] < path[i, j]: ...``
  guard inside ``k, i, j``: the guard is data-dependent, ``k`` is sequential,
  and per step only ``i`` is parallel (after an index-set split at ``i == k``)
  -- ``j`` carries a dependence through ``path[i, k]``;
* **correlation** -- masked write ``stddev[stddev <= 0.1] = 1.0``,
  expressed scalarly as ``if a[i] <= thr: a[i] = 1.0`` (elementwise
  data-dependent guard);
* **gramschmidt** -- safety guard around a division: ``if nrm > 0:
  q[:, k] = a[:, k] / nrm``.

Each test pins a value-preservation contract against a pure-numpy oracle.
A few also lock structural expectations that canonicalize is known to
deliver today (e.g. the inner-``j``-loop-invariant guard in the nussinov
boundary shape must be hoisted above the inner map / be the guard around
a single fused map, never replicated per j-iteration). When a structural
expectation depends on an unimplemented pass (e.g. the deferred
``CascadeInterstateEdgeAssignmentsUp`` for outer-loop iedge assignments),
the test is marked ``strict=True`` xfail with a precise reason linking to
the design doc.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _ncond_blocks(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock))


# ----------------------------------------------------------------------
# Nussinov boundary guards
# ----------------------------------------------------------------------


@dace.program
def nussinov_boundary_guards(table: dace.int32[N, N]):
    """Reduced nussinov core: nested ``i, j`` walk with two boundary
    guards. ``if i + 1 < N`` is invariant over the inner ``j`` loop and
    is the canonical hoist target; ``if j - 1 >= 0`` reads ``j`` and
    must stay inside.
    """
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                if table[i, j - 1] > table[i, j]:
                    table[i, j] = table[i, j - 1]
            if i + 1 < N:
                if table[i + 1, j] > table[i, j]:
                    table[i, j] = table[i + 1, j]


def _nussinov_oracle(table):
    n = table.shape[0]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if j - 1 >= 0 and table[i, j - 1] > table[i, j]:
                table[i, j] = table[i, j - 1]
            if i + 1 < n and table[i + 1, j] > table[i, j]:
                table[i, j] = table[i + 1, j]
    return table


def test_nussinov_boundary_guards_value_preserving():
    n = 12
    rng = np.random.default_rng(0)
    base = rng.integers(0, 9, size=(n, n)).astype(np.int32)
    exp = _nussinov_oracle(base.copy())
    sdfg = nussinov_boundary_guards.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = base.copy()
    sdfg(table=got, N=n)
    assert np.array_equal(got, exp)


# ----------------------------------------------------------------------
# Floyd-Warshall: data-dependent guarded update
# ----------------------------------------------------------------------


@dace.program
def floyd_warshall_step(path: dace.float64[N, N], k: dace.int32):
    """One outer-``k`` step of Floyd-Warshall. Both ``i`` and ``j`` conflict on
    row/column ``k``; splitting the index set at ``i == k`` makes ``i`` parallel,
    while ``j`` stays order-carried for an arbitrary ``path`` (see
    ``test_floyd_warshall_step_i_parallel_via_index_split``).
    """
    for i in range(N):
        for j in range(N):
            if path[i, k] + path[k, j] < path[i, j]:
                path[i, j] = path[i, k] + path[k, j]


def _fw_step_oracle(path, k):
    n = path.shape[0]
    for i in range(n):
        for j in range(n):
            if path[i, k] + path[k, j] < path[i, j]:
                path[i, j] = path[i, k] + path[k, j]
    return path


def test_floyd_warshall_step_value_preserving():
    n = 9
    rng = np.random.default_rng(1)
    base = rng.uniform(0.5, 3.0, size=(n, n)).astype(np.float64)
    np.fill_diagonal(base, 0.0)
    for k in range(n):
        exp = _fw_step_oracle(base.copy(), k)
        sdfg = floyd_warshall_step.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        got = base.copy()
        sdfg(path=got, k=np.int32(k), N=n)
        assert np.allclose(got, exp), f'fw step k={k} mismatch'
        base = exp


def test_floyd_warshall_step_nonzero_diagonal_stays_ordered():
    """The counterexample that makes ``j``-parallelism observable: a non-zero
    ``path[k, k]`` (which the value test above rules out by construction). A
    ``j``-parallel step would read the pre-step ``path[0, 1] = 5`` and store
    ``path[0, 2] = 7``; the loop as written must store ``4``.
    """
    base = np.array([[0.0, 5.0, 9.0], [7.0, -3.0, 2.0], [4.0, 6.0, 0.0]])
    exp = _fw_step_oracle(base.copy(), 1)
    assert exp[0, 2] == 4.0
    sdfg = floyd_warshall_step.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = base.copy()
    sdfg(path=got, k=np.int32(1), N=3)
    assert np.allclose(got, exp)


def test_floyd_warshall_step_i_parallel_via_index_split():
    """Structural check: ``i`` is parallelized by an index-set split at ``i == k``
    into ``[0, k)``, ``{k}``, ``(k, N)`` -- three 1D maps -- and ``j`` stays
    sequential, which is the strongest SOUND result for this kernel.

    ``j`` is NOT parallel: iteration ``j == k`` writes ``path[i, k]``, which every
    other ``j`` reads, so the loop is order-carried. With ``k = 1`` and
    ``path = [[0, 5, 9], [7, -3, 2], [4, 6, 0]]`` the loop as written yields
    ``path[0, 2] = 4`` (it reuses the ``path[0, 1] = 2`` written at ``j == 1``)
    while any ``j``-parallel form yields ``7``. Real Floyd-Warshall gets away with
    the parallel form only because ``path[k, k] == 0`` makes the ``i == k`` /
    ``j == k`` updates no-ops; ``path`` here is an arbitrary input, so canonicalize
    must not assume it. ``i`` has the same conflict on row ``k`` and that is
    exactly what the split at ``k`` removes.
    """
    sdfg = floyd_warshall_step.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(maps) == 3, f'expected the three i-segments of the split at k to map, got {len(maps)}'
    assert all(len(m.map.params) == 1 for m in maps), 'each segment maps the i dimension alone'


# ----------------------------------------------------------------------
# Correlation-style masked write
# ----------------------------------------------------------------------


@dace.program
def masked_threshold_write(a: dace.float64[N], thr: dace.float64):
    """Scalar form of ``stddev[stddev <= 0.1] = 1.0`` from
    ``correlation``. Every iteration is independent; the data-dependent
    guard must not block map-ification.
    """
    for i in dace.map[0:N]:
        if a[i] <= thr:
            a[i] = 1.0


def _masked_oracle(a, thr):
    out = a.copy()
    out[out <= thr] = 1.0
    return out


def test_masked_threshold_write_value_preserving():
    n = 16
    rng = np.random.default_rng(2)
    a = rng.uniform(-0.5, 1.5, size=n).astype(np.float64)
    exp = _masked_oracle(a, 0.1)
    sdfg = masked_threshold_write.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    got = a.copy()
    sdfg(a=got, thr=np.float64(0.1), N=n)
    assert np.allclose(got, exp)


def test_masked_threshold_write_stays_one_map():
    """The map is already explicit (``dace.map[0:N]``); canonicalize must
    not split it. Independent elementwise writes are the cheapest
    parallel shape we have and any split would be a regression."""
    sdfg = masked_threshold_write.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) == 1, f'masked elementwise write split to {_nmaps(sdfg)} maps'


# ----------------------------------------------------------------------
# Loop-invariant boundary guard (synthetic nussinov-like distilled case)
# ----------------------------------------------------------------------


@dace.program
def loop_invariant_guard_over_inner(a: dace.float64[N, N], b: dace.float64[N, N], lim: dace.int32):
    """``if lim < N`` is invariant on the inner ``j`` loop. A correct
    pipeline hoists the guard out of the inner loop (either above
    ``map j`` or above the entire ``i, j`` nest), exposing two clean
    parallel maps with no per-iteration conditional dispatch.
    """
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            if lim < N:
                b[i, j] = a[i, j] * 2.0
            else:
                b[i, j] = a[i, j]


def test_loop_invariant_guard_over_inner_value_preserving():
    n, lim_lo, lim_hi = 8, 5, 99
    rng = np.random.default_rng(3)
    a = rng.standard_normal((n, n)).astype(np.float64)

    for lim in (lim_lo, lim_hi):
        sdfg = loop_invariant_guard_over_inner.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        got = np.zeros_like(a)
        sdfg(a=a, b=got, lim=np.int32(lim), N=n)
        exp = a * (2.0 if lim < n else 1.0)
        assert np.allclose(got, exp), f'lim={lim}'


def test_loop_invariant_guard_over_inner_hoisted_to_top():
    """The invariant ``if lim < N`` guard (condition reads only outer-SDFG
    symbols ``lim``, ``N``, never the map parameters) is hoisted out of the
    fully-parallel ``i, j`` nest to the SDFG top level. The nest collapses to a
    single ``map[i, j]`` carrying the guard in its body; ``MoveMapInvariantIfUp``
    then lifts the guard out, one map copy per branch -- leaving exactly one
    top-level ``ConditionalBlock`` whose branches are clean unconditional
    parallel maps."""
    sdfg = loop_invariant_guard_over_inner.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    top_conds = [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]
    assert len(top_conds) == 1, ('expected the (lim < N) guard hoisted to the SDFG '
                                 f'top level; got {_ncond_blocks(sdfg)} conditional '
                                 f'block(s) total, {len(top_conds)} at top level')


# ----------------------------------------------------------------------
# Gramschmidt-style safety guard around a division
# ----------------------------------------------------------------------


@dace.program
def safety_guarded_divide(a: dace.float64[N], out: dace.float64[N], denom: dace.float64):
    """Polybench gramschmidt's ``if nrm > 0`` shape -- a single outer-scope
    guard around a parallel inner write. The guard is invariant on the
    inner loop; the cleanest canonical shape is ``if denom > 0: map i:
    out[i] = a[i] / denom``.
    """
    if denom > 0.0:
        for i in dace.map[0:N]:
            out[i] = a[i] / denom
    else:
        for i in dace.map[0:N]:
            out[i] = 0.0


def test_safety_guarded_divide_value_preserving():
    n = 12
    rng = np.random.default_rng(4)
    a = rng.standard_normal(n).astype(np.float64)
    for denom in (1.7, -0.4):
        sdfg = safety_guarded_divide.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        out = np.zeros(n)
        sdfg(a=a, out=out, denom=np.float64(denom), N=n)
        exp = (a / denom) if denom > 0 else np.zeros(n)
        assert np.allclose(out, exp), f'denom={denom}'


def test_safety_guarded_divide_keeps_outer_guard_at_top():
    """The top-level guard is already in the right place; canonicalize
    must not push it inside either map (which would re-introduce a
    per-iteration conditional dispatch).
    """
    sdfg = safety_guarded_divide.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert any(isinstance(c, ConditionalBlock) for c in sdfg.nodes()), \
        'top-level safety guard was pushed inside the map nest'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
