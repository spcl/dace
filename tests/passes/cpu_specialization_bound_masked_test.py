# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map behind ``slot < count`` stops after one masked slot only when masked slots have no further effect."""

import ast

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.bound_masked_maps import (
    ABSORBED,
    SAME,
    index_polynomial,
    masked_value,
)

N = dace.symbol("N", dtype=dace.int64, positive=True)
M = dace.symbol("M", dtype=dace.int64, positive=True)


@dace.program
def absorbed_force(x: dace.float64[N, 3], counts: dace.int32[N], neigh: dace.int32[N, M], f: dace.float64[N, 3]):
    slot = np.arange(M)
    valid = slot[None, :] < counts[:, None]
    j = np.where(valid, neigh, 0)
    d = x[:, None, :] - x[j]
    rsq = np.sum(d * d, axis=2)
    within = valid & (rsq < 4.0)
    r2inv = 1.0 / np.where(within, rsq, 1.0)
    fpair = np.where(within, r2inv * r2inv, 0.0)
    f[:] = np.sum(fpair[:, :, None] * d, axis=1)


@dace.program
def exact_sum(x: dace.float64[N, M], counts: dace.int32[N], f: dace.float64[N]):
    slot = np.arange(M)
    valid = slot[None, :] < counts[:, None]
    f[:] = np.sum(np.where(valid, x * x, 0.0), axis=1)


@dace.program
def masked_slot_has_effect(x: dace.float64[N, M], counts: dace.int32[N], f: dace.float64[N]):
    slot = np.arange(M)
    valid = slot[None, :] < counts[:, None]
    f[:] = np.sum(np.where(valid, x, 1.0), axis=1)


@dace.program
def masked_factor_varies(x: dace.float64[N, M], counts: dace.int32[N], f: dace.float64[N]):
    slot = np.arange(M)
    valid = slot[None, :] < counts[:, None]
    f[:] = np.sum(np.where(valid, 1.0, 0.0) * x, axis=1)


def specialized(program, name):
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = name
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu", validate=True)
    return sdfg


def bounded(sdfg):
    return [
        node
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and any(not c.startswith("IN_") for c in node.in_connectors)
    ]


@dace.program
def two_masks(x: dace.float64[N, M], a: dace.int32[N], b: dace.int32[N], f: dace.float64[N]):
    slot = np.arange(M)
    f[:] = np.sum(np.where((slot[None, :] < a[:, None]) & (slot[None, :] < b[:, None]), x * x, 0.0), axis=1)


def expected_force(x, counts, neigh):
    out = np.zeros_like(x)
    m = neigh.shape[1]
    for i in range(x.shape[0]):
        for s in range(m):
            valid = s < counts[i]
            d = x[i] - x[neigh[i, s] if valid else 0]
            rsq = np.sum(d * d)
            within = valid and rsq < 4.0
            r2inv = 1.0 / (rsq if within else 1.0)
            out[i] += (r2inv * r2inv if within else 0.0) * d
    return out


def test_absorbed_keeps_one_masked_slot():
    sdfg = specialized(absorbed_force, "bound_masked_absorbed")
    assert len(bounded(sdfg)) == 1
    rng = np.random.default_rng(3)
    n, m = 7, 5
    x = rng.random((n, 3))
    # Out of range on both sides, none live, all live
    counts = np.array([0, 1, 3, 5, 9, -2, 4], dtype=np.int32)
    neigh = rng.integers(0, n, (n, m), dtype=np.int32)
    for poison in (None, np.nan, np.inf):
        y = x.copy()
        if poison is not None:
            y[0, 1] = poison  # What every masked slot reads, as a non-finite position
        f = np.zeros_like(y)
        sdfg(x=y, counts=counts, neigh=neigh, f=f, N=n, M=m)
        np.testing.assert_array_equal(np.isnan(f), np.isnan(expected_force(y, counts, neigh)))
        np.testing.assert_allclose(f, expected_force(y, counts, neigh))


def test_exact_zero_keeps_one_masked_slot():
    sdfg = specialized(exact_sum, "bound_masked_exact")
    assert len(bounded(sdfg)) == 1
    n, m = 4, 3
    x = np.arange(1.0, 13.0).reshape(n, m).copy()
    x[1, 2] = np.nan  # Masked out: a select never reads it
    counts = np.array([0, 2, 7, -1], dtype=np.int32)
    f = np.zeros(n)
    sdfg(x=x, counts=counts, f=f, N=n, M=m)
    np.testing.assert_allclose(f, [0.0, 16.0 + 25.0, 49.0 + 64.0 + 81.0, 0.0])


def test_masked_constant_is_not_neutral():
    sdfg = specialized(masked_slot_has_effect, "bound_masked_effect")
    assert not bounded(sdfg)
    x = np.arange(1.0, 7.0).reshape(2, 3).copy()
    f = np.zeros(2)
    sdfg(x=x, counts=np.array([1, 2], dtype=np.int32), f=f, N=2, M=3)
    np.testing.assert_allclose(f, [1.0 + 2.0, 4.0 + 5.0 + 1.0])


def test_masked_factor_reading_the_slot_refuses():
    sdfg = specialized(masked_factor_varies, "bound_masked_varies")
    assert not bounded(sdfg)
    x = np.array([[1.0, 2.0, np.inf]])
    f = np.zeros(1)
    sdfg(x=x, counts=np.array([1], dtype=np.int32), f=f, N=1, M=3)
    assert np.isnan(f[0])


def test_two_masks_bound_the_map_once():
    sdfg = specialized(two_masks, "bound_masked_two")
    assert [sum(not c.startswith("IN_") for c in entry.in_connectors) for entry in bounded(sdfg)] == [1]
    x = np.arange(1.0, 7.0).reshape(2, 3).copy()
    f = np.zeros(2)
    sdfg(x=x, a=np.array([1, 3], dtype=np.int32), b=np.array([3, 2], dtype=np.int32), f=f, N=2, M=3)
    np.testing.assert_allclose(f, [1.0, 16.0 + 25.0])


def test_absorbed_test_picks_no_branch():
    """``0.0 * v`` as a condition may be NaN: neither branch is known."""
    code = ast.parse("0.0 if t else 1.0", mode="eval").body
    assert masked_value(code, "s", {"t": ABSORBED}, None) == SAME


def test_fill_that_is_no_polynomial_is_no_index():
    assert index_polynomial(ast.parse("int64_t(i) // 2", mode="eval").body, "i") is None


if __name__ == "__main__":
    test_absorbed_keeps_one_masked_slot()
    test_exact_zero_keeps_one_masked_slot()
    test_masked_constant_is_not_neutral()
    test_masked_factor_reading_the_slot_refuses()
    test_two_masks_bound_the_map_once()
    test_absorbed_test_picks_no_branch()
    test_fill_that_is_no_polynomial_is_no_index()
