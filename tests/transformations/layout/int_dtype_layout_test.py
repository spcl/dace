# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Layout transforms over INTEGER element types, written as an EXPLICIT elementwise map.

An integer elementwise kernel written as an explicit map (``for i,j,k: C[i,j,k]=A[i,j,k]*B[i,j,k]``)
is wrapped by the frontend in a nested *scalar-computation* SDFG (int elements are symbol candidates,
so the body is lowered through a scalar sub-SDFG whose value is bound into an interstate assignment
and then inlined into the tasklet code). ``prepare_for_layout`` runs ``ExpandNestedSDFGInputs``,
which widens those scalar connectors to whole-array descriptors; the tasklet's inlined bare read
``__out = tmp * ...`` then has no valid array form (pointer arithmetic on ``const int64_t*``).

``ExpandNestedSDFGInputs`` now handles that case: a bare symbolic read of a widened scalar connector
is converted to a dataflow read -- an input connector fed by the single element ``A[offset]`` -- so
the explicit-map integer kernel lowers correctly. These tests exercise that fix: int64 / int32
elementwise, under Permute, Block, and combined Block+Permute, all bit-exact (integer array_equal)."""
import itertools

import numpy
import pytest
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N", dtype=dace.int64)


def make_mul(dtype):

    @dace.program
    def mul3d(A: dtype[N, N, N], B: dtype[N, N, N], C: dtype[N, N, N]):
        for i, j, k in dace.map[0:N, 0:N, 0:N]:  # explicit map -> scalar-wrapper nested SDFG (the fixed path)
            C[i, j, k] = A[i, j, k] * B[i, j, k]

    return mul3d


PROGRAMS = {numpy.int64: make_mul(dace.int64), numpy.int32: make_mul(dace.int32)}


def _inputs(npdt, n, seed):
    rng = numpy.random.default_rng(seed)
    A = rng.integers(-5, 6, size=(n, ) * 3).astype(npdt)
    B = rng.integers(-5, 6, size=(n, ) * 3).astype(npdt)
    return A, B, (A * B)


def _block_pack(a, blocked_dim, factor):
    """Lay ``a`` out into SplitDimensions' physical shape: split ``blocked_dim`` into
    ``(size/factor, factor)`` and move the inner tile axis to the END (append-last convention)."""
    expanded = []
    for d in range(a.ndim):
        if d == blocked_dim:
            expanded += [a.shape[d] // factor, factor]
        else:
            expanded.append(a.shape[d])
    inner = blocked_dim + 1
    order = [ax for ax in range(a.ndim + 1) if ax != inner] + [inner]
    return a.reshape(expanded).transpose(order).copy()


@pytest.mark.parametrize("npdt", [numpy.int64, numpy.int32])
@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_int_permute(npdt, perm):
    """Every dim permutation of an integer elementwise kernel is bit-exact (transparent permute)."""
    n = 4
    A, B, ref = _inputs(npdt, n, sum(perm))
    sdfg = PROGRAMS[npdt].to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})
    C = numpy.zeros((n, ) * 3, dtype=npdt)
    sdfg(A=A.copy(), B=B.copy(), C=C, N=n)
    assert numpy.array_equal(C, ref)


@pytest.mark.parametrize("npdt", [numpy.int64, numpy.int32])
@pytest.mark.parametrize("blocked_dim,factor", [(0, 2), (1, 2), (2, 2), (0, 4), (2, 4)])
def test_int_block(npdt, blocked_dim, factor):
    """Blocking any dimension of an integer elementwise kernel is bit-exact."""
    n = 8
    A, B, ref = _inputs(npdt, n, blocked_dim * 10 + factor)
    sdfg = PROGRAMS[npdt].to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    masks = [d == blocked_dim for d in range(3)]
    factors = [factor if d == blocked_dim else 1 for d in range(3)]
    SplitDimensions(split_map={"A": (masks, factors)}).apply_pass(sdfg, {})
    normalize_schedule_for_layout(sdfg)
    shp = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["A"].shape)
    A_packed = _block_pack(A, blocked_dim, factor)
    assert A_packed.shape == shp
    C = numpy.zeros((n, ) * 3, dtype=npdt)
    sdfg(A=A_packed, B=B.copy(), C=C, N=n)
    assert numpy.array_equal(C, ref)


@pytest.mark.parametrize("npdt", [numpy.int64, numpy.int32])
def test_int_block_then_permute(npdt):
    """Block dim 0 of the input then transparently permute it: still bit-exact."""
    n = 8
    A, B, ref = _inputs(npdt, n, 7)
    sdfg = PROGRAMS[npdt].to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    SplitDimensions(split_map={"A": ([True, False, False], [2, 1, 1])}).apply_pass(sdfg, {})
    normalize_schedule_for_layout(sdfg)
    PermuteDimensions(permute_map={"A": [3, 0, 2, 1]}, add_permute_maps=True).apply_pass(sdfg, {})
    # external interface stays the block-packed shape (permute is transparent).
    shp = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["A"].shape)
    A_packed = _block_pack(A, 0, 2)
    assert A_packed.shape == shp
    C = numpy.zeros((n, ) * 3, dtype=npdt)
    sdfg(A=A_packed, B=B.copy(), C=C, N=n)
    assert numpy.array_equal(C, ref)


if __name__ == "__main__":
    for dt in (numpy.int64, numpy.int32):
        for p in itertools.permutations(range(3)):
            test_int_permute(dt, p)
        for bd, fac in [(0, 2), (1, 2), (2, 2), (0, 4), (2, 4)]:
            test_int_block(dt, bd, fac)
        test_int_block_then_permute(dt)
    print("int dtype layout tests PASS")
