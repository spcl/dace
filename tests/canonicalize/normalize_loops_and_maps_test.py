# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``NormalizeLoopsAndMaps``: every map range becomes ``0:trip:1``
    while the result stays identical. Covers offset, non-unit, negative and
    symbolic steps, and a mixed multi-dimensional map. Each test compares the
    SDFG end-to-end against a deep-copied pre-pass reference run.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps

N, S = dace.symbol('N'), dace.symbol('S')


@dace.program
def offset_stride(A: dace.float64[40], B: dace.float64[40]):
    for i in dace.map[3:31:4]:
        B[i] = A[i] * 2.0 + 1.0


@dace.program
def negative_step(A: dace.float64[40], B: dace.float64[40]):
    for i in dace.map[20:1:-2]:
        B[i] = A[i] - 3.0


@dace.program
def symbolic_step(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N:S]:
        B[i] = A[i] + 5.0


@dace.program
def mixed_2d(A: dace.float64[30, 30], B: dace.float64[30, 30]):
    for i, j in dace.map[2:27:3, 5:25:2]:
        B[i, j] = A[i, j] * 4.0


def _map_ranges(sdfg: dace.SDFG):
    out = []
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            out.extend((b, e, s) for b, e, s in n.map.range.ranges)
    return out


def _assert_canonical(sdfg: dace.SDFG):
    for b, _e, s in _map_ranges(sdfg):
        assert b == 0, f"map start not 0: {b}"
        assert s == 1, f"map step not 1: {s}"


def _run(program, simplify_inputs, **kw):
    sdfg = program.to_sdfg(simplify=False)
    ref = copy.deepcopy(sdfg)

    pre = {k: v.copy() for k, v in simplify_inputs.items()}
    ref(**pre, **kw)

    changed = NormalizeLoopsAndMaps().apply_pass(sdfg, {})
    assert changed is not None, "pass did not normalize a non-canonical map"
    sdfg.validate()
    _assert_canonical(sdfg)

    post = {k: v.copy() for k, v in simplify_inputs.items()}
    sdfg(**post, **kw)
    for k in simplify_inputs:
        assert np.allclose(post[k], pre[k]), f"mismatch on {k}"
    return pre


def test_offset_stride():
    a = np.random.rand(40)
    b = np.full(40, -1.0)
    pre = _run(offset_stride, dict(A=a, B=b))
    exp = np.full(40, -1.0)
    for i in range(3, 31, 4):
        exp[i] = a[i] * 2.0 + 1.0
    assert np.allclose(pre['B'], exp)


def test_negative_step():
    a = np.random.rand(40)
    b = np.full(40, -1.0)
    pre = _run(negative_step, dict(A=a, B=b))
    exp = np.full(40, -1.0)
    for i in range(20, 1, -2):
        exp[i] = a[i] - 3.0
    assert np.allclose(pre['B'], exp)


def test_symbolic_step():
    n, s = 64, 5
    a = np.random.rand(n)
    b = np.full(n, -1.0)
    pre = _run(symbolic_step, dict(A=a, B=b), N=n, S=s)
    exp = np.full(n, -1.0)
    for i in range(0, n, s):
        exp[i] = a[i] + 5.0
    assert np.allclose(pre['B'], exp)


def test_mixed_2d():
    a = np.random.rand(30, 30)
    b = np.full((30, 30), -1.0)
    pre = _run(mixed_2d, dict(A=a, B=b))
    exp = np.full((30, 30), -1.0)
    for i in range(2, 27, 3):
        for j in range(5, 25, 2):
            exp[i, j] = a[i, j] * 4.0
    assert np.allclose(pre['B'], exp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
