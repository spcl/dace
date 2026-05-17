# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the generalized ``MoveIfIntoMap`` transformation.

    A conditional that guards inner maps inside an outer map's body is pushed
    into each inner map, exposing the maps to fusion/collapse. The pass now
    accepts the common Python-frontend shape where inner ``dace.map`` bodies
    are plain ``Tasklet`` subgraphs (normalized into a ``NestedSDFG`` first),
    and still rejects hoisting a condition that depends on an outer-map
    parameter (unsound). All kernels use the dace Python frontend only.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap

N, M = dace.symbol('N'), dace.symbol('M')


@dace.program
def guarded_two(flag: dace.int32, A: dace.float64[N, M], B: dace.float64[N, M], C: dace.float64[N, M]):
    for j in dace.map[0:M]:
        if flag > 0:
            for i in dace.map[0:N]:
                B[i, j] = A[i, j] + 1.0
            for i in dace.map[0:N]:
                C[i, j] = A[i, j] * 2.0


@dace.program
def guarded_one(flag: dace.int32, A: dace.float64[N, M], B: dace.float64[N, M]):
    for j in dace.map[0:M]:
        if flag > 0:
            for i in dace.map[0:N]:
                B[i, j] = A[i, j] + 1.0


@dace.program
def guarded_by_outer_param(A: dace.float64[N, M], B: dace.float64[N, M]):
    for j in dace.map[0:M]:
        if j > 0:
            for i in dace.map[0:N]:
                B[i, j] = A[i, j] + 1.0


def _num_conditional_blocks(sdfg: dace.SDFG) -> int:
    return sum(1 for _ in sdfg.all_control_flow_blocks() if isinstance(_, ConditionalBlock))


def test_two_sibling_guarded_maps_pushed_in():
    n, m = 6, 5
    a = np.random.rand(n, m)
    sdfg = guarded_two.to_sdfg(simplify=True)

    def run(s, flag):
        b = np.zeros((n, m))
        c = np.zeros((n, m))
        s(flag=np.int32(flag), A=a.copy(), B=b, C=c, N=n, M=m)
        return b, c

    ref_on = run(copy.deepcopy(sdfg), 1)
    ref_off = run(copy.deepcopy(sdfg), 0)

    applied = sdfg.apply_transformations_repeated(MoveIfIntoMap)
    assert applied >= 1
    sdfg.validate()

    b_on, c_on = run(sdfg, 1)
    b_off, c_off = run(sdfg, 0)
    assert np.allclose(b_on, ref_on[0]) and np.allclose(c_on, ref_on[1])
    assert np.allclose(b_off, ref_off[0]) and np.allclose(c_off, ref_off[1])
    assert np.allclose(b_on, a + 1.0) and np.allclose(c_on, a * 2.0)
    assert np.allclose(b_off, 0.0) and np.allclose(c_off, 0.0)


def test_single_guarded_map_pushed_in():
    n, m = 7, 4
    a = np.random.rand(n, m)
    sdfg = guarded_one.to_sdfg(simplify=True)

    ref = np.zeros((n, m))
    copy.deepcopy(sdfg)(flag=np.int32(1), A=a.copy(), B=ref, N=n, M=m)

    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) >= 1
    sdfg.validate()

    out = np.zeros((n, m))
    sdfg(flag=np.int32(1), A=a.copy(), B=out, N=n, M=m)
    assert np.allclose(out, ref) and np.allclose(out, a + 1.0)


def test_condition_on_outer_param_is_rejected():
    """Hoisting a condition that varies with the outer map parameter would
    change semantics; the pass must refuse and leave the result correct."""
    n, m = 5, 6
    a = np.random.rand(n, m)
    sdfg = guarded_by_outer_param.to_sdfg(simplify=True)

    ref = np.zeros((n, m))
    copy.deepcopy(sdfg)(A=a.copy(), B=ref, N=n, M=m)

    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 0

    out = np.zeros((n, m))
    sdfg(A=a.copy(), B=out, N=n, M=m)
    assert np.allclose(out, ref)
    expected = a + 1.0
    expected[:, 0] = 0.0
    assert np.allclose(out, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
