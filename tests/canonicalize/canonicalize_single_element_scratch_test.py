# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization leaves no single-element transient array: map fusion's ``(1, 1)`` scratch buffers end as scalars."""

import numpy as np

import dace
from dace import data
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")
M = dace.symbol("M")
T = dace.symbol("T")


@dace.program
def fused_through_a_scratch(A: dace.float64[N, M], C: dace.float64[N, M]):
    w = np.zeros((N, M))
    for t in range(T):
        for i in range(N):
            for j in range(M):
                w[i, j] = A[i, j] * t
        C[:] += w


@dace.program
def single_element_argument(A: dace.float64[N, M], s: dace.float64[1, 1]):
    for i in range(N):
        for j in range(M):
            s[0, 0] += A[i, j]


def single_element_transients(sdfg: dace.SDFG) -> list[str]:
    return [
        name
        for sd in sdfg.all_sdfgs_recursive()
        for name, desc in sd.arrays.items()
        if type(desc) is data.Array and desc.transient and all(d == 1 for d in desc.shape)
    ]


def test_fusion_scratch_ends_as_a_scalar():
    sdfg = fused_through_a_scratch.to_sdfg()
    canonicalize(sdfg, validate=True)
    assert single_element_transients(sdfg) == []
    A = np.random.rand(7, 5)
    C = np.random.rand(7, 5)
    expected = C + A * sum(range(3))
    sdfg(A=A, C=C, N=7, M=5, T=3)
    assert np.allclose(C, expected)


def test_single_element_argument_stays_an_array():
    sdfg = single_element_argument.to_sdfg()
    canonicalize(sdfg, validate=True)
    assert type(sdfg.arrays["s"]) is data.Array
    A = np.random.rand(7, 5)
    s = np.ones((1, 1))
    sdfg(A=A, s=s, N=7, M=5)
    assert np.isclose(s[0, 0], 1 + A.sum())


if __name__ == "__main__":
    test_fusion_scratch_ends_as_a_scalar()
    test_single_element_argument_stays_an_array()
