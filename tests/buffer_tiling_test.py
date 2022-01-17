# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import BufferTiling
import unittest
import numpy as np

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def conv3x3(weights: dace.float32[3, 3], A: dace.float32[I, J], B: dace.float32[I, J]):
    @dace.map
    def conv3x3(y: _[1:I - 1], x: _[1:J - 1]):
        inp << A[y - 1:y + 2, x - 1:x + 2]
        w << weights
        out >> B[y, x]
        out = (w[0, 0] * inp[0, 0] + w[0, 1] * inp[0, 1] + w[0, 2] * inp[0, 2] + w[1, 0] * inp[1, 0] +
               w[1, 1] * inp[1, 1] + w[1, 2] * inp[1, 2] + w[2, 0] * inp[2, 0] + w[2, 1] * inp[2, 1] +
               w[2, 2] * inp[2, 2])


@dace.program
def conv3x3_transposed(weights: dace.float32[3, 3], A: dace.float32[I, J], B: dace.float32[I, J]):
    @dace.map
    def conv3x3(y: _[1:I - 1], x: _[1:J - 1]):
        inp << A[x - 1:x + 2, y - 1:y + 2]
        w << weights
        out >> B[y, x]
        out = (w[0, 0] * inp[0, 0] + w[0, 1] * inp[0, 1] + w[0, 2] * inp[0, 2] + w[1, 0] * inp[1, 0] +
               w[1, 1] * inp[1, 1] + w[1, 2] * inp[1, 2] + w[2, 0] * inp[2, 0] + w[2, 1] * inp[2, 1] +
               w[2, 2] * inp[2, 2])


@dace.program
def conv5x5(weights: dace.float32[5, 5], A: dace.float32[I, J], B: dace.float32[I, J]):
    @dace.map
    def conv5x5(y: _[3:I - 3], x: _[3:J - 3]):
        inp << A[y - 2:y + 3, x - 2:x + 3]
        w << weights
        out >> B[y, x]
        out = (w[0, 0] * inp[0, 0] + w[0, 1] * inp[0, 1] + w[0, 2] * inp[0, 2] + w[0, 3] * inp[0, 3] +
               w[0, 4] * inp[0, 4] + w[1, 0] * inp[1, 0] + w[1, 1] * inp[1, 1] + w[1, 2] * inp[1, 2] +
               w[1, 3] * inp[1, 3] + w[1, 4] * inp[1, 4] + w[2, 0] * inp[2, 0] + w[2, 1] * inp[2, 1] +
               w[2, 2] * inp[2, 2] + w[2, 3] * inp[2, 3] + w[2, 4] * inp[2, 4] + w[3, 0] * inp[3, 0] +
               w[3, 1] * inp[3, 1] + w[3, 2] * inp[3, 2] + w[3, 3] * inp[3, 3] + w[3, 4] * inp[3, 4] +
               w[4, 0] * inp[4, 0] + w[4, 1] * inp[4, 1] + w[4, 2] * inp[4, 2] + w[4, 3] * inp[4, 3] +
               w[4, 4] * inp[4, 4])


@dace.program
def conv3x3_5x5(w3: dace.float32[3, 3], w5: dace.float32[5, 5], A: dace.float32[I, J], B: dace.float32[I, J]):
    buf = dace.ndarray([I, J], dtype=dace.float32)
    conv3x3(w3, A, buf)
    conv5x5(w5, buf, B)


@dace.program
def conv3x3_5x5_transposed(w3: dace.float32[3, 3], w5: dace.float32[5, 5], A: dace.float32[I, J], B: dace.float32[I,
                                                                                                                  J]):
    buf = dace.ndarray([I, J], dtype=dace.float32)
    conv3x3_transposed(w3, A, buf)
    conv5x5(w5, buf, B)


def _semantic_eq(tile_sizes, program):
    w3 = np.random.rand(3, 3).astype(np.float32)
    w5 = np.random.rand(5, 5).astype(np.float32)
    A = np.random.rand(16, 16).astype(np.float32)
    B1 = np.zeros((16, 16), dtype=np.float32)
    B2 = np.zeros((16, 16), dtype=np.float32)

    sdfg = program.to_sdfg()
    sdfg.name = f"{sdfg.name}_{'_'.join(map(str, tile_sizes))}"
    sdfg.simplify()
    sdfg(w3=w3, w5=w5, A=A, B=B1, I=A.shape[0], J=A.shape[1])

    count = sdfg.apply_transformations(BufferTiling, options={'tile_sizes': tile_sizes})
    assert count > 0
    sdfg(w3=w3, w5=w5, A=A, B=B2, I=A.shape[0], J=A.shape[1])

    assert np.allclose(B1, B2)


def test_basic():
    _semantic_eq([3, 3], conv3x3_5x5)


def test_transposed():
    _semantic_eq([3, 3], conv3x3_5x5_transposed)


def test_tile_size_1():
    _semantic_eq([1, 1], conv3x3_5x5)


def test_tile_size_1_transposed():
    _semantic_eq([1, 1], conv3x3_5x5_transposed)


if __name__ == '__main__':
    test_basic()
    test_transposed()
    test_tile_size_1()
    test_tile_size_1_transposed()
