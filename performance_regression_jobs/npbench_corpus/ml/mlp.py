# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``mlp`` (ml) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'C_in': 3, 'N': 8, 'S0': 30000, 'S1': 2000, 'S2': 2000}
INPUT_ARGS = ('C_in', 'N', 'S0', 'S1', 'S2')
ARRAY_ARGS = ('input', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

C_in, N, S0, S1, S2, N1, N2 = (dc.symbol(s, dtype=dc.int64) for s in ('C_in', 'N', 'S0', 'S1', 'S2', 'N1', 'N2'))


def initialize(C_in, N, S0, S1, S2, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    mlp_sizes = [S0, S1, S2]
    input = np.random.rand(N, C_in).astype(datatype)
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=datatype)
    b1 = rng.random((mlp_sizes[0], ), dtype=datatype)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=datatype)
    b2 = rng.random((mlp_sizes[1], ), dtype=datatype)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=datatype)
    b3 = rng.random((mlp_sizes[2], ), dtype=datatype)
    out = np.zeros((N, mlp_sizes[2]), dtype=datatype)
    return (input, w1, b1, w2, b2, w3, b3, out)


def reference(input, w1, b1, w2, b2, w3, b3, out):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    out[:] = softmax(x @ w3 + b3)


@dc.program
def relu(x: dc_float[N1, N2]):
    return np.maximum(x, 0)


@dc.program
def softmax(x: dc_float[N1, N2]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


@dc.program
def kernel(input: dc_float[N, C_in], w1: dc_float[C_in, S0], b1: dc_float[S0], w2: dc_float[S0, S1], b2: dc_float[S1],
           w3: dc_float[S1, S2], b3: dc_float[S2]):
    x1 = relu(input @ w1 + b1)
    x2 = relu(x1 @ w2 + b2)
    x3 = softmax(x2 @ w3 + b3)
    return x3


CORPUS = dict(name='mlp',
              dwarf='ml',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
