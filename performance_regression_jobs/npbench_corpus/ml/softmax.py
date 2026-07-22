# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``softmax`` (ml) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'N': 16, 'H': 16, 'SM': 128}
INPUT_ARGS = ('N', 'H', 'SM')
ARRAY_ARGS = ('x', 'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

N, H, SM = (dc.symbol(s, dc.int64) for s in ('N', 'H', 'SM'))


def initialize(N, H, SM, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=datatype)
    out = np.zeros_like(x)
    return (x, out)


def reference(x, out):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    out[:] = tmp_out / tmp_sum


@dc.program
def kernel(x: dc_float[N, H, SM, SM]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


@dc.program
def softmax_gpu(x: dc_float[N, H, SM, SM], out: dc_float[N, H, SM, SM]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    out[:] = tmp_out / tmp_sum


CORPUS = dict(name='softmax',
              dwarf='ml',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
