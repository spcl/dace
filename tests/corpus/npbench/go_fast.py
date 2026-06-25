# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``go_fast`` (map_reduce).

Self-contained fusion of the npbench repo pieces -- the ``initialize`` input
generator, the numpy ``reference`` kernel, the ``@dace.program`` ``kernel``, and
the ``S`` dataset preset -- so the corpus needs no external npbench dependency.
``CORPUS`` is the uniform descriptor the loader (:mod:`tests.corpus.npbench`)
consumes.
"""
import numpy as np

import dace as dc

dc_float = dc.float32

#: ``S`` dataset preset (from the benchmark manifest).
SIZES = {"N": 2000}
#: ``initialize`` positional args (symbol order) and the array names it returns.
INPUT_ARGS = ("N", )
ARRAY_ARGS = ("a", "out")
#: Arrays compared against the reference.
OUTPUT_ARGS = ("out", )

N = dc.symbol("N", dtype=dc.int64)


def initialize(N, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    a = rng.random((N, N), dtype=datatype)
    out = np.zeros((N, N), dtype=datatype)
    return a, out


def reference(a, out):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    out[:] = a + trace


@dc.program
def kernel(a: dc_float[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace


CORPUS = dict(name="go_fast",
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
