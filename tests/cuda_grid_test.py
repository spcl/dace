# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function
import os

import dace
from dace.transformation.dataflow import GPUTransformMap, Vectorization
from dace.codegen import compiled_sdfg
import numpy as np
import pytest
import subprocess

N = dace.symbol('N')


def was_vectorized(sdfg: dace.SDFG) -> bool:
    """ Tests whether a binary contains 128-bit CUDA memory operations. """
    csdfg: compiled_sdfg.CompiledSDFG = sdfg.compile()
    output: bytes = subprocess.check_output(
        ['cuobjdump', '-sass', csdfg.filename], stderr=subprocess.STDOUT)
    del csdfg
    return b'.128' in output


@dace.program(dace.float32[N], dace.float32[N])
def cudahello(V, Vout):
    # Transient variable
    @dace.map(_[0:N])
    def multiplication(i):
        in_V << V[i]
        out >> Vout[i]
        out = in_V * 2.0


def _test(sdfg):
    N.set(52)

    print('Vector double CUDA %d' % (N.get()))

    V = dace.ndarray([N], dace.float32)
    Vout = dace.ndarray([N], dace.float32)
    V[:] = np.random.rand(N.get()).astype(dace.float32.type)
    Vout[:] = dace.float32(0)

    sdfg(V=V, Vout=Vout, N=N)

    diff = np.linalg.norm(2 * V - Vout) / N.get()
    print("Difference:", diff)
    assert diff <= 1e-5


def test_cpu():
    sdfg = cudahello.to_sdfg()
    sdfg.name = "cuda_grid_cpu"
    _test(sdfg)


@pytest.mark.gpu
def test_gpu():
    sdfg = cudahello.to_sdfg()
    sdfg.name = "cuda_grid_gpu"
    assert sdfg.apply_transformations(GPUTransformMap) == 1
    _test(sdfg)


@pytest.mark.gpu
def test_gpu_vec():
    sdfg: dace.SDFG = cudahello.to_sdfg()
    sdfg.name = "cuda_grid_gpu_vec"
    assert sdfg.apply_transformations([GPUTransformMap, Vectorization]) == 2
    _test(sdfg)

    # Skip if not CUDA
    if dace.Config.get_bool('compiler', 'cuda', 'backend') == 'cuda':
        assert was_vectorized(sdfg)


if __name__ == "__main__":
    test_cpu()
