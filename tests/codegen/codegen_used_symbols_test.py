# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests used-symbols in code generation."""
import dace
import numpy
import pytest


n0i, n0j, n0k = (dace.symbol(s, dtype=dace.int32) for s in ('n0i', 'n0j', 'n0k'))
n1i, n1j, n1k = (dace.symbol(s, dtype=dace.int64) for s in ('n1i', 'n1j', 'n1k'))


@dace.program
def rprj3(r: dace.float64[n0i, n0j, n0k], s: dace.float64[n1i, n1j, n1k]):

    for i, j, k in dace.map[1:s.shape[0] - 1, 1:s.shape[1] - 1, 1:s.shape[2] - 1]:

        s[i, j, k] = (
            0.5000 * r[2 * i, 2 * j, 2 * k] +
            0.2500 * (r[2 * i - 1, 2 * j, 2 * k] + r[2 * i + 1, 2 * j, 2 * k] + r[2 * i, 2 * j - 1, 2 * k] +
                      r[2 * i, 2 * j + 1, 2 * k] + r[2 * i, 2 * j, 2 * k - 1] + r[2 * i, 2 * j, 2 * k + 1]) +
            0.1250 * (r[2 * i - 1, 2 * j - 1, 2 * k] + r[2 * i - 1, 2 * j + 1, 2 * k] +
                      r[2 * i + 1, 2 * j - 1, 2 * k] + r[2 * i + 1, 2 * j + 1, 2 * k] +
                      r[2 * i - 1, 2 * j, 2 * k - 1] + r[2 * i - 1, 2 * j, 2 * k + 1] +
                      r[2 * i + 1, 2 * j, 2 * k - 1] + r[2 * i + 1, 2 * j, 2 * k + 1] +
                      r[2 * i, 2 * j - 1, 2 * k - 1] + r[2 * i, 2 * j - 1, 2 * k + 1] +
                      r[2 * i, 2 * j + 1, 2 * k - 1] + r[2 * i, 2 * j + 1, 2 * k + 1]) +
            0.0625 * (r[2 * i - 1, 2 * j - 1, 2 * k - 1] + r[2 * i - 1, 2 * j - 1, 2 * k + 1] +
                      r[2 * i - 1, 2 * j + 1, 2 * k - 1] + r[2 * i - 1, 2 * j + 1, 2 * k + 1] +
                      r[2 * i + 1, 2 * j - 1, 2 * k - 1] + r[2 * i + 1, 2 * j - 1, 2 * k + 1] +
                      r[2 * i + 1, 2 * j + 1, 2 * k - 1] + r[2 * i + 1, 2 * j + 1, 2 * k + 1]))


def test_codegen_used_symbols_cpu():

    rng = numpy.random.default_rng(42)
    r = rng.random((10, 10, 10))
    s_ref = numpy.zeros((4, 4, 4))
    s_val = numpy.zeros((4, 4, 4))

    rprj3.f(r, s_ref)
    rprj3(r, s_val)

    assert numpy.allclose(s_ref, s_val)


def test_codegen_used_symbols_cpu_2():

    @dace.program
    def rprj3_nested(r: dace.float64[n0i, n0j, n0k], s: dace.float64[n1i, n1j, n1k]):
        rprj3(r, s)

    rng = numpy.random.default_rng(42)
    r = rng.random((10, 10, 10))
    s_ref = numpy.zeros((4, 4, 4))
    s_val = numpy.zeros((4, 4, 4))

    rprj3.f(r, s_ref)
    rprj3_nested(r, s_val)

    assert numpy.allclose(s_ref, s_val)


@pytest.mark.gpu
def test_codegen_used_symbols_gpu():

    sdfg = rprj3.to_sdfg()
    for _, desc in sdfg.arrays.items():
        if not desc.transient and isinstance(desc, dace.data.Array):
            desc.storage = dace.StorageType.GPU_Global
    sdfg.apply_gpu_transformations()
    func = sdfg.compile()

    try:
        import cupy

        rng = numpy.random.default_rng(42)
        r = rng.random((10, 10, 10))
        r_dev = cupy.asarray(r)
        s_ref = numpy.zeros((4, 4, 4))
        s_val = cupy.zeros((4, 4, 4))

        rprj3.f(r, s_ref)
        func(r=r_dev, s=s_val, n0i=10, n0j=10, n0k=10, n1i=4, n1j=4, n1k=4)

        assert numpy.allclose(s_ref, s_val)
    
    except (ImportError, ModuleNotFoundError):
        pass


if __name__ == "__main__":

    test_codegen_used_symbols_cpu()
    test_codegen_used_symbols_cpu_2()
    test_codegen_used_symbols_gpu()
