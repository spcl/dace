# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for GPU kernels with scalar outputs. """
import numpy as np
import pytest
import dace
import dace.libraries.blas as blas


@pytest.mark.gpu
def test_dot_gpu():
    @dace.program
    def dot(x: dace.float64[20], y: dace.float64[20]):
        return x @ y

    x = np.random.rand(20)
    y = np.random.rand(20)
    reference = x @ y
    sdfg = dot.to_sdfg()
    sdfg.apply_gpu_transformations()

    # Expand pure version
    oldimpl = blas.default_implementation
    blas.default_implementation = 'pure'

    daceres = sdfg(x=x, y=y)

    # Revert default implementation
    blas.default_implementation = oldimpl

    assert np.allclose(daceres, reference)


@pytest.mark.gpu
def test_scalar_output():
    @dace.program
    def scaltest(A: dace.float64[20, 20]):
        scal = dace.define_local_scalar(dace.float64)
        for _ in dace.map[0:1]:
            with dace.tasklet:
                inp << A[1, 1]
                out >> scal
                out = inp + 5
        return scal

    sdfg = scaltest.to_sdfg()
    sdfg.apply_gpu_transformations()

    A = np.random.rand(20, 20)
    ret = sdfg(A=A)
    assert np.allclose(ret, A[1, 1] + 5)


@pytest.mark.gpu
def test_scalar_output_ptr_access():
    sdfg = dace.SDFG("scalptrtest")
    state = sdfg.add_state()
    sdfg.add_scalar("scal", dace.float64, transient=True, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("__return", [1], dace.float64)

    tasklet = state.add_tasklet(
        "write",
        {},
        {"outp": dace.pointer(dace.float64)},
        """
        double a = 5;
        cudaMemcpyAsync(outp, &a, 1 * sizeof(double), cudaMemcpyHostToDevice,
                        __state->gpu_context->streams[0]);
        """,
        language=dace.dtypes.Language.CPP,
    )
    access_scal = state.add_access("scal")

    write_unsqueezed = state.add_write("__return")
    state.add_edge(tasklet, "outp", access_scal, None, sdfg.make_array_memlet("scal"))
    state.add_edge(access_scal, None, write_unsqueezed, None, sdfg.make_array_memlet("scal"))

    ret = sdfg()
    assert np.allclose(ret, 5)


if __name__ == '__main__':
    test_dot_gpu()
    test_scalar_output()
    test_scalar_output_ptr_access()
