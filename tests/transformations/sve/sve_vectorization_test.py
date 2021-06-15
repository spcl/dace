# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow.sve.vectorization import SVEVectorization

N = dace.symbol('N')


def test_basic_stride():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_irregular_stride():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N * N]:
            with dace.tasklet:
                a << A[i * i]
                b >> B[i * i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # [i * i] has a stride of 2i + 1 which is not constant (cannot be vectorized)
    assert sdfg.apply_transformations(SVEVectorization) == 0


def test_diagonal_stride():
    @dace.program
    def program(A: dace.float32[N, N], B: dace.float32[N, N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i, i]
                b >> B[i, i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # [i, i] has a stride of N + 1, so it is perfectly fine
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_unsupported_type():
    @dace.program
    def program(A: dace.complex64[N], B: dace.complex64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # Complex datatypes are currently not supported by the codegen
    assert sdfg.apply_transformations(SVEVectorization) == 0


def test_supported_wcr():
    @dace.program
    def program(A: dace.float32[N], B: dace.int32[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: x + y)[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # Complex datatypes are currently not supported by the codegen
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_unsupported_wcr():
    @dace.program
    def program(A: dace.float32[N], B: dace.int32[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: x + y)[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # Vector WCR not supported in SVE
    assert sdfg.apply_transformations(SVEVectorization) == 0
