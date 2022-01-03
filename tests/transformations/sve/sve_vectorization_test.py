# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.sdfg.graph import NodeNotFoundError
import dace
import numpy as np
from dace.transformation.dataflow.sve.vectorization import SVEVectorization
from dace import SDFG
import dace.dtypes as dtypes

N = dace.symbol('N')


def find_connector_by_name(sdfg: SDFG, name: str):
    """
    Utility function to obtain the type of a connector by its name
    """
    for node, _ in sdfg.start_state.all_nodes_recursive():
        if name in node.in_connectors:
            return node.in_connectors[name]
        elif name in node.out_connectors:
            return node.out_connectors[name]

    raise RuntimeError(f'Could not find connector "{name}"')


def test_basic_stride():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(coarsen=True)
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_irregular_stride():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N * N]:
            with dace.tasklet:
                a << A[i * i]
                b >> B[i * i]
                b = a

    sdfg = program.to_sdfg(coarsen=True)
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

    sdfg = program.to_sdfg(coarsen=True)
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

    sdfg = program.to_sdfg(coarsen=True)
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

    sdfg = program.to_sdfg(coarsen=True)
    # Complex datatypes are currently not supported by the codegen
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_first_level_vectorization():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i, j in dace.map[0:N, 0:N]:
            with dace.tasklet:
                a_scal << A[i]
                a_vec << A[j]
                b >> B[j]
                b = a_vec

    sdfg = program.to_sdfg(coarsen=True)
    sdfg.apply_transformations(SVEVectorization)

    # i is constant in the vectorized map
    assert not isinstance(find_connector_by_name(sdfg, 'a_scal'), dtypes.vector)
    # j is the innermost param
    assert isinstance(find_connector_by_name(sdfg, 'a_vec'), dtypes.vector)


def test_stream_push():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        S_out = dace.define_stream(dace.float32, N)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> S_out(-1)
                b = a
        S_out >> B

    sdfg = program.to_sdfg(coarsen=True)
    # Stream push is possible
    assert sdfg.apply_transformations(SVEVectorization) == 1


def test_stream_pop():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        S_in = dace.define_stream(dace.float32, N)
        S_in << A
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << S_in(-1)
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(coarsen=True)
    # Stream pop is not implemented yet
    assert sdfg.apply_transformations(SVEVectorization) == 0
