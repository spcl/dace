# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data
from dace.sdfg.graph import NodeNotFoundError
import dace
from dace import SDFG
import dace.sdfg.nodes as nodes
import dace.sdfg.analysis.vector_inference as vector_inference
import pytest
from dace.transformation.dataflow import MergeSourceSinkArrays

N = dace.symbol('N')


def find_tasklet_by_connector(sdfg: SDFG, name: str):
    for node, _ in sdfg.start_state.all_nodes_recursive():
        if name in node.in_connectors:
            return node
        elif name in node.out_connectors:
            return node

    raise NodeNotFoundError(f'Could not find connector "{name}"')


def find_map_entry(sdfg: SDFG):
    for node, _ in sdfg.start_state.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            return node

    raise NodeNotFoundError(f'Could not find map entry')


def vectorize(sdfg: SDFG) -> vector_inference.VectorInferenceGraph:
    return vector_inference.infer_vectors(sdfg, sdfg.start_state, find_map_entry(sdfg), -1, apply=False)


def is_vector_connector(inf: vector_inference.VectorInferenceGraph, conn: str, is_in: bool):
    return inf.get_constraint((find_tasklet_by_connector(inf.sdfg,
                                                         conn), conn, is_in)) == vector_inference.InferenceNode.Vector


def has_vector_accessnode(inf: vector_inference.VectorInferenceGraph):
    for node, _ in inf.sdfg.start_state.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode) and isinstance(node.desc(inf.sdfg), data.Scalar):
            return inf.get_constraint(node) == vector_inference.InferenceNode.Vector
    return False


def test_simple():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(simplify=True)
    inf = vectorize(sdfg)
    assert is_vector_connector(inf, 'a', True)
    assert is_vector_connector(inf, 'b', False)


def test_always_scalar_output():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = 0.0  # looks like b is a scalar (but isn't)

    sdfg = program.to_sdfg(simplify=True)
    inf = vectorize(sdfg)

    assert is_vector_connector(inf, 'a', True)
    # Even though b is always a scalar (according to code inference), the output still must be a vector
    # because the memlet contains the loop param (becomes a broadcast in the Tasklet)
    assert is_vector_connector(inf, 'b', False)


def test_scalar_accessnode_vector():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            scal = dace.define_local_scalar(dace.float32)
            with dace.tasklet:
                a << A[i]
                x_out >> scal
                x_out = a  # x_out and scal should be a vector
            with dace.tasklet:
                x_in << scal
                b >> B[i]
                b = x_in

    sdfg = program.to_sdfg(simplify=True)
    inf = vectorize(sdfg)

    assert is_vector_connector(inf, 'a', True)
    assert is_vector_connector(inf, 'x_out', False)
    assert is_vector_connector(inf, 'x_in', True)
    assert is_vector_connector(inf, 'b', False)

    assert has_vector_accessnode(inf)


def test_scalar_accessnode_scalar():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            scal = dace.define_local_scalar(dace.float32)
            with dace.tasklet:
                a << A[0]
                x_out >> scal
                x_out = a
            with dace.tasklet:
                x_in << scal
                b >> B[i]
                b = x_in

    sdfg = program.to_sdfg(simplify=True)
    inf = vectorize(sdfg)

    # Except for b every connector is scalar
    # (b is vector because of Memlet containing loop param)
    assert not is_vector_connector(inf, 'a', True)
    assert not is_vector_connector(inf, 'x_out', False)
    assert not is_vector_connector(inf, 'x_in', True)
    assert is_vector_connector(inf, 'b', False)

    assert not has_vector_accessnode(inf)


def test_array_accessnode_scalar():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            arr = dace.define_local(N, dace.float32)
            with dace.tasklet:
                a << A[0]
                x_out >> arr[0]  # now write into an Array access node
                x_out = a
            with dace.tasklet:
                x_in << arr[0]
                b >> B[i]
                b = x_in

    sdfg = program.to_sdfg(simplify=True)
    inf = vectorize(sdfg)

    # Again except for b every connector is scalar
    # (b is vector because of Memlet containing loop param)
    assert not is_vector_connector(inf, 'a', True)
    assert not is_vector_connector(inf, 'x_out', False)
    assert not is_vector_connector(inf, 'x_in', True)
    assert is_vector_connector(inf, 'b', False)


def test_array_accessnode_violation():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            arr = dace.define_local(N, dace.float32)
            with dace.tasklet:
                a << A[i]
                # Reading a vector but storing a scalar (must fail)
                x_out >> arr[0]
                x_out = a
            with dace.tasklet:
                x_in << arr[i]
                b >> B[i]
                b = x_in

    sdfg = program.to_sdfg(simplify=True)
    with pytest.raises(vector_inference.VectorInferenceException):
        vectorize(sdfg)


def test_array_accessnode_complicated():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            arr = dace.define_local(N, dace.float32)
            scal = dace.define_local_scalar(dace.float32)
            with dace.tasklet:
                a << A[0]
                x_out >> arr[i]
                y_out >> scal
                x_out = a  # x_out looks like a scalar, but must be a vector (broadcast within Tasklet)
                y_out = a
            with dace.tasklet:
                x_in << arr[0]
                y_in << scal
                b >> B[i]
                b = x_in * y_in

    # The challenge here is that one Tasklet writes a vector into an Array access node
    # But the other Tasklet then reads a scalar from it (is okay because its an Array)

    sdfg = program.to_sdfg(simplify=True)
    inf = vectorize(sdfg)

    assert not is_vector_connector(inf, 'a', True)
    assert is_vector_connector(inf, 'x_out', False)
    assert not is_vector_connector(inf, 'y_out', False)
    assert not is_vector_connector(inf, 'x_in', True)
    assert not is_vector_connector(inf, 'y_in', True)
    assert is_vector_connector(inf, 'b', False)

    assert not has_vector_accessnode(inf)


def test_multi_input():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            scal = dace.define_local_scalar(dace.float32)
            with dace.tasklet:
                a << A[i]
                x_out >> scal
                x_out = a
            with dace.tasklet:
                x_in << scal
                y_in << scal
                z_in << scal
                b >> B[i]
                b = x_in * y_in * z_in

    # x_in, y_in and z_in are vectors

    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(MergeSourceSinkArrays)
    sdfg.simplify()
    inf = vectorize(sdfg)

    assert is_vector_connector(inf, 'a', True)
    assert is_vector_connector(inf, 'x_out', False)
    assert is_vector_connector(inf, 'x_in', True)
    assert is_vector_connector(inf, 'y_in', True)
    assert is_vector_connector(inf, 'z_in', True)
    assert is_vector_connector(inf, 'b', False)

    assert has_vector_accessnode(inf)


def test_multi_input_violation():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            scal = dace.define_local_scalar(dace.float32)
            with dace.tasklet:
                a << A[i]
                x_out >> scal
                x_out = a
            with dace.tasklet:
                x_in << scal
                y_in << scal
                z_in << scal
                b >> B[i]
                b = z_in

    # This is an artificial constraint test where z_in wants scal as
    # a scalar instead of a vector
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(MergeSourceSinkArrays)
    sdfg.simplify()
    inf = vector_inference.VectorInferenceGraph(sdfg, sdfg.start_state, find_map_entry(sdfg), -1)
    inf.set_constraint((find_tasklet_by_connector(sdfg, 'z_in'), 'z_in', True), vector_inference.InferenceNode.Scalar)

    # It has to fail because the access node will be inferred as vector
    # (since someone writes a vector into it)
    with pytest.raises(vector_inference.VectorInferenceException):
        inf.infer()


if __name__ == '__main__':
    test_simple()
    test_always_scalar_output()
    test_scalar_accessnode_vector()
    test_scalar_accessnode_scalar()
    test_array_accessnode_scalar()
    test_array_accessnode_violation()
    test_multi_input()
    test_multi_input_violation()
