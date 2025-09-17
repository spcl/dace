import dace
import copy
import numpy
import pytest

from dace.transformation.layout.split_dimension import SplitDimensions

N = dace.symbol("N")

@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = 0.5 * (A[i] + B[i])

@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.5 * (A[i, j] + B[i, j])

@dace.program
def tadd(A: dace.float64[N, N, N], B: dace.float64[N, N, N], C: dace.float64[N, N, N]):
    for i, j, k in dace.map[0:N, 0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j, k] = 0.5 * (A[i, j, k] + B[i, j, k])

def _add_interstate_access(sdfg: dace.SDFG, arr_name: str, arr: dace.data.Array, split_map):
    dim_count = len(arr.shape)
    access_str = ("2," * dim_count)[:-1]
    state = next(iter(sdfg.all_states()))
    parent_graph = state.parent_graph
    sdfg.add_symbol("X", stype=numpy.float64)
    second_state = parent_graph.add_state_after(state, is_start_block=False,
                                                assignments={"X": f"{arr_name}[{access_str}]"})

    pass


def test_vector_dim_split_with_block_size():
    original_sdfg = vadd.to_sdfg()
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = "vadd_split"

    split_map = {
        "A": ([True], [16]),
        "B": ([True], [32]),
    }

    SplitDimensions(split_map=split_map).apply_pass(transformed_sdfg, {})
    transformed_sdfg.validate()

    _N = 16*32
    A1 = numpy.random.rand(_N)
    B1 = numpy.random.rand(_N)
    C1 = numpy.random.rand(_N)
    original_sdfg(A=A1, B=B1, C=C1, N=_N)

    A2 = A1.copy()
    B2 = B1.copy()
    C2 = C1.copy()
    transformed_sdfg(A=A2, B=B2, C=C2, N=_N)

    assert all(a1 == a2 for a1,a2 in zip(A1, A2))
    assert all(b1 == b2 for b1,b2 in zip(B1, B2))
    
    if not numpy.allclose(C2, C1):
        print(C2-C1)
    else:
        pass
    assert numpy.allclose(C2, C1)

def test_matrix_dim_split_with_block_size():
    original_sdfg = vadd.to_sdfg()
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = "vadd_split"

    split_map = {
        "A": ([True], [16]),
        "B": ([True], [32]),
    }

    SplitDimensions(split_map=split_map).apply_pass(transformed_sdfg, {})
    transformed_sdfg.validate()
    transformed_sdfg.save("a1.sdfgz", compress=True)

    _N = 16*32
    A1 = numpy.random.rand(_N)
    B1 = numpy.random.rand(_N)
    C1 = numpy.random.rand(_N)
    original_sdfg(A=A1, B=B1, C=C1, N=_N)

    A2 = A1.copy()
    B2 = B1.copy()
    C2 = C1.copy()
    transformed_sdfg(A=A2, B=B2, C=C2, N=_N)

    assert all(a1 == a2 for a1,a2 in zip(A1, A2))
    assert all(b1 == b2 for b1,b2 in zip(B1, B2))
    
    if not numpy.allclose(C2, C1):
        print(C2-C1)
    else:
        pass
    assert numpy.allclose(C2, C1)

def test_vector_dim_split_with_block_size():
    original_sdfg = madd.to_sdfg()
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = "madd_split"

    split_map = {
        "A": ([True, True], [4, 4]),
        "B": ([True, True], [8, 8]),
    }

    SplitDimensions(split_map=split_map).apply_pass(transformed_sdfg, {})
    transformed_sdfg.validate()
    transformed_sdfg.save("a2.sdfgz", compress=True)

    _N = 8*8*8*8
    A1 = numpy.random.rand(_N)
    B1 = numpy.random.rand(_N)
    C1 = numpy.random.rand(_N)
    original_sdfg(A=A1, B=B1, C=C1, N=_N)

    A2 = A1.copy()
    B2 = B1.copy()
    C2 = C1.copy()
    transformed_sdfg(A=A2, B=B2, C=C2, N=_N)

    assert all(a1 == a2 for a1,a2 in zip(A1, A2))
    assert all(b1 == b2 for b1,b2 in zip(B1, B2))
    
    if not numpy.allclose(C2, C1):
        print(C2-C1)
    else:
        pass
    assert numpy.allclose(C2, C1)

def test_tensor_dim_split_with_block_size():
    original_sdfg = tadd.to_sdfg()
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = "tadd_split"

    split_map = {
        "A": ([False, True, True], [1, 4, 4]),
        "B": ([False, True, True], [1, 8, 8]),
    }

    SplitDimensions(split_map=split_map).apply_pass(transformed_sdfg, {})
    transformed_sdfg.validate()
    transformed_sdfg.save("a3.sdfgz", compress=True)

    _N = 8*8*8*8*2
    A1 = numpy.random.rand(_N)
    B1 = numpy.random.rand(_N)
    C1 = numpy.random.rand(_N)
    original_sdfg(A=A1, B=B1, C=C1, N=_N)

    A2 = A1.copy()
    B2 = B1.copy()
    C2 = C1.copy()
    transformed_sdfg(A=A2, B=B2, C=C2, N=_N)

    assert all(a1 == a2 for a1,a2 in zip(A1, A2))
    assert all(b1 == b2 for b1,b2 in zip(B1, B2))
    
    if not numpy.allclose(C2, C1):
        print(C2-C1)
    else:
        pass
    assert numpy.allclose(C2, C1)

if __name__ == "__main__":
    test_vector_dim_split_with_block_size()
    test_matrix_dim_split_with_block_size()
    test_tensor_dim_split_with_block_size()
    # TODO:
    # interstate edge tests