import dace
import copy
import numpy
from dace.properties import CodeBlock
import pytest

from dace.transformation.layout.collapse_dimension import CollapseDimensions

N = dace.symbol("N")
M = dace.symbol("M")

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


@dace.program
def tadd_blocked(A: dace.float64[N, N, N], B: dace.float64[N, N, N], C: dace.float64[N, N, N]):
    for i, j, k in dace.map[0:N, 0:N:8, 0:N:8] @ dace.ScheduleType.Sequential:
        for jj, kk in dace.map[j:j + 8:1, k:k + 8:1] @ dace.ScheduleType.Sequential:
            C[i, jj, kk] = 0.5 * (A[i, jj, kk] + B[i, jj, kk])


def _add_interstate_access(sdfg: dace.SDFG, arr_name: str, arr: dace.data.Array):
    dim_count = len(arr.shape)
    access_str: str = ("2," * dim_count)[:-1]
    state = next(iter(sdfg.all_states()))
    parent_graph = state.parent_graph
    sdfg.add_symbol("X", stype=numpy.float64)
    second_state = parent_graph.add_state_after(state,
                                                is_start_block=False,
                                                assignments={"X": f"{arr_name}[{access_str}]"})
    sdfg.add_array("sc1", dtype=dace.float64, transient=False, shape=(1, ))
    sdfg.add_array("sc2", dtype=dace.float64, transient=False, shape=(1, ))
    t1 = second_state.add_tasklet(name="read_from_sym", inputs=set(), outputs={"_out"}, code="_out = X")
    t1.add_out_connector("_out")
    second_state.add_edge(t1, "_out", second_state.add_access("sc1"), None, dace.memlet.Memlet(expr="sc1[0]"))

    t2 = second_state.add_tasklet(name="read_from_arr", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
    t2.add_in_connector("_in")
    t2.add_out_connector("_out")

    second_state.add_edge(second_state.add_access(arr_name), None, t2, "_in",
                          dace.memlet.Memlet(expr=f"{arr_name}[{access_str}]"))
    second_state.add_edge(t2, "_out", second_state.add_access("sc2"), None, dace.memlet.Memlet(expr=f"sc2[0]"))



def test_tensor_add():
    original_sdfg = tadd.to_sdfg()
    original_sdfg.name = "tensor_add"
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = "tensor_add_collapsed"

    collapse_map = {
        "A": (0, 1,),
        "B": (0, 1,),
        "C": (0, 1,),
    }

    CollapseDimensions(collapse_map=collapse_map).apply_pass(transformed_sdfg, {})
    transformed_sdfg.validate()

    _N = 16
    A1 = numpy.random.rand(_N, _N, _N)
    B1 = numpy.random.rand(_N, _N, _N)
    C1 = numpy.random.rand(_N, _N, _N)
    original_sdfg(A=A1, B=B1, C=C1, N=_N)

    A2 = A1.copy()
    B2 = B1.copy()
    C2 = C1.copy()
    transformed_sdfg(A=A2, B=B2, C=C2, N=_N)

    assert numpy.array_equal(A1, A2)
    assert numpy.array_equal(B1, B2)

    if not numpy.array_equal(C2, C1):
        print(C2 - C1)
    assert numpy.array_equal(C2, C1)


def test_tensor_add_with_tiling():
    original_sdfg = tadd_blocked.to_sdfg()
    original_sdfg.name = "tensor_add_with_tiling"
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = "tensor_add_with_tiling_collapsed"

    collapse_map = {
        "A": (0, 1,),
        "B": (0, 1,),
        "C": (0, 1,),
    }

    CollapseDimensions(collapse_map=collapse_map, verbose=False).apply_pass(transformed_sdfg, {})
    transformed_sdfg.validate()

    _N = 16
    A1 = numpy.random.rand(_N, _N, _N)
    B1 = numpy.random.rand(_N, _N, _N)
    C1 = numpy.random.rand(_N, _N, _N)
    original_sdfg(A=A1, B=B1, C=C1, N=_N)

    A2 = A1.copy()
    B2 = B1.copy()
    C2 = C1.copy()
    transformed_sdfg(A=A2, B=B2, C=C2, N=_N)

    assert numpy.array_equal(A1, A2)
    assert numpy.array_equal(B1, B2)

    if not numpy.array_equal(C2, C1):
        print(C2 - C1)
    assert numpy.array_equal(C2, C1)


def test_matrix_add_with_interstate_access():
    original_sdfg = madd.to_sdfg()
    original_sdfg.name = "matrix_add_with_iedge"
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = "matrix_add_with_iedge_collapsed"
    transformed_sdfg.validate()
    _add_interstate_access(sdfg=transformed_sdfg, arr_name="A", arr=transformed_sdfg.arrays["A"])
    transformed_sdfg.validate()

    collapse_map = {
        "A": (0, 1),
        "B": (0, 1),
        "B": (0, 1),
    }

    CollapseDimensions(collapse_map=collapse_map).apply_pass(transformed_sdfg, {})
    transformed_sdfg.validate()


    _N = 16 * 4 * 2
    A1 = numpy.random.rand(_N, _N)
    B1 = numpy.random.rand(_N, _N)
    C1 = numpy.random.rand(_N, _N)
    original_sdfg(A=A1, B=B1, C=C1, N=_N)

    A2 = A1.copy()
    B2 = B1.copy()
    C2 = C1.copy()
    SC1 = numpy.random.rand(1)
    SC2 = numpy.random.rand(1)
    transformed_sdfg(A=A2, B=B2, C=C2, N=_N, sc1=SC1, sc2=SC2)

    assert numpy.array_equal(A1, A2)
    assert numpy.array_equal(B1, B2)

    if not numpy.array_equal(C2, C1):
        print(C2 - C1)
    assert numpy.array_equal(C2, C1)
    assert numpy.array_equal(SC1, SC2)
    assert numpy.array_equal(SC2[0], A2[2,2])
    assert numpy.array_equal(SC1[0], A2[2,2])


@dace.program
def matrix_add(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M:1, 0:N:1]:
        C[i, j] = A[i, j] + B[i, j]

def test_matrix_add():
    sdfg1 = matrix_add.to_sdfg()
    sdfg1.name = "matrix_add"
    sdfg2 = copy.deepcopy(sdfg1)
    sdfg2.name = "matrix_add"
    CollapseDimensions(collapse_map={"A": (0, 1), "B": (0, 1), "C": (0, 1)}).apply_pass(sdfg2, {})

    # Create test data
    Mv, Nv = 64, 32
    A = numpy.random.rand(Mv, Nv)
    B = numpy.random.rand(Mv, Nv)
    C1 = numpy.zeros((Mv, Nv))
    C2 = numpy.zeros((Mv, Nv))

    # Run original SDFG
    sdfg1(A=A, B=B, C=C1, M=Mv, N=Nv)

    # Run transformed SDFG
    sdfg2(A=A, B=B, C=C2, M=Mv, N=Nv)

    # Compare
    if not numpy.allclose(C1, C2):
        print("ERROR: Results differ!")
        max_diff = numpy.max(numpy.abs(C1 - C2))
        print("Max difference:", max_diff)
        print("C1:", C1)
        print("C2:", C2)
    assert numpy.allclose(C1, C2)


def test_matrix_add_partial():
    sdfg1 = matrix_add.to_sdfg()
    sdfg1.name = "matrix_add_partial"
    sdfg2 = copy.deepcopy(sdfg1)
    sdfg2.name = "matrix_add_partial_collapsed"
    CollapseDimensions(collapse_map={"A": (0, 1), }).apply_pass(sdfg2, {})

    # Create test data
    Mv, Nv = 64, 32
    A = numpy.random.rand(Mv, Nv)
    B = numpy.random.rand(Mv, Nv)
    C1 = numpy.zeros((Mv, Nv))
    C2 = numpy.zeros((Mv, Nv))

    # Run original SDFG
    sdfg1(A=A, B=B, C=C1, M=Mv, N=Nv)

    # Run transformed SDFG
    sdfg2(A=A, B=B, C=C2, M=Mv, N=Nv)

    # Compare
    if not numpy.allclose(C1, C2):
        print("ERROR: Results differ!")
        max_diff = numpy.max(numpy.abs(C1 - C2))
        print("Max difference:", max_diff)
        print("C1:", C1)
        print("C2:", C2)
    assert numpy.allclose(C1, C2)


if __name__ == "__main__":
    # Basic tests
    test_matrix_add()
    test_matrix_add_partial()
    test_tensor_add()
    # Blocked shape tests (For the perfect match optimization)
    test_tensor_add_with_tiling()
    # Interstate edge tests
    test_matrix_add_with_interstate_access()
