import copy
import dace
import pytest
import numpy
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap

N = dace.symbol("N")

@dace.program
def elementwise_constexpr_size(A: dace.float64[512] @ dace.dtypes.StorageType.GPU_Global,
                                B: dace.float64[512] @ dace.dtypes.StorageType.GPU_Global):
    for i in dace.map[0:512] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i] = 2.0 * B[i]


@dace.program
def elementwise_small_constexpr_size(A: dace.float64[16] @ dace.dtypes.StorageType.GPU_Global,
                                     B: dace.float64[16] @ dace.dtypes.StorageType.GPU_Global):
    for i in dace.map[0:16] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i] = 2.0 * B[i]


@dace.program
def elementwise_symbolic(A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
                         B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global):
    for i in dace.map[0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i] = 2.0 * B[i]


@dace.program
def elementwise_with_floor_div(A: dace.float64[512] @ dace.dtypes.StorageType.GPU_Global,
                               B: dace.float64[512] @ dace.dtypes.StorageType.GPU_Global,
                               i_beg: dace.int64, i_end: dace.int64):
    # To avoid shadowing error coming from the Python-frontend an assignment is necessary
    sym_i_beg = i_beg
    sym_i_end = i_end
    for i in dace.map[sym_i_beg:(sym_i_end//2)] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i] = 2.0 * B[i]


def _run_and_compare(prog, A_host, B_host, constants=None):
    """Run SDFG with and without AddThreadBlockMap and compare results."""
    import cupy
    # Prepare GPU arrays
    A_gpu = cupy.asarray(A_host)
    B_gpu = cupy.asarray(B_host)

    # Build SDFGs
    sdfg = prog.to_sdfg()

    # Count GPU_ThreadBlock maps
    threadblock_maps = [
        m for m, _ in sdfg.all_nodes_recursive()
        if isinstance(m, dace.sdfg.nodes.MapEntry)
        and m.map.schedule == dace.dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert len(threadblock_maps) == 0, f"Expected 0 GPU_ThreadBlock map before transformation, got {len(threadblock_maps)}"


    sdfg_expanded = copy.deepcopy(sdfg)

    # Apply AddThreadBlockMap transformation
    sdfg_expanded.apply_transformations_once_everywhere(AddThreadBlockMap)


    # Count GPU_ThreadBlock maps
    threadblock_maps = [
        m for m, _ in sdfg_expanded.all_nodes_recursive()
        if isinstance(m, dace.sdfg.nodes.MapEntry)
        and m.map.schedule == dace.dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert len(threadblock_maps) == 1, f"Expected 1 GPU_ThreadBlock map after transformation, got {len(threadblock_maps)}"

    # Run both SDFGs
    A_ref = cupy.zeros_like(A_gpu)
    args = copy.deepcopy(constants)
    args["A"] = A_ref
    args["B"] = B_gpu
    sdfg(**args)
    A_out = cupy.zeros_like(A_gpu)
    args = copy.deepcopy(constants)
    args["A"] = A_out
    args["B"] = B_gpu
    sdfg_expanded(**args)

    # Compare results on host
    numpy.testing.assert_allclose(cupy.asnumpy(A_ref), cupy.asnumpy(A_out))


@pytest.mark.gpu
def test_elementwise_constexpr_size():
    A = numpy.zeros(512, dtype=numpy.float64)
    B = numpy.random.rand(512).astype(numpy.float64)
    _run_and_compare(elementwise_constexpr_size, A, B)


@pytest.mark.gpu
def test_elementwise_small_constexpr_size():
    A = numpy.zeros(16, dtype=numpy.float64)
    B = numpy.random.rand(16).astype(numpy.float64)
    _run_and_compare(elementwise_small_constexpr_size, A, B)


symbol_params = [16, 32, 512]
@pytest.mark.gpu
@pytest.mark.parametrize("symbol_param", symbol_params)
def test_elementwise_symbolic(symbol_param):
    A = numpy.zeros(symbol_param, dtype=numpy.float64)
    B = numpy.random.rand(symbol_param).astype(numpy.float64)
    _run_and_compare(elementwise_symbolic, A, B, constants={"N": symbol_param})

@pytest.mark.gpu
def test_elementwise_with_floor_div():
    A = numpy.zeros(512, dtype=numpy.float64)
    B = numpy.random.rand(512).astype(numpy.float64)
    _run_and_compare(elementwise_with_floor_div, A, B, constants={"i_beg": 0, "i_end": 1024})


if __name__ == "__main__":
    test_elementwise_constexpr_size()
    test_elementwise_small_constexpr_size()
    test_elementwise_symbolic()
    test_elementwise_with_floor_div()