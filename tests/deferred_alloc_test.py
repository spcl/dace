import dace
from dace.transformation.dataflow.redundant_array import RedundantArray, RedundantSecondArray
from dace.transformation.interstate.state_fusion import StateFusion
import numpy
import pytest


@pytest.fixture(params=[dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.GPU_Global, dace.dtypes.StorageType.CPU_Pinned])
def storage_type(request):
    return request.param

@pytest.fixture(params=[True, False])
def transient(request):
    return request.param

@pytest.fixture
def schedule_type(storage_type):
    if storage_type == dace.dtypes.StorageType.CPU_Heap:
        return dace.dtypes.ScheduleType.Sequential
    elif storage_type == dace.dtypes.StorageType.GPU_Global:
        return dace.dtypes.ScheduleType.GPU_Device
    elif storage_type == dace.dtypes.StorageType.CPU_Pinned:
        return dace.dtypes.ScheduleType.Sequential

def _get_trivial_alloc_sdfg(storage_type: dace.dtypes.StorageType, transient: bool, write_size="0:2"):
    sdfg = dace.sdfg.SDFG(name=f"deferred_alloc_test_1")

    sdfg.add_array(name="A", shape=(15, "__dace_defer"), dtype=dace.float32, storage=storage_type, transient=transient)

    state = sdfg.add_state("main")

    an_1 = state.add_access('A')
    an_1.add_in_connector('_write_size')

    sdfg.add_array(name="user_size", shape=(2,), dtype=dace.uint64)
    an_2 = state.add_access("user_size")

    state.add_edge(an_2, None, an_1, '_write_size',
                dace.Memlet(expr=f"user_size[{write_size}]") )

    return sdfg

def _get_assign_map_sdfg(storage_type: dace.dtypes.StorageType, transient: bool, schedule_type: dace.dtypes.ScheduleType.Default):
    sdfg = dace.sdfg.SDFG(name=f"deferred_alloc_test_2")

    sdfg.add_array(name="A", shape=(15, "__dace_defer"), dtype=dace.float32, storage=storage_type,
                    lifetime=dace.dtypes.AllocationLifetime.SDFG, transient=transient)

    state = sdfg.add_state("main")

    an_1 = state.add_access('A')
    an_1.add_in_connector('_write_size')
    an_1.add_out_connector('_read_size')

    sdfg.add_array(name="user_size", shape=(2,), dtype=dace.uint64)
    an_2 = state.add_access("user_size")

    state.add_edge(an_2, None, an_1, '_write_size',
                dace.Memlet(expr="user_size[0:2]") )

    map_entry, map_exit = state.add_map(name="map",ndrange={"i":dace.subsets.Range([(0,15-1,1)]),"j":dace.subsets.Range([(0,"__A_dim1_size-1", 1)]) },
                                        schedule=schedule_type)
    state.add_edge(an_1, '_read_size', map_entry, "__A_dim1_size", dace.Memlet(expr="A_size[1]"))
    map_entry.add_in_connector("__A_dim1_size")
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")

    t1 = state.add_tasklet(name="assign", inputs={}, outputs={"_out"}, code="_out=3.0")
    state.add_edge(map_entry, None, t1, None, dace.Memlet(None))
    state.add_edge(t1, "_out", map_exit, "IN_A", dace.Memlet(expr="A[i, j]"))

    an_3 = state.add_access('A')
    state.add_edge(map_exit, "OUT_A", an_3, None, dace.Memlet(data="A", subset=dace.subsets.Range([(0,15-1, 1), (0,"__A_dim1_size-1", 1)])))

    arr_name, arr = sdfg.add_array(name="example_array", dtype=dace.float32, shape=(1,), transient=False, storage=storage_type)
    arrn = state.add_access(arr_name)

    if storage_type == dace.dtypes.StorageType.CPU_Heap:
        assert (schedule_type == dace.dtypes.ScheduleType.Sequential or schedule_type == dace.dtypes.ScheduleType.CPU_Multicore)
    elif storage_type == dace.dtypes.StorageType.GPU_Global:
        assert (schedule_type == dace.dtypes.ScheduleType.GPU_Device)
    elif storage_type == dace.dtypes.StorageType.CPU_Pinned:
        assert (schedule_type == dace.dtypes.ScheduleType.Sequential)

    an_3.add_out_connector('_read_size')
    map_entry2, map_exit2 = state.add_map(name="map2",ndrange={"i":dace.subsets.Range([(0,15-1,1)]),"j":dace.subsets.Range([(0,"__A_dim1_size-1", 1)])},
                                            schedule=schedule_type)
    state.add_edge(an_3, '_read_size', map_entry2, "__A_dim1_size", dace.Memlet(expr="A_size[1]"))
    state.add_edge(an_3, None, map_entry2, "IN_A", dace.Memlet(expr="A[0:15, 0:__A_dim1_size]"))
    map_entry2.add_in_connector("__A_dim1_size")
    map_entry2.add_in_connector("IN_A")
    map_entry2.add_out_connector("OUT_A")
    map_exit2.add_in_connector("IN_A")
    map_exit2.add_out_connector("OUT_A")

    t2 = state.add_tasklet(name="check", inputs={"_in"}, outputs={"_out"}, code='_out = _in', language=dace.dtypes.Language.Python)
    state.add_edge(map_entry2, "OUT_A", t2, "_in", dace.Memlet(expr="A[i, j]"))
    state.add_edge(t2, "_out", map_exit2, "IN_A", dace.Memlet(expr="A[i, j]"))

    an_5 = state.add_access('A')
    state.add_edge(map_exit2, "OUT_A", an_5, None, dace.Memlet(data="A", subset=dace.subsets.Range([(0,15-1, 1), (0,"__A_dim1_size-1", 1)])))

    state.add_edge(an_5, None, arrn, None, dace.memlet.Memlet("A[7, 7]"))

    return sdfg

def _valid_to_reallocate(transient, storage_type):
    return transient and (storage_type in dace.dtypes.REALLOCATABLE_STORAGES)

def _test_trivial_realloc(storage_type: dace.dtypes.StorageType, transient: bool):
    sdfg = _get_trivial_alloc_sdfg(storage_type, transient)
    try:
        sdfg.validate()
    except Exception:
        if not _valid_to_reallocate(transient, storage_type):
            return
        else:
            raise AssertionError("Realloc with transient data failed when it was expected not to.")

    if not _valid_to_reallocate(transient, storage_type):
        raise AssertionError("Realloc with non-transient data did not fail when it was expected to.")

    sdfg.compile()

    sdfg.simplify()
    sdfg.apply_transformations_repeated([StateFusion, RedundantArray, RedundantSecondArray])
    sdfg.validate()
    sdfg.compile()


def _test_realloc_use(storage_type: dace.dtypes.StorageType, transient: bool, schedule_type: dace.dtypes.ScheduleType):
    sdfg = _get_assign_map_sdfg(storage_type, transient, schedule_type)
    try:
        sdfg.validate()
    except Exception:
        if not _valid_to_reallocate(transient, storage_type):
            return
        else:
            raise AssertionError("Realloc-use with transient data failed when it was expected not to.")

    if not _valid_to_reallocate(transient, storage_type):
        raise AssertionError("Realloc-use with non-transient data did not fail when it was expected to.")

    compiled_sdfg = sdfg.compile()
    if storage_type == dace.dtypes.StorageType.CPU_Heap or storage_type == dace.dtypes.StorageType.CPU_Pinned:
        arr = numpy.array([-1.0]).astype(numpy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr[0] == 3.0 )
    elif storage_type == dace.dtypes.StorageType.GPU_Global:
        try:
            import cupy
        except Exception:
            return

        arr = cupy.array([-1.0]).astype(cupy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr.get()[0] == 3.0 )

    sdfg.simplify()
    sdfg.apply_transformations_repeated([StateFusion, RedundantArray, RedundantSecondArray])
    sdfg.validate()
    compiled_sdfg = sdfg.compile()
    if storage_type == dace.dtypes.StorageType.CPU_Heap or storage_type == dace.dtypes.StorageType.CPU_Pinned:
        arr = numpy.array([-1.0]).astype(numpy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr[0] == 3.0 )
    elif storage_type == dace.dtypes.StorageType.GPU_Global:
        try:
            import cupy
        except Exception:
            return

        arr = cupy.array([-1.0]).astype(cupy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr.get()[0] == 3.0 )

@pytest.mark.gpu
def test_realloc_use_gpu(transient: bool):
    _test_realloc_use(dace.dtypes.StorageType.GPU_Global, transient, dace.dtypes.ScheduleType.GPU_Device)

def test_realloc_use_cpu(transient: bool):
    _test_realloc_use(dace.dtypes.StorageType.CPU_Heap, transient, dace.dtypes.ScheduleType.Sequential)

@pytest.mark.gpu
def test_realloc_use_cpu_pinned(transient: bool):
    _test_realloc_use(dace.dtypes.StorageType.CPU_Pinned, transient, dace.dtypes.ScheduleType.Sequential)

@pytest.mark.gpu
def test_trivial_realloc_gpu(transient: bool):
    _test_trivial_realloc(dace.dtypes.StorageType.GPU_Global, transient)

@pytest.mark.gpu
def test_trivial_realloc_cpu_pinned(transient: bool):
    _test_trivial_realloc(dace.dtypes.StorageType.CPU_Pinned, transient)

def test_trivial_realloc_cpu(transient: bool):
    _test_trivial_realloc(dace.dtypes.StorageType.CPU_Heap, transient)


def _add_realloc_inside_map(sdfg: dace.SDFG, schedule_type: dace.dtypes.ScheduleType):
    pre_state = sdfg.states()[0]
    state = sdfg.add_state("s2")
    sdfg.add_edge(pre_state, state, dace.InterstateEdge(None, None))

    map_entry, map_exit = state.add_map(name="map2",ndrange={"i":dace.subsets.Range([(0,4,1)])},
                                        schedule=schedule_type)
    an_2 = state.add_access('A')
    an_2.add_in_connector("_write_size")

    t1 = state.add_tasklet(name="assign", inputs={}, outputs={"__out"}, code="_out=8")
    t1.add_out_connector("__out")

    _, _ = sdfg.add_array("tmp0", shape=(2, ), dtype=numpy.uint64, transient=True)
    sca = state.add_access("tmp0")

    state.add_edge(map_entry, None, t1, None, dace.Memlet(None))
    state.add_edge(t1, "__out", sca, None, dace.Memlet("tmp0[0]"))
    state.add_edge(sca, None, an_2, "_write_size", dace.Memlet("tmp0"))
    state.add_edge(an_2, None, map_exit, None, dace.Memlet(None))

def test_realloc_inside_map_gpu():
    sdfg =_get_assign_map_sdfg(dace.dtypes.StorageType.GPU_Global, True, dace.dtypes.ScheduleType.GPU_Device)
    _add_realloc_inside_map(sdfg, dace.dtypes.ScheduleType.GPU_Device)
    try:
        sdfg.validate()
    except Exception:
        return

    pytest.fail("Realloc-use with non-transient data and incomplete write did not fail when it was expected to.")

def test_realloc_inside_map_cpu_pinned():
    sdfg =_get_assign_map_sdfg(dace.dtypes.StorageType.CPU_Pinned, True, dace.dtypes.ScheduleType.Sequential)
    _add_realloc_inside_map(sdfg, dace.dtypes.ScheduleType.Sequential)
    try:
        sdfg.validate()
    except Exception:
        return

    pytest.fail("Realloc-use with non-transient data and incomplete write did not fail when it was expected to.")


def test_realloc_inside_map_cpu():
    sdfg =_get_assign_map_sdfg(dace.dtypes.StorageType.CPU_Heap, True, dace.dtypes.ScheduleType.CPU_Multicore)
    _add_realloc_inside_map(sdfg, dace.dtypes.ScheduleType.CPU_Multicore)
    try:
        sdfg.validate()
    except Exception:
        return

    pytest.fail("Realloc-use with non-transient data and incomplete write did not fail when it was expected to.")

def _get_conditional_alloc_sdfg(storage_type: dace.dtypes.StorageType, transient: bool, schedule_type: dace.dtypes.ScheduleType, defer_expr_instead_of_symbol: bool = False):
    sdfg = dace.sdfg.SDFG(name=f"deferred_alloc_test_2")

    if not defer_expr_instead_of_symbol:
        sdfg.add_array(name="A", shape=("__dace_defer", "__dace_defer"), dtype=dace.float32, storage=storage_type,
                        lifetime=dace.dtypes.AllocationLifetime.SDFG, transient=transient)
    else:
        sdfg.add_array(name="A", shape=("4 * __dace_defer", "8 * __dace_defer"), dtype=dace.float32, storage=storage_type,
                        lifetime=dace.dtypes.AllocationLifetime.SDFG, transient=transient)

    sdfg.add_scalar(name="path", transient=False, dtype=numpy.uint64)

    start = sdfg.add_state("s1")
    iftrue = sdfg.add_state("s1_0")
    iffalse = sdfg.add_state("s1_1")
    assigntrue = sdfg.add_state("s2_0")
    assignfalse = sdfg.add_state("s2_1")
    state = sdfg.add_state("s3")

    sdfg.add_edge(start, iftrue, dace.InterstateEdge("path == 1"))
    sdfg.add_edge(start, iffalse, dace.InterstateEdge("path != 1"))
    sdfg.add_edge(iftrue, assigntrue, dace.InterstateEdge(None))
    sdfg.add_edge(iffalse, assignfalse, dace.InterstateEdge(None))
    sdfg.add_edge(assigntrue, state, dace.InterstateEdge(None))
    sdfg.add_edge(assignfalse, state, dace.InterstateEdge(None))

    s1name, s1 = sdfg.add_array(name="size1", shape=(2,), dtype=numpy.uint64, storage=dace.dtypes.StorageType.Register,
                    lifetime=dace.dtypes.AllocationLifetime.SDFG, transient=False)
    s2name, s2 = sdfg.add_array(name="size2", shape=(2,), dtype=numpy.uint64, storage=dace.dtypes.StorageType.Register,
                    lifetime=dace.dtypes.AllocationLifetime.SDFG, transient=False)

    an_2_0 = assigntrue.add_access('A')
    an_2_0.add_in_connector('_write_size')
    an_u_2_0 = assigntrue.add_access("size1")
    assigntrue.add_edge(an_u_2_0, None, an_2_0, "_write_size", dace.memlet.Memlet("size1"))

    an_2_1 = assignfalse.add_access('A')
    an_2_1.add_in_connector('_write_size')
    an_u_2_1 = assignfalse.add_access("size2")
    assignfalse.add_edge(an_u_2_1, None, an_2_1, "_write_size", dace.memlet.Memlet("size2"))

    if storage_type == dace.dtypes.StorageType.CPU_Heap:
        assert (schedule_type == dace.dtypes.ScheduleType.Sequential or schedule_type == dace.dtypes.ScheduleType.CPU_Multicore)
    elif storage_type == dace.dtypes.StorageType.GPU_Global:
        assert (schedule_type == dace.dtypes.ScheduleType.GPU_Device)
    elif storage_type == dace.dtypes.StorageType.CPU_Pinned:
        assert (schedule_type == dace.dtypes.ScheduleType.Sequential)

    an_3 = state.add_access('A')
    an_3.add_out_connector('_read_size')
    map_entry, map_exit = state.add_map(name="map",ndrange={"i":dace.subsets.Range([(0,"__A_0-1",1)]),
                                                            "j":dace.subsets.Range([(0,"__A_1-1", 1)])},
                                            schedule=schedule_type)
    state.add_edge(an_3, '_read_size', map_entry, "__A_0", dace.Memlet(expr="A_size[0]"))
    state.add_edge(an_3, '_read_size', map_entry, "__A_1", dace.Memlet(expr="A_size[1]"))
    map_entry.add_in_connector("__A_0")
    map_entry.add_in_connector("__A_1")
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")

    t1 = state.add_tasklet(name="assign", inputs={}, outputs={"_out"}, code="_out=3.0")
    state.add_edge(map_entry, None, t1, None, dace.Memlet(None))
    state.add_edge(t1, "_out", map_exit, "IN_A", dace.Memlet(expr="A[i, j]"))

    an_4 = state.add_access('A')
    state.add_edge(map_exit, "OUT_A", an_4, None, dace.Memlet(data="A", subset=dace.subsets.Range([(0,"__A_0-1", 1), (0,"__A_1-1", 1)])))

    an_4.add_out_connector('_read_size')
    map_entry2, map_exit2 = state.add_map(name="map2",ndrange={"i":dace.subsets.Range([(0,"__A_0-1",1)]),"j":dace.subsets.Range([(0,"__A_1-1", 1)])},
                                            schedule=schedule_type)
    state.add_edge(an_4, '_read_size', map_entry2, "__A_0", dace.Memlet(expr="A_size[0]"))
    state.add_edge(an_4, '_read_size', map_entry2, "__A_1", dace.Memlet(expr="A_size[1]"))
    state.add_edge(an_4, None, map_entry2, "IN_A", dace.Memlet(expr="A[0:__A_0, 0:__A_1]"))
    map_entry2.add_in_connector("__A_0")
    map_entry2.add_in_connector("__A_1")
    map_entry2.add_in_connector("IN_A")
    map_entry2.add_out_connector("OUT_A")
    map_exit2.add_in_connector("IN_A")
    map_exit2.add_out_connector("OUT_A")

    t2 = state.add_tasklet(name="check", inputs={"_in"}, outputs={"_out"}, code='_out = _in', language=dace.dtypes.Language.Python)
    state.add_edge(map_entry2, "OUT_A", t2, "_in", dace.Memlet(expr="A[i, j]"))
    state.add_edge(t2, "_out", map_exit2, "IN_A", dace.Memlet(expr="A[i, j]"))

    an_5 = state.add_access('A')
    state.add_edge(map_exit2, "OUT_A", an_5, None, dace.Memlet(data="A", subset=dace.subsets.Range([(0,"__A_0-1", 1), (0,"__A_1-1", 1)])))

    arr_name, arr = sdfg.add_array(name="example_array", dtype=dace.float32, shape=(1,), transient=False, storage=storage_type)
    arrn = state.add_access(arr_name)
    state.add_edge(an_5, None, arrn, None, dace.memlet.Memlet("A[0, 0]"))

    return sdfg

@pytest.mark.gpu
def test_conditional_alloc_gpu():
    sdfg =_get_conditional_alloc_sdfg(dace.dtypes.StorageType.GPU_Global, True, dace.dtypes.ScheduleType.GPU_Device)
    sdfg.validate()
    size1 = numpy.array([1, 1]).astype(numpy.uint64)
    size2 = numpy.array([22, 22]).astype(numpy.uint64)
    try:
        import cupy
    except Exception:
        return

    arr = cupy.array([-1.0]).astype(cupy.float32)
    sdfg(path=1, size1=size1, size2=size2, example_array=arr)
    assert ( arr.get()[0] == 3.0 )

@pytest.mark.gpu
def test_conditional_alloc_cpu_pinned():
    sdfg =_get_conditional_alloc_sdfg(dace.dtypes.StorageType.CPU_Pinned, True, dace.dtypes.ScheduleType.Sequential)
    sdfg.validate()
    size1 = numpy.array([1, 1]).astype(numpy.uint64)
    size2 = numpy.array([22, 22]).astype(numpy.uint64)
    arr = numpy.array([-1.0]).astype(numpy.float32)
    sdfg(path=1, size1=size1, size2=size2, example_array=arr)
    assert ( arr[0] == 3.0 )

def test_conditional_alloc_cpu():
    sdfg =_get_conditional_alloc_sdfg(dace.dtypes.StorageType.CPU_Heap, True, dace.dtypes.ScheduleType.CPU_Multicore)
    sdfg.validate()
    size1 = numpy.array([1, 1]).astype(numpy.uint64)
    size2 = numpy.array([22, 22]).astype(numpy.uint64)
    arr = numpy.array([-1.0]).astype(numpy.float32)
    sdfg(path=0, size1=size1, size2=size2, example_array=arr)
    assert ( arr[0] == 3.0 )

@pytest.mark.gpu
def test_conditional_alloc_with_expr_gpu():
    sdfg =_get_conditional_alloc_sdfg(dace.dtypes.StorageType.GPU_Global, True, dace.dtypes.ScheduleType.GPU_Device, True)
    sdfg.validate()
    size1 = numpy.array([1, 1]).astype(numpy.uint64)
    size2 = numpy.array([22, 22]).astype(numpy.uint64)
    try:
        import cupy
    except Exception:
        return

    arr = cupy.array([-1.0]).astype(cupy.float32)
    sdfg(path=1, size1=size1, size2=size2, example_array=arr)
    assert ( arr.get()[0] == 3.0 )

@pytest.mark.gpu
def test_conditional_alloc_with_expr_cpu_pinned():
    sdfg =_get_conditional_alloc_sdfg(dace.dtypes.StorageType.CPU_Pinned, True, dace.dtypes.ScheduleType.Sequential, True)
    sdfg.validate()
    size1 = numpy.array([1, 1]).astype(numpy.uint64)
    size2 = numpy.array([22, 22]).astype(numpy.uint64)
    arr = numpy.array([-1.0]).astype(numpy.float32)
    sdfg(path=1, size1=size1, size2=size2, example_array=arr)
    assert ( arr[0] == 3.0 )

def test_conditional_alloc_with_expr_cpu():
    sdfg =_get_conditional_alloc_sdfg(dace.dtypes.StorageType.CPU_Heap, True, dace.dtypes.ScheduleType.CPU_Multicore, True)
    sdfg.validate()
    size1 = numpy.array([1, 1]).astype(numpy.uint64)
    size2 = numpy.array([22, 22]).astype(numpy.uint64)
    arr = numpy.array([-1.0]).astype(numpy.float32)
    sdfg(path=0, size1=size1, size2=size2, example_array=arr)
    assert ( arr[0] == 3.0 )

def test_incomplete_write_dimensions_1():
    sdfg = _get_trivial_alloc_sdfg(dace.dtypes.StorageType.CPU_Heap, True, "1:2")
    try:
        sdfg.validate()
    except Exception:
        return

    pytest.fail("Realloc-use with transient data and incomplete write did not fail when it was expected to.")

def test_incomplete_write_dimensions_2():
    sdfg = _get_trivial_alloc_sdfg(dace.dtypes.StorageType.CPU_Heap, False, "1:2")
    try:
        sdfg.validate()
    except Exception:
        return

    pytest.fail("Realloc-use with non-transient data and incomplete write did not fail when it was expected to.")


if __name__ == "__main__":
    print(f"Trivial Realloc within map, cpu")
    test_realloc_inside_map_cpu()
    print(f"Trivial Realloc within map, gpu")
    test_realloc_inside_map_gpu()
    print(f"Trivial Realloc within map, cpu pinned")
    test_realloc_inside_map_cpu_pinned()

    print(f"Trivial Realloc with storage, cpu")
    test_trivial_realloc_cpu(True)
    print(f"Trivial Realloc with storage, gpu")
    test_trivial_realloc_gpu(True)
    print(f"Trivial Realloc with storage, cpu pinned")
    test_trivial_realloc_cpu_pinned(True)

    print(f"Trivial Realloc with storage, cpu, on non-transient data")
    test_trivial_realloc_cpu(False)
    print(f"Trivial Realloc-Use with storage, gpu, on non-transient data")
    test_trivial_realloc_gpu(False)
    print(f"Trivial Realloc with storage, cpu pinned, on non-transient data")
    test_trivial_realloc_cpu_pinned(False)

    print(f"Trivial Realloc-Use with storage, cpu")
    test_realloc_use_cpu(True)
    print(f"Trivial Realloc-Use with storage, gpu")
    test_realloc_use_gpu(True)
    print(f"Trivial Realloc-Use with storage, cpu pinned")
    test_realloc_use_cpu_pinned(True)
    print(f"Realloc with incomplete write one, validation")
    test_incomplete_write_dimensions_1()
    print(f"Realloc with incomplete write two, validation")
    test_incomplete_write_dimensions_2()

    print(f"Test conditional alloc with use, cpu")
    test_conditional_alloc_cpu()
    print(f"Test conditional alloc with use, gpu")
    test_conditional_alloc_gpu()
    print(f"Test conditional alloc with use, cpu pinned")
    test_conditional_alloc_cpu_pinned()

    print(f"Test conditional alloc with use and the shape as a non-trivial expression, cpu")
    test_conditional_alloc_with_expr_cpu()
    print(f"Test conditional alloc with use and the shape as a non-trivial expression, gpu")
    test_conditional_alloc_with_expr_gpu()
    print(f"Test conditional alloc with use and the shape as a non-trivial expression, cpu pinned")
    test_conditional_alloc_with_expr_cpu_pinned()
