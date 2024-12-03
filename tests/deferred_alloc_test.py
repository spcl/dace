import dace
from dace.transformation.dataflow.redundant_array import RedundantArray, RedundantSecondArray
from dace.transformation.interstate.state_fusion import StateFusion
import numpy
import cupy
import pytest

@pytest.fixture(params=[dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.GPU_Global])
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

def _get_trivial_alloc_sdfg(storage_type: dace.dtypes.StorageType, transient: bool, write_size="0:2"):
    sdfg = dace.sdfg.SDFG(name="deferred_alloc_test")

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
    sdfg = dace.sdfg.SDFG(name="deferred_alloc_test_4")

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
        assert (schedule_type == dace.dtypes.ScheduleType.Sequential)
    elif storage_type == dace.dtypes.StorageType.GPU_Global:
        assert (schedule_type == dace.dtypes.ScheduleType.GPU_Device)

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


def _valid_to_reallocate(transient, storage_type, scope):
    return transient and (storage_type == dace.dtypes.StorageType.GPU_Global or storage_type == dace.dtypes.StorageType.CPU_Heap)

def test_trivial_realloc(storage_type: dace.dtypes.StorageType, transient: bool):
    sdfg = _get_trivial_alloc_sdfg(storage_type, transient)
    try:
        sdfg.validate()
    except Exception:
        if not _valid_to_reallocate(transient, storage_type, None):
            return
        else:
            raise AssertionError("Realloc with transient data failed when it was expected not to.")

    if not _valid_to_reallocate(transient, storage_type, None):
        raise AssertionError("Realloc with non-transient data did not fail when it was expected to.")

    sdfg.compile()

    sdfg.simplify()
    sdfg.apply_transformations_repeated([StateFusion, RedundantArray, RedundantSecondArray])
    sdfg.validate()
    sdfg.compile()

def test_realloc_use(storage_type: dace.dtypes.StorageType, transient: bool, schedule_type: dace.dtypes.ScheduleType):
    sdfg = _get_assign_map_sdfg(storage_type, transient, schedule_type)
    try:
        sdfg.validate()
    except Exception:
        if not _valid_to_reallocate(transient, storage_type, None):
            return
        else:
            raise AssertionError("Realloc-use with transient data failed when it was expected not to.")

    if not _valid_to_reallocate(transient, storage_type, None):
        raise AssertionError("Realloc-use with non-transient data did not fail when it was expected to.")

    compiled_sdfg = sdfg.compile()
    if storage_type == dace.dtypes.StorageType.CPU_Heap:
        arr = numpy.array([-1.0]).astype(numpy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr[0] == 3.0 )
    if storage_type == dace.dtypes.StorageType.GPU_Global:
        arr = cupy.array([-1.0]).astype(cupy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr.get()[0] == 3.0 )

    sdfg.simplify()
    sdfg.apply_transformations_repeated([StateFusion, RedundantArray, RedundantSecondArray])
    sdfg.validate()
    compiled_sdfg = sdfg.compile()
    if storage_type == dace.dtypes.StorageType.CPU_Heap:
        arr = numpy.array([-1.0]).astype(numpy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr[0] == 3.0 )
    if storage_type == dace.dtypes.StorageType.GPU_Global:
        arr = cupy.array([-1.0]).astype(cupy.float32)
        user_size = numpy.array([10, 10]).astype(numpy.uint64)
        compiled_sdfg(user_size=user_size, example_array=arr)
        assert ( arr.get()[0] == 3.0 )

def test_realloc_inside_map():
    pass


def test_all_combinations(storage_type, transient, schedule_type):
    test_trivial_realloc(storage_type, transient)
    test_realloc_use(storage_type, transient, schedule_type)

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
    for storage_type, schedule_type in [(dace.dtypes.StorageType.CPU_Heap, dace.dtypes.ScheduleType.Sequential),
                                        (dace.dtypes.StorageType.GPU_Global, dace.dtypes.ScheduleType.GPU_Device)]:
        print(f"Trivial Realloc with storage {storage_type}")
        test_trivial_realloc(storage_type, True)
        print(f"Trivial Realloc-Use with storage {storage_type}")
        test_realloc_use(storage_type, True, schedule_type)

    for storage_type, schedule_type in [(dace.dtypes.StorageType.CPU_Heap, dace.dtypes.ScheduleType.Sequential),
                                        (dace.dtypes.StorageType.GPU_Global, dace.dtypes.ScheduleType.GPU_Device)]:
        print(f"Trivial Realloc with storage {storage_type} on non-transient data")
        test_trivial_realloc(storage_type, False)
        print(f"Trivial Realloc-Use with storage {storage_type} on non-transient data")
        test_realloc_use(storage_type, False, schedule_type)

    # Try some other combinations
    for transient in [True, False]:
        for storage_type, schedule_type in [(dace.dtypes.StorageType.CPU_Heap, dace.dtypes.ScheduleType.Sequential),
                                            (dace.dtypes.StorageType.GPU_Global, dace.dtypes.ScheduleType.GPU_Device)]:
            print(f"Trivial Realloc with storage {storage_type} on transient:{transient} data")
            test_trivial_realloc(storage_type, transient)
            print(f"Trivial Realloc-Use with storage {storage_type} on transient:{transient} data")
            test_realloc_use(storage_type, transient, schedule_type)

    print(f"Realloc with incomplete write 1")
    test_incomplete_write_dimensions_1()
    print(f"Realloc with incomplete write 2")
    test_incomplete_write_dimensions_2()