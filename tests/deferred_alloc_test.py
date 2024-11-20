import dace
import numpy

def _get_trivial_alloc_sdfg(storage_type: dace.dtypes.StorageType, transient: bool, write_size="0:2"):
    sdfg = dace.sdfg.SDFG(name="deferred_alloc_test")

    sdfg.add_array(name="A", shape=(15, "__dace_defer"), dtype=dace.float32, storage=storage_type, transient=transient)

    state = sdfg.add_state("main")

    an_1 = state.add_access('A')
    an_1.add_in_connector('_write_size')

    an_2 = state.add_array(name="user_size", shape=(2,), dtype=numpy.uint64)

    state.add_edge(an_2, None, an_1, '_write_size',
                dace.Memlet(expr=f"user_size[{write_size}]") )

    sdfg.save("def_alloc_1.sdfg")

    return sdfg


def _get_assign_map_sdfg(storage_type: dace.dtypes.StorageType, transient: bool):
    sdfg = dace.sdfg.SDFG(name="deferred_alloc_test_4")

    sdfg.add_array(name="A", shape=(15, "__dace_defer"), dtype=dace.float32, storage=storage_type,
                    lifetime=dace.dtypes.AllocationLifetime.SDFG, transient=transient)

    state = sdfg.add_state("main")

    an_1 = state.add_access('A')
    an_1.add_in_connector('_write_size')
    an_1.add_out_connector('_read_size')

    an_2 = state.add_array(name="user_size", shape=(2,), dtype=numpy.uint64)

    state.add_edge(an_2, None, an_1, '_write_size',
                dace.Memlet(expr="user_size[0:2]") )

    map_entry, map_exit = state.add_map(name="map",ndrange={"i":dace.subsets.Range([(0,15-1,1)]),"j":dace.subsets.Range([(0,"__A_dim1_size-1", 1)]) })
    state.add_edge(an_1, '_read_size', map_entry, "__A_dim1_size", dace.Memlet(expr="A_size[1]"))
    map_entry.add_in_connector("__A_dim1_size")
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")

    t1 = state.add_tasklet(name="assign", inputs={}, outputs={"_out"}, code="_out=3.0")
    state.add_edge(map_entry, None, t1, None, dace.Memlet(None))
    state.add_edge(t1, "_out", map_exit, "IN_A", dace.Memlet(expr="A[i, j]"))

    an_3 = state.add_access('A')
    state.add_edge(map_exit, "OUT_A", an_3, None, dace.Memlet(data="A", subset=dace.subsets.Range([(0,15-1, 1), (0,"__A_dim1_size-1", 1)])))

    an_3.add_out_connector('_read_size')
    map_entry2, map_exit2 = state.add_map(name="map2",ndrange={"i":dace.subsets.Range([(0,15-1,1)]),"j":dace.subsets.Range([(0,"__A_dim1_size-1", 1)]) })
    state.add_edge(an_3, '_read_size', map_entry2, "__A_dim1_size", dace.Memlet(expr="A_size[1]"))
    state.add_edge(an_3, None, map_entry2, "IN_A", dace.Memlet(expr="A[0:15, 0:__A_dim1_size]"))
    map_entry2.add_in_connector("__A_dim1_size")
    map_entry2.add_in_connector("IN_A")
    map_entry2.add_out_connector("OUT_A")
    map_exit2.add_in_connector("IN_A")
    map_exit2.add_out_connector("OUT_A")

    t2 = state.add_tasklet(name="check", inputs={"_in"}, outputs={"_out"}, code='if (_in != 5.0){ throw std::runtime_error("fail"); } \n _out=_in;', language=dace.dtypes.Language.CPP)
    state.add_edge(map_entry2, "OUT_A", t2, "_in", dace.Memlet(expr="A[i, j]"))
    state.add_edge(t2, "_out", map_exit2, "IN_A", dace.Memlet(expr="A[i, j]"))

    an_5 = state.add_access('A')
    state.add_edge(map_exit2, "OUT_A", an_5, None, dace.Memlet(data="A", subset=dace.subsets.Range([(0,15-1, 1), (0,"__A_dim1_size-1", 1)])))

    sdfg.save("def_alloc_4.sdfg")

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

def test_realloc_use(storage_type: dace.dtypes.StorageType, transient: bool):
    sdfg = _get_assign_map_sdfg(storage_type, transient)
    try:
        sdfg.validate()
    except Exception:
        if not _valid_to_reallocate(transient, storage_type, None):
            return
        else:
            raise AssertionError("Realloc-use with transient data failed when it was expected not to.")

    if not _valid_to_reallocate(transient, storage_type, None):
        raise AssertionError("Realloc-use with non-transient data did not fail when it was expected to.")

    sdfg.compile()

def test_incomplete_write_dimensions_1():
    sdfg =  _get_trivial_alloc_sdfg(dace.dtypes.StorageType.CPU_Heap, True, "1:2")
    try:
        sdfg.validate()
    except Exception:
        return

    raise AssertionError("Realloc-use with transient data and incomplete write did not fail when it was expected to.")

def test_incomplete_write_dimensions_2():
    sdfg =  _get_trivial_alloc_sdfg(dace.dtypes.StorageType.CPU_Heap, False, "1:2")
    try:
        sdfg.validate()
    except Exception:
        return

    raise AssertionError("Realloc-use with non-transient data and incomplete write did not fail when it was expected to.")

def test_realloc_inside_map():
    pass

if __name__ == "__main__":
    for storage_type in [dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.GPU_Global]:
        print(f"Trivial Realloc with storage {storage_type}")
        test_trivial_realloc(storage_type, True)
        print(f"Trivial Realloc-Use with storage {storage_type}")
        test_realloc_use(storage_type, True)

    for storage_type in [dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.GPU_Global]:
        print(f"Trivial Realloc with storage {storage_type} on non-transient data")
        test_trivial_realloc(storage_type, False)
        print(f"Trivial Realloc-Use with storage {storage_type} on non-transient data")
        test_realloc_use(storage_type, False)

    # Try some other combinations
    for transient in [True, False]:
        for storage_type in [dace.dtypes.StorageType.Default, dace.dtypes.StorageType.Register]:
            print(f"Trivial Realloc with storage {storage_type} on transient:{transient} data")
            test_trivial_realloc(storage_type, transient)
            print(f"Trivial Realloc-Use with storage {storage_type} on transient:{transient} data")
            test_realloc_use(storage_type, transient)

    print(f"Realloc with incomplete write 1")
    test_incomplete_write_dimensions_1()
    print(f"Realloc with incomplete write 2")
    test_incomplete_write_dimensions_2()