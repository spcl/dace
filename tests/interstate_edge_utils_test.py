import dace
import typing


def _get_sdfg() -> typing.Tuple[dace.SDFG, dace.InterstateEdge]:
    sdfg = dace.SDFG("interstate_util_test")

    # Add symbols and arrays
    scalar1_name, scalar1 = sdfg.add_scalar("scalar1", dace.int32, transient=True, find_new_name=False)
    scalar2_name, scalar2 = sdfg.add_scalar("scalar2", dace.int32, transient=True, find_new_name=False)
    array1_name, array1 = sdfg.add_array("array1", (10, ), dace.int32, transient=True, find_new_name=False)
    sym1_name = sdfg.add_symbol("symbol1", dace.int32, find_new_name=False)
    sym2_name = sdfg.add_symbol("symbol2", dace.int32, find_new_name=False)
    sym3_name = sdfg.add_symbol("symbol3", dace.int32, find_new_name=False)
    sym4_name = sdfg.add_symbol("symbol4", dace.int32, find_new_name=False)

    # Add states and some init code
    state1 = sdfg.add_state("s1")
    state2 = sdfg.add_state("s2")
    a1 = state1.add_access(scalar1_name)
    a2 = state1.add_access(scalar2_name)
    t1 = state1.add_tasklet("tasklet1", {}, {"_out"}, "_out = 1")
    t2 = state1.add_tasklet("tasklet2", {}, {"_out"}, "_out = 2")
    state1.add_edge(t1, "_out", a1, None, dace.Memlet(f"{scalar1_name}"))
    state1.add_edge(t2, "_out", a2, None, dace.Memlet(f"{scalar2_name}"))

    # Add interstate edge with some assignments
    interstate_assignments = {
        scalar1_name: sym1_name,
        sym2_name: scalar2_name,
        sym3_name: f"{array1_name}[1]",
    }
    e = sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=interstate_assignments))
    sdfg.validate()
    return sdfg, e


def test_read_symbols():
    sdfg_and_edge: typing.Tuple[dace.SDFG, dace.InterstateEdge] = _get_sdfg()
    e: dace.InterstateEdge = sdfg_and_edge[1]
    assert e.data.read_symbols() == {"scalar2", "symbol1", "array1"}


def test_used_symbols():
    sdfg_and_edge: typing.Tuple[dace.SDFG, dace.InterstateEdge] = _get_sdfg()
    e: dace.InterstateEdge = sdfg_and_edge[1]
    assert e.data.used_symbols() == {"scalar2", "symbol1", "array1"}
    assert e.data.used_symbols(all_symbols=True) == e.data.used_symbols(all_symbols=False)


def test_all_used_symbols():
    sdfg_and_edge: typing.Tuple[dace.SDFG, dace.InterstateEdge] = _get_sdfg()
    e: dace.InterstateEdge = sdfg_and_edge[1]
    assert e.data.used_symbols(
        all_symbols=True, union_lhs_symbols=True) == {"scalar1", "scalar2", "symbol1", "symbol2", "symbol3", "array1"}
    assert e.data.used_symbols(all_symbols=False, union_lhs_symbols=True) == e.data.used_symbols(all_symbols=True,
                                                                                                 union_lhs_symbols=True)


def test_all_read_sdfg_symbols():
    sdfg_and_edge: typing.Tuple[dace.SDFG, dace.InterstateEdge] = _get_sdfg()
    sdfg: dace.SDFG = sdfg_and_edge[0]
    e: dace.InterstateEdge = sdfg_and_edge[1]
    assert e.data.used_sdfg_symbols(arrays=sdfg.arrays, union_lhs_symbols=False) == {"symbol1"}


def test_all_read_arrays():
    sdfg_and_edge: typing.Tuple[dace.SDFG, dace.InterstateEdge] = _get_sdfg()
    sdfg: dace.SDFG = sdfg_and_edge[0]
    e: dace.InterstateEdge = sdfg_and_edge[1]
    assert e.data.used_arrays(arrays=sdfg.arrays, union_lhs_symbols=False) == {"scalar2", "array1"}


def test_all_used_arrays():
    sdfg_and_edge: typing.Tuple[dace.SDFG, dace.InterstateEdge] = _get_sdfg()
    sdfg: dace.SDFG = sdfg_and_edge[0]
    e: dace.InterstateEdge = sdfg_and_edge[1]
    assert e.data.used_arrays(arrays=sdfg.arrays, union_lhs_symbols=True) == {"scalar2", "scalar1", "array1"}


if __name__ == "__main__":
    test_read_symbols()
    test_used_symbols()
    test_all_used_symbols()
    test_all_read_sdfg_symbols()
    test_all_read_arrays()
    test_all_used_arrays()
    print("All tests passed!")
