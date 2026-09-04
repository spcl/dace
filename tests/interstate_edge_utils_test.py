import dace
import typing
import pytest


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


def test_writing_to_scalar_on_iedge_is_invalid():
    # SDFG can't write to scalars on interstate edges catch for validity
    with pytest.raises(dace.sdfg.validation.InvalidSDFGInterstateEdgeError,
                       match="Assignment to a scalar or an array detected in an interstate edge"):
        sdfg_and_edge: typing.Tuple[dace.SDFG, dace.InterstateEdge] = _get_sdfg()
        sdfg: dace.SDFG = sdfg_and_edge[0]
        sdfg.validate()


def _get_sdfg_writing_to_scalar_inside_a_region() -> dace.SDFG:
    """The offending assignment sits on an interstate edge INSIDE a loop, not at the top level."""
    sdfg = dace.SDFG("interstate_write_in_region")
    sdfg.add_scalar("scalar1", dace.int32, transient=True, find_new_name=False)
    sdfg.add_symbol("symbol1", dace.int32, find_new_name=False)

    entry_state = sdfg.add_state("entry", is_start_block=True)
    loop = dace.sdfg.state.LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1", sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(entry_state, loop, dace.InterstateEdge())

    body1 = loop.add_state("body1", is_start_block=True)
    body2 = loop.add_state("body2")
    loop.add_edge(body1, body2, dace.InterstateEdge(assignments={"scalar1": "symbol1"}))
    sdfg.reset_cfg_list()
    return sdfg


def test_writing_to_scalar_on_iedge_inside_a_region_is_invalid():
    """The scalar/array-write check must run on every region, not only on the top-level SDFG.

    ``validate_control_flow_region`` recurses into nested regions but handed the check the SDFG
    every time, so the top-level edges were re-checked once per region while a region's own
    interstate edges were never checked at all.
    """
    sdfg = _get_sdfg_writing_to_scalar_inside_a_region()
    with pytest.raises(dace.sdfg.validation.InvalidSDFGInterstateEdgeError,
                       match="Assignment to a scalar or an array detected in an interstate edge"):
        sdfg.validate()


def test_region_check_does_not_reject_a_plain_symbol_assignment():
    """Only assignments whose target is a data descriptor are rejected; ordinary symbols are fine."""
    sdfg = dace.SDFG("interstate_symbol_in_region")
    sdfg.add_array("array1", (10, ), dace.int32, transient=True, find_new_name=False)
    sdfg.add_symbol("symbol1", dace.int32, find_new_name=False)

    entry_state = sdfg.add_state("entry", is_start_block=True)
    loop = dace.sdfg.state.LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1", sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(entry_state, loop, dace.InterstateEdge())

    body1 = loop.add_state("body1", is_start_block=True)
    body2 = loop.add_state("body2")
    loop.add_edge(body1, body2, dace.InterstateEdge(assignments={"symbol1": "i"}))
    sdfg.reset_cfg_list()
    sdfg.validate()


if __name__ == "__main__":
    test_read_symbols()
    test_used_symbols()
    test_all_used_symbols()
    test_all_read_sdfg_symbols()
    test_all_read_arrays()
    test_all_used_arrays()
    test_writing_to_scalar_on_iedge_is_invalid()
    test_writing_to_scalar_on_iedge_inside_a_region_is_invalid()
    test_region_check_does_not_reject_a_plain_symbol_assignment()
