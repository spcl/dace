# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tripwire: ``fix_nsdfg_connector_array_shapes_mismatch`` silently rebuilds the
connector with ``expected_shape_collapsed_full`` whenever the original shape
matches *none* of the four expected interpretations
(``full`` / ``strided`` / ``collapsed_full`` / ``collapsed_strided``).

The function should raise instead so the caller knows the input is inconsistent.
Today it picks ``collapsed_full`` as a guess and proceeds, which can corrupt
inner-SDFG accesses or downstream codegen.

Expected to xfail until the planned pass-through-subsets redesign deletes
``fix_nsdfg_connector_array_shapes_mismatch`` entirely.
"""
import pytest
import dace
from dace.transformation.passes.vectorization.vectorization_utils import fix_nsdfg_connector_array_shapes_mismatch


@pytest.mark.xfail(reason="fix_nsdfg_connector_array_shapes_mismatch silently rebuilds with collapsed_full "
                   "on a no-match input; cloudsc-class callers legitimately rely on this rebuild path. "
                   "Planned to be deleted by the pass-through-subsets redesign.")
def test_shape_mismatch_should_raise_not_silently_rebuild():
    # Inner SDFG with a connector whose shape doesn't match the parent memlet's subset
    # under any of the 4 expected interpretations.
    inner_sdfg = dace.SDFG("inner")
    # Connector declared as (3, 4) — neither 'full' (would be (5, 9)), 'strided' ((5, 9)),
    # 'collapsed_full' ((5, 9)), nor 'collapsed_strided' ((5, 9)) for the subset below.
    inner_sdfg.add_array("conn", shape=(3, 4), dtype=dace.float64, transient=False)
    inner_state = inner_sdfg.add_state("s", is_start_block=True)
    # The inner SDFG must reference the connector or the function does nothing useful;
    # add a trivial access so the connector is "live".
    inner_state.add_access("conn")

    # Parent SDFG with a 2D array and a memlet that takes a 5x9 slice.
    parent_sdfg = dace.SDFG("parent")
    parent_sdfg.add_array("arr", shape=(10, 10), dtype=dace.float64)
    parent_state = parent_sdfg.add_state("p", is_start_block=True)
    parent_an = parent_state.add_access("arr")
    nsdfg_node = parent_state.add_nested_sdfg(inner_sdfg, {"conn"}, set())
    parent_state.add_edge(parent_an, None, nsdfg_node, "conn", dace.memlet.Memlet("arr[0:5, 0:9]"))

    # Today: silently rebuilds inner_sdfg.arrays['conn'] to expected_shape_collapsed_full=(5,9).
    # Wanted post-redesign: raises with the four expected shapes shown.
    with pytest.raises((NotImplementedError, ValueError, AssertionError)):
        fix_nsdfg_connector_array_shapes_mismatch(parent_state, nsdfg_node)
