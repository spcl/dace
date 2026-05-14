# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression guard: ``fix_nsdfg_connector_array_shapes_mismatch`` must raise
when the proposed rebuild would EXPAND the connector (any new dim is larger
than the corresponding original dim). The historical behaviour silently
rebuilt with ``expected_shape_collapsed_full`` whatever the input — corrupting
inner-SDFG accesses when the inner code was sized for the smaller shape.

The narrowing case (cloudsc-class: connector declared as the FULL outer-array
shape, memlet subset is a smaller slice → rebuild to ``collapsed_full`` is
legitimate) remains accepted.

Was xfailed until the pass-through-subsets redesign; flipped to a regular
test once ``_raise_on_expansion_rebuild_mismatch`` was added as the guard.
"""
import pytest
import dace
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import fix_nsdfg_connector_array_shapes_mismatch


def test_shape_mismatch_should_raise_not_silently_rebuild():
    """Original connector ``(3, 4)`` + parent memlet ``arr[0:5, 0:9]`` →
    no expected-shape interpretation matches, and the candidate rebuild
    ``(5, 9)`` would EXPAND each dim past the original. Must raise."""
    inner_sdfg = dace.SDFG("test_shape_mismatch_expand_inner")
    inner_sdfg.add_array("conn", shape=(3, 4), dtype=dace.float64, transient=False)
    inner_state = inner_sdfg.add_state("s", is_start_block=True)
    inner_state.add_access("conn")  # connector must be "live"

    parent_sdfg = dace.SDFG("test_shape_mismatch_expand_parent")
    parent_sdfg.add_array("arr", shape=(10, 10), dtype=dace.float64)
    parent_state = parent_sdfg.add_state("p", is_start_block=True)
    parent_an = parent_state.add_access("arr")
    nsdfg_node = parent_state.add_nested_sdfg(inner_sdfg, {"conn"}, set())
    parent_state.add_edge(parent_an, None, nsdfg_node, "conn", dace.memlet.Memlet("arr[0:5, 0:9]"))

    with pytest.raises(ValueError, match="would EXPAND a non-placeholder dim"):
        fix_nsdfg_connector_array_shapes_mismatch(parent_state, nsdfg_node)


def test_shape_mismatch_narrowing_still_accepted():
    """Cloudsc-class shape: connector ``(10, 10)``, parent memlet
    ``arr[0:5, 0:9]`` → rebuild to ``(5, 9)`` narrows each dim. The
    guard must NOT trip; rebuild proceeds silently."""
    inner_sdfg = dace.SDFG("test_shape_mismatch_narrow_inner")
    # Connector oversized vs the memlet — typical cloudsc shape.
    inner_sdfg.add_array("conn", shape=(10, 10), dtype=dace.float64, transient=False)
    inner_state = inner_sdfg.add_state("s", is_start_block=True)
    inner_state.add_access("conn")

    parent_sdfg = dace.SDFG("test_shape_mismatch_narrow_parent")
    parent_sdfg.add_array("arr", shape=(10, 10), dtype=dace.float64)
    parent_state = parent_sdfg.add_state("p", is_start_block=True)
    parent_an = parent_state.add_access("arr")
    nsdfg_node = parent_state.add_nested_sdfg(inner_sdfg, {"conn"}, set())
    parent_state.add_edge(parent_an, None, nsdfg_node, "conn", dace.memlet.Memlet("arr[0:5, 0:9]"))

    # Should NOT raise — narrowing is legitimate.
    fix_nsdfg_connector_array_shapes_mismatch(parent_state, nsdfg_node)
    # Rebuilt connector shape is the narrowed (5, 9).
    new_shape = nsdfg_node.sdfg.arrays["conn"].shape
    assert tuple(int(d) for d in new_shape) == (5, 9), \
        f"narrowing rebuild produced unexpected shape {new_shape}"
