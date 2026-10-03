# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import dace
from dace.sdfg.validation import InvalidSDFGEdgeError
import pytest


def test_add_scope_connectors():
    sdfg = dace.SDFG("add_scope_connectors_sdfg")
    state = sdfg.add_state(is_start_block=True)
    me: dace.nodes.MapEntry
    mx: dace.nodes.MapExit
    me, mx = state.add_map("test_map", ndrange={"__i0": "0:10"})
    assert all(len(mn.in_connectors) == 0 and len(mn.out_connectors) == 0 for mn in [me, mx])
    me.add_in_connector("IN_T", dtype=dace.float64)
    assert len(me.in_connectors) == 1 and me.in_connectors["IN_T"] is dace.float64 and len(me.out_connectors) == 0
    assert len(mx.in_connectors) == 0 and len(mx.out_connectors) == 0

    # Because there is already an `IN_T` this call will fail.
    assert not me.add_scope_connectors("T")
    assert len(me.in_connectors) == 1 and me.in_connectors["IN_T"] is dace.float64 and len(me.out_connectors) == 0
    assert len(mx.in_connectors) == 0 and len(mx.out_connectors) == 0

    # Now it will work, because we specify force, however, the current type for `IN_T` will be overridden.
    assert me.add_scope_connectors("T", force=True)
    assert len(me.in_connectors) == 1 and me.in_connectors["IN_T"].type is None
    assert len(me.out_connectors) == 1 and me.out_connectors["OUT_T"].type is None
    assert len(mx.in_connectors) == 0 and len(mx.out_connectors) == 0

    # Now tries to the full adding.
    assert mx.add_scope_connectors("B", dtype=dace.int64)
    assert len(mx.in_connectors) == 1 and mx.in_connectors["IN_B"] is dace.int64
    assert len(mx.out_connectors) == 1 and mx.out_connectors["OUT_B"] is dace.int64


def test_invalid_empty_memlet():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    t, me, mx = state.add_mapped_tasklet('test_map',
                                         dict(i='0:20'), {'inp': dace.Memlet('A[i]')},
                                         'out = inp + 1', {'out': dace.Memlet('B[i]')},
                                         external_edges=True)
    unused = state.add_access('A')
    state.add_nedge(me, unused, dace.Memlet())
    e = state.add_nedge(unused, t, dace.Memlet())

    # This should pass
    sdfg.validate()

    # Add invalid empty memlet
    state.remove_edge(e)
    state.add_edge(unused, None, t, 'something', dace.Memlet())
    t.add_in_connector('something')

    with pytest.raises(InvalidSDFGEdgeError, match='Empty memlet connected to tasklet'):
        sdfg.validate()


def test_access_node_deepcopy_keeps_every_property():
    """``AccessNode.__deepcopy__`` assigns fields by hand; none may be missed.

    The hand-written copy is a hot-path shortcut over the base class's copy of ``__dict__``, so a
    Property added to ``Node`` afterwards is not copied unless someone remembers to add a line --
    and reading an unassigned Property raises ``AttributeError`` rather than returning its default.
    ``specialization_hint`` went missing exactly that way and took the whole canonicalize pipeline
    with it, since it deepcopies graph parts constantly.
    """
    sdfg = dace.SDFG("access_node_deepcopy")
    sdfg.add_array("a", [8], dace.float64)
    state = sdfg.add_state()
    node = state.add_access("a")

    clone = copy.deepcopy(node)

    missing = [name for name in type(node).__properties__ if f"_{name}" not in vars(clone)]
    assert not missing, f"deepcopy dropped {missing}"
    # Reading them must not raise, and the values must survive.
    assert clone.data == node.data
    assert clone.specialization_hint == node.specialization_hint
    assert clone.guid != node.guid, "a copy is a new element and needs its own id"


def test_access_node_deepcopy_carries_a_set_property():
    """A property with a VALUE must arrive on the copy, not just exist on it."""
    sdfg = dace.SDFG("access_node_deepcopy_value")
    sdfg.add_array("a", [8], dace.float64)
    state = sdfg.add_state()
    node = state.add_access("a")
    node.specialization_hint = "kept sequential on the host; a device pass could tile it"

    assert copy.deepcopy(node).specialization_hint == node.specialization_hint


if __name__ == "__main__":
    test_add_scope_connectors()
    test_invalid_empty_memlet()
    test_access_node_deepcopy_keeps_every_property()
    test_access_node_deepcopy_carries_a_set_property()
