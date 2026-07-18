# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for ``dace.data.Structure`` member access in the nextgen frontend:
member reads/writes use dotted data paths (``tracers.data``) resolved through
the base Structure (the only registered container), and assignments to
``Reference``-typed members re-point them via :class:`RefSetNode`.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

TRACER_SIZE = dace.symbol('TRACER_SIZE')
NUMBER_OF_TRACERS = dace.symbol('NUMBER_OF_TRACERS')
MyTracers = dace.data.Structure(members={
    "data":
    dace.data.Array(dace.float32, (TRACER_SIZE, NUMBER_OF_TRACERS), strides=(1, TRACER_SIZE)),
    "vapor":
    dace.data.ArrayReference(dace.float32, (TRACER_SIZE, )),
    "ice":
    dace.data.ArrayReference(dace.float32, (TRACER_SIZE, )),
},
                                name="Tracer_Bundle")


@dace.program
def initialize_all(tracers: MyTracers, fill_value: int) -> None:
    tracers.vapor = tracers.data[:, 2]
    tracers.ice = tracers.data[:, 0]

    # loop over all arrays and initialize them with `fill_value`
    for index in range(NUMBER_OF_TRACERS):
        tracers.data[:, index] = fill_value

    # let's assume this one is different for some reason
    tracers.vapor[:] = fill_value + 1


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_structure_members():
    tree = nextgen.parse_program(initialize_all)

    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    refsets = _nodes_of_type(tree, tn.RefSetNode)
    assert sorted(node.target for node in refsets) == ['tracers.ice', 'tracers.vapor']
    assert all(node.memlet.data == 'tracers.data' for node in refsets)

    # Only the base Structure is registered; members resolve through it.
    assert sorted(tree.containers) == ['fill_value', 'tracers']
    assert 'tracers.data' in tree.containers  # NestedDict member resolution

    # The fill loop writes into the 'data' member, the final write into 'vapor'.
    writes = [
        memlet.data for tasklet in _nodes_of_type(tree, tn.TaskletNode) for memlet in tasklet.out_memlets.values()
    ]
    assert 'tracers.data' in writes
    assert 'tracers.vapor' in writes


def test_structure_vs_old_frontend():
    tree = nextgen.parse_program(initialize_all)
    old_tree = initialize_all.to_schedule_tree()

    for root in (tree, old_tree):
        assert not _nodes_of_type(root, tn.PythonCallbackNode)
        assert len(_nodes_of_type(root, tn.RefSetNode)) == 2
        # Same container convention: base structure registered, members as
        # dotted paths only. (The old frontend additionally materializes the
        # loop variable 'index' as a scalar container; nextgen binds it as a
        # symbol.)
        assert {'tracers', 'fill_value'} <= set(root.containers)
        assert not any('.' in name for name in root.containers)


def test_structure_unknown_member():

    @dace.program
    def unknown_member(tracers: MyTracers, out: dace.float32[10]):
        out[0] = tracers.nonexistent

    tree = nextgen.parse_program(unknown_member)
    # Unknown members are a feature gap: interpreter fallback, not a crash.
    assert len(_nodes_of_type(tree, tn.PythonCallbackNode)) == 1


def test_structure_execution():
    tree = nextgen.parse_program(initialize_all)
    sdfg = tree.as_sdfg()
    func = sdfg.compile()

    tracer_size, tracer_count = 5, 4
    # 'data' has strides (1, TRACER_SIZE): column-major layout
    arr = np.zeros((tracer_size, tracer_count), dtype=np.float32, order='F')
    tracers = MyTracers.dtype._typeclass.as_ctypes()(data=arr.__array_interface__['data'][0])
    func(tracers=tracers, fill_value=7, TRACER_SIZE=tracer_size, NUMBER_OF_TRACERS=tracer_count)

    # Fill writes 7 everywhere; the final write goes through the 'vapor'
    # reference, which was re-pointed to column 2 before the fill.
    expected = np.full((tracer_size, tracer_count), 7, dtype=np.float32)
    expected[:, 2] = 8
    assert np.allclose(arr, expected)


if __name__ == '__main__':
    test_structure_members()
    test_structure_vs_old_frontend()
    test_structure_unknown_member()
    test_structure_execution()
