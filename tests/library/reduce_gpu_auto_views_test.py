# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPUAuto reduce expansion reads an operand's storage and class from the container, not from an alias.

Expanding a library node needs neither a GPU nor a compiler, so these run anywhere.
"""
from collections.abc import Iterator

import dace
from dace import data, dtypes
from dace.libraries.standard.nodes import Reduce
from dace.sdfg import nodes


def gpu_auto_reduce_sdfg(name: str, view_storage: dtypes.StorageType | None = None) -> dace.SDFG:
    """A rank-4 sum over the last axis of a device array, read through an ``ArrayView`` of ``view_storage`` if given."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [2, 3, 4, 5], dace.float32, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [2, 3, 4], dace.float32, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()

    source, source_memlet = state.add_read('A'), dace.Memlet('A[0:2, 0:3, 0:4, 0:5]')
    if view_storage is not None:
        sdfg.add_view('A_view', [2, 3, 4, 5], dace.float32, storage=view_storage)
        view = state.add_access('A_view')
        state.add_edge(source, None, view, 'views', source_memlet)
        source, source_memlet = view, dace.Memlet('A_view[0:2, 0:3, 0:4, 0:5]')

    reduce_node = Reduce('reduce', wcr='lambda a, b: a + b', axes=[3], identity=0)
    reduce_node.implementation = 'GPUAuto'
    reduce_node.add_in_connector('_in')
    reduce_node.add_out_connector('_out')
    state.add_node(reduce_node)
    state.add_edge(source, None, reduce_node, '_in', source_memlet)
    state.add_edge(reduce_node, '_out', state.add_write('B'), None, dace.Memlet('B[0:2, 0:3, 0:4]'))
    sdfg.validate()
    return sdfg


def nested_descriptors(sdfg: dace.SDFG, name: str) -> Iterator[data.Data]:
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                if name in node.sdfg.arrays:
                    yield node.sdfg.arrays[name]
                yield from nested_descriptors(node.sdfg, name)


def map_schedules(sdfg: dace.SDFG) -> set[dtypes.ScheduleType]:
    return {node.map.schedule for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)}


def test_reduce_reading_a_view_gets_a_plain_array_inside():
    """A cloned ``ArrayView`` descriptor is a view of nothing inside the nested SDFG, which validation refuses."""
    sdfg = gpu_auto_reduce_sdfg('gpu_auto_reduce_views_class', dtypes.StorageType.GPU_Global)
    sdfg.expand_library_nodes()
    sdfg.validate()

    inner = list(nested_descriptors(sdfg, '_in'))
    assert inner, 'the GPUAuto expansion did not run'
    assert not any(isinstance(desc, data.View) for desc in inner)


def test_reduce_reads_storage_through_the_view():
    """A view of a device array is a device operand, whatever its own descriptor declares."""
    # A view owns no storage, so Default is what one looks like before anything has matched it to its container.
    sdfg = gpu_auto_reduce_sdfg('gpu_auto_reduce_views_storage', dtypes.StorageType.Default)
    sdfg.expand_library_nodes()

    assert dtypes.ScheduleType.GPU_Device in map_schedules(sdfg)


def test_reduce_reshaped_input_keeps_offset_at_the_new_rank():
    sdfg = gpu_auto_reduce_sdfg('gpu_auto_reduce_views_offset')
    sdfg.expand_library_nodes()

    inner = list(nested_descriptors(sdfg, '_in'))
    assert inner, 'the GPUAuto expansion did not run'
    for desc in inner:
        desc.validate()
        assert len(desc.offset) == len(desc.shape) == len(desc.strides)


if __name__ == '__main__':
    test_reduce_reading_a_view_gets_a_plain_array_inside()
    test_reduce_reads_storage_through_the_view()
    test_reduce_reshaped_input_keeps_offset_at_the_new_rank()
