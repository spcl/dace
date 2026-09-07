# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPUAuto reduce expansion must leave its ``_in`` descriptor in a readable state.

The expansion reshapes the input descriptor to the shape the reduction schedule wants. ``offset``
is rank-dependent, so assigning ``shape`` alone leaves it at the source array's rank: the
descriptor then fails its own ``validate`` with ``Offset must be the same size as shape``, and
deserialization answers with an unregistered placeholder instead of an ``Array``, which is how the
SDFG stops surviving a serialization round trip (``testing.serialization``, on in CI).

Expanding a library node needs neither a GPU nor a compiler, so this runs anywhere.
"""

import pytest

import dace
from dace import data, dtypes
from dace.sdfg import nodes
from dace.libraries.standard.nodes import Reduce


def gpu_auto_reduce_sdfg() -> dace.SDFG:
    """A rank-4 reduction over the last axis, which is where the reshape changes the rank."""
    sdfg = dace.SDFG('gpu_auto_reduce')
    sdfg.add_array('A', [2, 3, 4, 5], dace.float32, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [2, 3, 4], dace.float32, storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state()
    reduce_node = Reduce('reduce', wcr='lambda a, b: a + b', axes=[3], identity=0)
    reduce_node.implementation = 'GPUAuto'
    reduce_node.add_in_connector('_in')
    reduce_node.add_out_connector('_out')
    state.add_node(reduce_node)
    state.add_edge(state.add_read('A'), None, reduce_node, '_in', dace.Memlet('A[0:2, 0:3, 0:4, 0:5]'))
    state.add_edge(reduce_node, '_out', state.add_write('B'), None, dace.Memlet('B[0:2, 0:3, 0:4]'))

    sdfg.validate()
    return sdfg


def all_descriptors(sdfg: dace.SDFG):
    """Every data descriptor in ``sdfg``, including the ones the expansion nested inside it."""
    for name, desc in sdfg.arrays.items():
        yield sdfg.label, name, desc
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                yield from all_descriptors(node.sdfg)


def gpu_auto_reduce_from_view_sdfg() -> dace.SDFG:
    """The same reduction, but reading through an ``ArrayView`` rather than the array itself.

    lenet's second reduce has this shape: it reads a view of the maxpool input.
    """
    sdfg = dace.SDFG('gpu_auto_reduce_from_view')
    sdfg.add_array('A', [2, 3, 4, 5], dace.float32, storage=dtypes.StorageType.GPU_Global)
    # Declared ``Default`` on purpose: a view owns no storage, so this is what one looks like
    # before anything has matched it to the container it aliases.
    sdfg.add_view('A_view', [2, 3, 4, 5], dace.float32, storage=dtypes.StorageType.Default)
    sdfg.add_array('B', [2, 3, 4], dace.float32, storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state()
    view = state.add_access('A_view')
    state.add_edge(state.add_read('A'), None, view, 'views', dace.Memlet('A[0:2, 0:3, 0:4, 0:5]'))

    reduce_node = Reduce('reduce', wcr='lambda a, b: a + b', axes=[3], identity=0)
    reduce_node.implementation = 'GPUAuto'
    reduce_node.add_in_connector('_in')
    reduce_node.add_out_connector('_out')
    state.add_node(reduce_node)
    state.add_edge(view, None, reduce_node, '_in', dace.Memlet('A_view[0:2, 0:3, 0:4, 0:5]'))
    state.add_edge(reduce_node, '_out', state.add_write('B'), None, dace.Memlet('B[0:2, 0:3, 0:4]'))

    sdfg.validate()
    return sdfg


def test_the_expansion_leaves_every_descriptor_valid():
    sdfg = gpu_auto_reduce_sdfg()
    sdfg.expand_library_nodes()

    invalid = []
    for owner, name, desc in all_descriptors(sdfg):
        try:
            desc.validate()
        except TypeError as exc:
            invalid.append(f'{owner}.{name}: {exc}')
    assert not invalid, 'the expansion produced descriptors that cannot be read back: ' + '; '.join(invalid)


def test_the_reshaped_input_keeps_its_offset_at_the_new_rank():
    """The specific pairing the bug broke, named so a regression says what it broke."""
    sdfg = gpu_auto_reduce_sdfg()
    sdfg.expand_library_nodes()

    reshaped = [desc for _, name, desc in all_descriptors(sdfg) if name == '_in']
    assert reshaped, 'the GPUAuto expansion did not run, so this test is anchored on nothing'
    for desc in reshaped:
        assert len(desc.offset) == len(desc.shape), f'offset {desc.offset} does not match shape {desc.shape}'
        assert len(desc.strides) == len(desc.shape), f'strides {desc.strides} do not match shape {desc.shape}'


def test_a_reduce_reading_a_view_gets_a_plain_array_inside():
    """``_in`` is a buffer the nested SDFG reaches through a connector, never an alias.

    Cloning the caller's descriptor carried its CLASS too, so a reduce reading an ``ArrayView`` gave
    the nested SDFG an ``_in`` that views nothing -- read by several edges inside, which validation
    refuses with "Ambiguous or invalid edge to/from a View access node".
    """
    sdfg = gpu_auto_reduce_from_view_sdfg()
    sdfg.expand_library_nodes()
    sdfg.validate()

    inner = [desc for _, name, desc in all_descriptors(sdfg) if name == '_in']
    assert inner, 'the GPUAuto expansion did not run, so this test is anchored on nothing'
    for desc in inner:
        assert not isinstance(desc, data.View), f'_in came out as {type(desc).__name__}, an alias of nothing'


def test_the_expansion_reads_storage_through_the_view():
    """A view of a device array is a device operand, whatever the alias itself declares.

    Asserted on the SCHEDULES the expansion produced, because that is the decision the storage
    drives: read off the alias, the ``Default`` above says host and the reduce silently takes the
    Pure fallback -- left on the wrong side of the machine rather than failing outright. The
    descriptor cannot be asserted on instead, since type inference re-derives a nested connector's
    storage from the outer container.
    """
    sdfg = gpu_auto_reduce_from_view_sdfg()
    sdfg.expand_library_nodes()

    schedules = {node.map.schedule for _, state, node in every_map(sdfg)}
    assert dtypes.ScheduleType.GPU_Device in schedules, (
        f'the reduce did not expand for the device, so the alias was read rather than the array '
        f'it views (schedules: {sorted(str(s) for s in schedules)})')


def every_map(sdfg: dace.SDFG):
    """Every map entry in ``sdfg``, including the ones inside nested SDFGs."""
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                yield sdfg.label, state, node
            elif isinstance(node, nodes.NestedSDFG):
                yield from every_map(node.sdfg)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
