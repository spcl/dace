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
from dace import dtypes
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
