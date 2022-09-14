# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import dace
import numpy as np
from dace.transformation.dataflow import MapTiling, OutLocalStorage
from dace.transformation.dataflow.local_storage import InLocalStorage

import dace.transformation.helpers as xfh

N = dace.symbol('N')


@dace.program
def copy_sdfg(A: dace.float32[N, N], B: dace.float32[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a << A[i, j]
            b >> B[i, j]
            b = a


def find_map_entries(sdfg):
    outer_map_entry = None
    inner_map_entry = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue

        if xfh.get_parent_map(sdfg.start_state, node) is None:
            assert outer_map_entry is None
            outer_map_entry = node
        else:
            assert inner_map_entry is None
            inner_map_entry = node
    assert not outer_map_entry is None
    assert not inner_map_entry is None

    return outer_map_entry, inner_map_entry


def test_in_local_storage_explicit():
    sdfg = copy_sdfg.to_sdfg()
    sdfg.simplify()

    sdfg.apply_transformations([MapTiling], options=[{"tile_sizes": [8]}])

    outer_map_entry, inner_map_entry = find_map_entries(sdfg)

    InLocalStorage.apply_to(sdfg=sdfg,
                            node_a=outer_map_entry,
                            node_b=inner_map_entry,
                            options={
                                "array": "A",
                                "create_array": True,
                                "prefix": "loc_"
                            },
                            save=True)

    # Finding relevant node
    local_storage_node = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.AccessNode):
            continue

        if node.data == "loc_A":
            assert local_storage_node is None
            local_storage_node = node
            break

    assert not local_storage_node is None

    # Check transient array created
    trans_array = local_storage_node.data
    assert trans_array in sdfg.arrays

    # Check properties
    desc = sdfg.arrays[local_storage_node.data]
    assert desc.shape == (8, 8)
    assert desc.transient == True

    # Check array was set correctly
    serialized = sdfg.transformation_hist[0].to_json()
    assert serialized["array"] == "A"


def test_in_local_storage_implicit():
    sdfg = copy_sdfg.to_sdfg()
    sdfg.simplify()

    sdfg.apply_transformations([MapTiling], options=[{"tile_sizes": [8]}])

    outer_map_entry, inner_map_entry = find_map_entries(sdfg)

    InLocalStorage.apply_to(sdfg=sdfg,
                            node_a=outer_map_entry,
                            node_b=inner_map_entry,
                            options={
                                "create_array": True,
                                "prefix": "loc_"
                            },
                            save=True)

    # Finding relevant node
    local_storage_node = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.AccessNode):
            continue

        if node.data == "loc_A":
            assert local_storage_node is None
            local_storage_node = node
            break

    assert not local_storage_node is None

    # Check transient array created
    trans_array = local_storage_node.data
    assert trans_array in sdfg.arrays

    # Check properties
    desc = sdfg.arrays[local_storage_node.data]
    assert desc.shape == (8, 8)
    assert desc.transient == True

    # Check array was set correctly
    serialized = sdfg.transformation_hist[0].to_json()
    assert serialized["array"] == None


def test_out_local_storage_explicit():
    sdfg = copy_sdfg.to_sdfg()
    sdfg.simplify()

    sdfg.apply_transformations([MapTiling], options=[{"tile_sizes": [8]}])

    outer_map_entry, inner_map_entry = find_map_entries(sdfg)
    outer_map_exit = sdfg.start_state.exit_node(outer_map_entry)
    inner_map_exit = sdfg.start_state.exit_node(inner_map_entry)

    OutLocalStorage.apply_to(sdfg=sdfg,
                             node_a=inner_map_exit,
                             node_b=outer_map_exit,
                             options={
                                 "array": "B",
                                 "create_array": True,
                                 "prefix": "loc_"
                             },
                             save=True)

    # Finding relevant node
    local_storage_node = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.AccessNode):
            continue

        if node.data == "loc_B":
            assert local_storage_node is None
            local_storage_node = node
            break

    assert not local_storage_node is None

    # Check transient array created
    trans_array = local_storage_node.data
    assert trans_array in sdfg.arrays

    # Check properties
    desc = sdfg.arrays[local_storage_node.data]
    assert desc.shape == (8, 8)
    assert desc.transient == True

    # Check array was set correctly
    serialized = sdfg.transformation_hist[0].to_json()
    assert serialized["array"] == "B"


def test_out_local_storage_implicit():
    sdfg = copy_sdfg.to_sdfg()
    sdfg.simplify()

    sdfg.apply_transformations([MapTiling], options=[{"tile_sizes": [8]}])

    outer_map_entry, inner_map_entry = find_map_entries(sdfg)
    outer_map_exit = sdfg.start_state.exit_node(outer_map_entry)
    inner_map_exit = sdfg.start_state.exit_node(inner_map_entry)

    OutLocalStorage.apply_to(sdfg=sdfg,
                             node_a=inner_map_exit,
                             node_b=outer_map_exit,
                             options={
                                 "create_array": True,
                                 "prefix": "loc_"
                             },
                             save=True)

    # Finding relevant node
    local_storage_node = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.AccessNode):
            continue

        if node.data == "loc_B":
            assert local_storage_node is None
            local_storage_node = node
            break

    assert not local_storage_node is None

    # Check transient array created
    trans_array = local_storage_node.data
    assert trans_array in sdfg.arrays

    # Check properties
    desc = sdfg.arrays[local_storage_node.data]
    assert desc.shape == (8, 8)
    assert desc.transient == True

    # Check array was set correctly
    serialized = sdfg.transformation_hist[0].to_json()
    assert serialized["array"] == None


@dace.program
def arange():
    out = np.ndarray([N], np.int32)
    for i in dace.map[0:N]:
        with dace.tasklet:
            o >> out[i]
            o = i
    return out


class LocalStorageTests(unittest.TestCase):

    def test_even(self):
        sdfg = arange.to_sdfg()
        sdfg.apply_transformations([MapTiling, OutLocalStorage], options=[{'tile_sizes': [8]}, {}])
        self.assertTrue(np.array_equal(sdfg(N=16), np.arange(16, dtype=np.int32)))

    def test_uneven(self):
        # For testing uneven decomposition, use longer buffer and ensure
        # it's not filled over
        output = np.ones(20, np.int32)
        sdfg = arange.to_sdfg()
        sdfg.apply_transformations([MapTiling, OutLocalStorage], options=[{'tile_sizes': [5]}, {}])
        dace.propagate_memlets_sdfg(sdfg)
        sdfg(N=16, __return=output)
        self.assertTrue(np.array_equal(output[:16], np.arange(16, dtype=np.int32)))
        self.assertTrue(np.array_equal(output[16:], np.ones(4, np.int32)))


if __name__ == '__main__':
    unittest.main()
    test_in_local_storage_explicit()
    test_in_local_storage_implicit()
    test_out_local_storage_explicit()
    test_out_local_storage_implicit()
