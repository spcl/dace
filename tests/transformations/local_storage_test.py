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
    assert "array" not in serialized or serialized["array"] is None


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
    assert "array" not in serialized or serialized["array"] is None


def _nested_body_sdfg(shape):
    """A map body written as a nested SDFG whose connectors are the whole containers.

    Under the nested SDFG contract (see ``dace.sdfg.dealias.integrate_nested_sdfg``) the connectors
    are ``A`` and ``B`` themselves and the memlets inside are written in their coordinates, so a
    transformation that puts a local copy behind one of them has to move the memlets inside with it.
    """
    inner = dace.SDFG('body')
    inner.add_array('a', shape, dace.float64)
    inner.add_array('b', shape, dace.float64)
    inner.add_symbol('i', dace.int64)
    inner.add_symbol('j', dace.int64)
    state = inner.add_state()
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x * 2')
    state.add_edge(state.add_read('a'), None, tasklet, 'x', dace.Memlet('a[i, j]'))
    state.add_edge(tasklet, 'y', state.add_write('b'), None, dace.Memlet('b[i, j]'))
    return inner


def _nested_map_sdfg(shape):
    """``B[i, j] = 2 * A[i, j]`` over two nested maps, with a nested SDFG as the body."""
    sdfg = dace.SDFG('nested_body')
    sdfg.add_array('A', shape, dace.float64)
    sdfg.add_array('B', shape, dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', dict(i='0:%d' % shape[0]))
    inner_entry, inner_exit = state.add_map('inner', dict(j='0:%d' % shape[1]))
    node = state.add_nested_sdfg(_nested_body_sdfg(shape), {'a'}, {'b'}, {'i': 'i', 'j': 'j'})
    state.add_memlet_path(state.add_read('A'),
                          outer_entry,
                          inner_entry,
                          node,
                          dst_conn='a',
                          memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(node,
                          inner_exit,
                          outer_exit,
                          state.add_write('B'),
                          src_conn='b',
                          memlet=dace.Memlet('B[i, j]'))
    return sdfg, state, outer_entry, inner_entry, node


def test_in_local_storage_into_nested_sdfg():
    """The local copy is read by a nested SDFG: its connector and memlets follow the copy."""
    shape = (4, 5)
    sdfg, state, outer_entry, inner_entry, node = _nested_map_sdfg(shape)
    InLocalStorage.apply_to(sdfg, dict(array='A'), node_a=outer_entry, node_b=inner_entry, verify=False, save=False)

    edge = next(e for e in state.in_edges(node) if e.dst_conn == 'a')
    assert node.sdfg.arrays['a'].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()

    A = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape).copy()
    B = np.zeros(shape)
    sdfg(A=A, B=B)
    assert np.allclose(B, A * 2)


def test_out_local_storage_out_of_nested_sdfg():
    """The local copy is written by a nested SDFG: its connector and memlets follow the copy."""
    shape = (4, 5)
    sdfg, state, outer_entry, inner_entry, node = _nested_map_sdfg(shape)
    OutLocalStorage.apply_to(sdfg,
                             dict(array='B'),
                             node_a=state.exit_node(inner_entry),
                             node_b=state.exit_node(outer_entry),
                             verify=False,
                             save=False)

    edge = next(e for e in state.out_edges(node) if e.src_conn == 'b')
    assert node.sdfg.arrays['b'].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()

    A = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape).copy()
    B = np.zeros(shape)
    sdfg(A=A, B=B)
    assert np.allclose(B, A * 2)


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
    test_in_local_storage_explicit()
    test_in_local_storage_implicit()
    test_out_local_storage_explicit()
    test_out_local_storage_implicit()
    test_in_local_storage_into_nested_sdfg()
    test_out_local_storage_out_of_nested_sdfg()
    unittest.main()
