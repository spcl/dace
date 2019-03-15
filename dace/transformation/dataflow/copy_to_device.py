""" Contains classes and functions that implement copying a nested SDFG
    and its dependencies to a given device. """

import dace
from copy import deepcopy as dcpy
from dace import data, properties, symbolic, types, subsets
from dace.graph import edges, graph, nodes, nxutil
from dace.transformation import pattern_matching
from math import ceil
import sympy
import networkx as nx


def change_storage(sdfg, storage):
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                node.desc(sdfg).storage = storage
            if isinstance(node, nodes.NestedSDFG):
                change_storage(node.sdfg, storage)


@properties.make_properties
class CopyToDevice(pattern_matching.Transformation):
    """ Implements the copy-to-device transformation, which copies a nested
        SDFG and its dependencies to a given device.

        The transformation changes all data storage types of a nested SDFG to 
        the given `storage` property, and creates new arrays and copies around
        the nested SDFG to that storage.
    """

    _nested_sdfg = nodes.NestedSDFG("", graph.OrderedDiGraph(), set(), set())

    storage = properties.Property(
        dtype=types.StorageType,
        desc="Nested SDFG storage",
        enum=types.StorageType,
        from_string=lambda x: types.StorageType[x],
        default=types.StorageType.Default)

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(CopyToDevice._nested_sdfg)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        nested_sdfg = graph.nodes()[candidate[CopyToDevice._nested_sdfg]]
        return nested_sdfg.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.state_id]
        nested_sdfg = state.nodes()[self.subgraph[CopyToDevice._nested_sdfg]]
        storage = self.storage

        for _, edge in enumerate(state.in_edges(nested_sdfg)):

            src, src_conn, dst, dst_conn, memlet = edge
            dataname = memlet.data
            memdata = sdfg.arrays[dataname]

            if isinstance(memdata, data.Array):
                new_data = sdfg.add_array(
                    'device_' + dataname + '_in',
                    memdata.dtype, [
                        symbolic.overapproximate(r)
                        for r in memlet.bounding_box_size()
                    ],
                    transient=True,
                    storage=storage)
            elif isinstance(memdata, data.Scalar):
                new_data = sdfg.add_scalar(
                    'device_' + dataname + '_in',
                    memdata.dtype,
                    transient=True,
                    storage=storage)
            else:
                raise NotImplementedError

            data_node = nodes.AccessNode('device_' + dataname + '_in')

            to_data_mm = dcpy(memlet)
            from_data_mm = dcpy(memlet)
            from_data_mm.data = 'device_' + dataname + '_in'
            offset = []
            for ind, r in enumerate(memlet.subset):
                offset.append(r[0])
                if isinstance(memlet.subset[ind], tuple):
                    begin = memlet.subset[ind][0] - r[0]
                    end = memlet.subset[ind][1] - r[0]
                    step = memlet.subset[ind][2]
                    from_data_mm.subset[ind] = (begin, end, step)
                else:
                    from_data_mm.subset[ind] -= r[0]

            state.remove_edge(edge)
            state.add_edge(src, src_conn, data_node, None, to_data_mm)
            state.add_edge(data_node, None, dst, dst_conn, from_data_mm)

        for _, edge in enumerate(state.out_edges(nested_sdfg)):

            src, src_conn, dst, dst_conn, memlet = edge
            dataname = memlet.data
            memdata = sdfg.arrays[dataname]

            if isinstance(memdata, data.Array):
                new_data = data.Array(
                    'device_' + dataname + '_out',
                    memdata.dtype, [
                        symbolic.overapproximate(r)
                        for r in memlet.bounding_box_size()
                    ],
                    transient=True,
                    storage=storage)
            elif isinstance(memdata, data.Scalar):
                new_data = sdfg.add_scalar(
                    'device_' + dataname + '_out',
                    memdata.dtype,
                    transient=True,
                    storage=storage)
            else:
                raise NotImplementedError

            data_node = nodes.AccessNode('device_' + dataname + '_out')

            to_data_mm = dcpy(memlet)
            from_data_mm = dcpy(memlet)
            to_data_mm.data = 'device_' + dataname + '_out'
            offset = []
            for ind, r in enumerate(memlet.subset):
                offset.append(r[0])
                if isinstance(memlet.subset[ind], tuple):
                    begin = memlet.subset[ind][0] - r[0]
                    end = memlet.subset[ind][1] - r[0]
                    step = memlet.subset[ind][2]
                    to_data_mm.subset[ind] = (begin, end, step)
                else:
                    to_data_mm.subset[ind] -= r[0]

            state.remove_edge(edge)
            state.add_edge(src, src_conn, data_node, None, to_data_mm)
            state.add_edge(data_node, None, dst, dst_conn, from_data_mm)

        # Change storage for all data inside nested SDFG to device.
        change_storage(nested_sdfg.sdfg, storage)


pattern_matching.Transformation.register_pattern(CopyToDevice)
