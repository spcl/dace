# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes and functions that implement copying a nested SDFG
    and its dependencies to a given device. """

from copy import deepcopy as dcpy
from dace import data, properties, symbolic, dtypes, registry
from dace.sdfg import graph, nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


def change_storage(sdfg, storage):
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                node.desc(sdfg).storage = storage
            if isinstance(node, nodes.NestedSDFG):
                change_storage(node.sdfg, storage)


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class CopyToDevice(transformation.Transformation):
    """ Implements the copy-to-device transformation, which copies a nested
        SDFG and its dependencies to a given device.

        The transformation changes all data storage types of a nested SDFG to
        the given `storage` property, and creates new arrays and copies around
        the nested SDFG to that storage.
    """

    _nested_sdfg = nodes.NestedSDFG("", graph.OrderedDiGraph(), {}, {})

    storage = properties.EnumProperty(dtype=dtypes.StorageType,
                                      desc="Nested SDFG storage",
                                      default=dtypes.StorageType.Default)

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(CopyToDevice._nested_sdfg)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        nested_sdfg = graph.nodes()[candidate[CopyToDevice._nested_sdfg]]

        for edge in graph.all_edges(nested_sdfg):
            # Stream inputs/outputs not allowed
            path = graph.memlet_path(edge)
            if ((isinstance(path[0].src, nodes.AccessNode) and isinstance(sdfg.arrays[path[0].src.data], data.Stream))
                    or (isinstance(path[-1].dst, nodes.AccessNode)
                        and isinstance(sdfg.arrays[path[-1].dst.data], data.Stream))):
                return False
            # WCR outputs with arrays are not allowed
            if (edge.data.wcr is not None and edge.data.subset.num_elements() != 1):
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        nested_sdfg = graph.nodes()[candidate[CopyToDevice._nested_sdfg]]
        return nested_sdfg.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.state_id]
        nested_sdfg = state.nodes()[self.subgraph[CopyToDevice._nested_sdfg]]
        storage = self.storage
        created_arrays = set()

        for _, edge in enumerate(state.in_edges(nested_sdfg)):

            src, src_conn, dst, dst_conn, memlet = edge
            dataname = memlet.data
            if dataname is None:
                continue
            memdata = sdfg.arrays[dataname]

            name = 'device_' + dataname + '_in'
            if name not in created_arrays:
                if isinstance(memdata, data.Array):
                    name, _ = sdfg.add_array('device_' + dataname + '_in',
                                             shape=[symbolic.overapproximate(r) for r in memlet.bounding_box_size()],
                                             dtype=memdata.dtype,
                                             transient=True,
                                             storage=storage,
                                             find_new_name=True)
                elif isinstance(memdata, data.Scalar):
                    name, _ = sdfg.add_scalar('device_' + dataname + '_in',
                                              dtype=memdata.dtype,
                                              transient=True,
                                              storage=storage,
                                              find_new_name=True)
                else:
                    raise NotImplementedError
                created_arrays.add(name)

            data_node = nodes.AccessNode(name)

            to_data_mm = dcpy(memlet)
            from_data_mm = dcpy(memlet)
            from_data_mm.data = name
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
            if dataname is None:
                continue
            memdata = sdfg.arrays[dataname]

            name = 'device_' + dataname + '_out'
            if name not in created_arrays:
                if isinstance(memdata, data.Array):
                    name, _ = sdfg.add_array(name,
                                             shape=[symbolic.overapproximate(r) for r in memlet.bounding_box_size()],
                                             dtype=memdata.dtype,
                                             transient=True,
                                             storage=storage,
                                             find_new_name=True)
                elif isinstance(memdata, data.Scalar):
                    name, _ = sdfg.add_scalar(name, dtype=memdata.dtype, transient=True, storage=storage)
                else:
                    raise NotImplementedError
                created_arrays.add(name)

            data_node = nodes.AccessNode(name)

            to_data_mm = dcpy(memlet)
            from_data_mm = dcpy(memlet)
            to_data_mm.data = name
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
