# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import dace

from collections import defaultdict
from dace.properties import make_properties
from dace.sdfg import utils as sdutil

from dace.transformation import transformation


@make_properties
class OnTheFlyMapFusion(transformation.SubgraphTransformation):
    """
    Performs fusion of two maps by replicating the contents of the first into the second map until all the
    input dependencies (memlets) of the second one are met.
    """
    def can_be_applied(self, state: dace.SDFGState, sdfg: dace.SDFG, parent_map_entry, child_map_entry):
        parent_map_exit = state.exit_node(parent_map_entry)
        nodes = sdutil.nodes_in_all_simple_paths(state, parent_map_exit, child_map_entry)
        nodes.remove(child_map_entry)
        nodes.remove(parent_map_exit)
        for node in nodes:
            if not isinstance(node, dace.nodes.AccessNode):
                return False

            srcs = state.in_edges(node)
            srcs = set(map(lambda edge: edge.src, srcs))
            if len(srcs) == 0:
                continue

            dests = state.out_edges(node)
            dests = set(map(lambda edge: edge.dst, dests))
            if len(srcs) > 1 or len(dests) > 1:
                return False

        return True

    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG, parent_map_entry, child_map_entry):
        parent_map_exit = state.exit_node(parent_map_entry)
        nodes = sdutil.nodes_in_all_simple_paths(state, parent_map_exit, child_map_entry)
        array_accesses = []
        for node in nodes:
            if not isinstance(node, dace.nodes.AccessNode):
                continue

            array_accesses.append(node)

        OnTheFlyMapFusion._update_map_connectors(state, parent_map_entry, child_map_entry, array_accesses)
        self._replicate_first_map(sdfg, parent_map_entry, parent_map_exit, child_map_entry, array_accesses)
        state.remove_nodes_from(
            state.all_nodes_between(parent_map_entry, parent_map_exit) | {parent_map_entry, parent_map_exit})

        for node in state.nodes():
            if not isinstance(node, dace.nodes.AccessNode):
                continue

            if state.in_degree(node) == 0 and state.out_degree(node) == 0:
                state.remove_node(node)

        child_map_entry.map.label = child_map_entry.map.label + "_OTF"
        print(child_map_entry.map.label)

    @staticmethod
    def _update_map_connectors(state, parent_map_entry, child_map_entry, array_accesses):
        for array_access in array_accesses:
            for edge in state.edges_between(array_access, child_map_entry):
                state.remove_edge_and_connectors(edge)

        for edge in state.in_edges(parent_map_entry):
            if child_map_entry.add_in_connector(edge.dst_conn + "_"):
                state.add_edge(edge.src, edge.src_conn, child_map_entry, edge.dst_conn + "_", edge.data)
            else:
                raise ValueError("Failed to connect")

    @staticmethod
    def _memlet_offsets(base_memlet, offset_memlet):
        """Compute subset offset of `offset_memlet` relative to `base_memlet`."""
        def offset(base_range, offset_range):
            b0, e0, s0 = base_range
            b1, e1, s1 = offset_range
            assert e1 - e0 == b1 - b0 and s0 == s1
            return int(e1 - e0)

        return tuple(offset(b, o) for b, o in zip(base_memlet.subset.ranges, offset_memlet.subset.ranges))

    @staticmethod
    def _read_offsets(state, array_name, first_map_exit, second_map_entry):
        """Compute offsets of read accesses in second map."""
        # Get output memlet of first tasklet
        output_edges = state.in_edges(first_map_exit)
        assert len(output_edges) == 1
        write_memlet = output_edges[0].data

        # Find read offsets by looping over second map entry connectors
        offsets = defaultdict(list)
        for edge in state.out_edges(second_map_entry):
            if edge.data.data == array_name:
                second_map_entry.remove_out_connector(edge.src_conn)
                state.remove_edge(edge)
                offset = OnTheFlyMapFusion._memlet_offsets(write_memlet, edge.data)
                offsets[offset].append(edge)

        return offsets

    @staticmethod
    def _copy_first_map_contents(sdfg, state, first_map_entry, first_map_exit):
        nodes = list(state.all_nodes_between(first_map_entry, first_map_exit) - {first_map_entry})
        new_nodes = [copy.deepcopy(node) for node in nodes]
        tmp_map = dict()
        for node in new_nodes:
            if isinstance(node, dace.nodes.AccessNode):
                data = sdfg.arrays[node.data]
                if isinstance(data, dace.data.Scalar) and data.transient:
                    tmp_name = sdfg.temp_data_name()
                    sdfg.add_scalar(tmp_name, data.dtype, transient=True)
                    tmp_map[node.data] = tmp_name
                    node.data = tmp_name
            state.add_node(node)
        id_map = {state.node_id(old): state.node_id(new) for old, new in zip(nodes, new_nodes)}

        def map_node(node):
            return state.node(id_map[state.node_id(node)])

        def map_memlet(memlet):
            memlet = copy.deepcopy(memlet)
            memlet.data = tmp_map.get(memlet.data, memlet.data)
            return memlet

        for edge in state.edges():
            if edge.src in nodes or edge.dst in nodes:
                src = map_node(edge.src) if edge.src in nodes else edge.src
                dst = map_node(edge.dst) if edge.dst in nodes else edge.dst
                edge_data = map_memlet(edge.data)
                state.add_edge(src, edge.src_conn, dst, edge.dst_conn, edge_data)

        return new_nodes

    def _replicate_first_map(self, sdfg, parent_map_entry, parent_map_exit, child_map_entry, array_accesses):
        """Replicate tasklet of first map for each read access in second map."""
        state = sdfg.node(self.state_id)
        for array_access in array_accesses:
            array_name = array_access.data
            array = sdfg.arrays[array_name]

            read_offsets = self._read_offsets(state, array_name, parent_map_exit, child_map_entry)

            # Replicate first map tasklets once for each read offset access and
            # connect them to other tasklets accordingly
            for offset, edges in read_offsets.items():
                nodes = self._copy_first_map_contents(sdfg, state, parent_map_entry, parent_map_exit)
                tmp_name = "__otf"
                tmp_name, _ = sdfg.add_scalar(tmp_name, array.dtype, transient=True, find_new_name=True)
                tmp_access = state.add_access(tmp_name)

                for node in nodes:
                    for edge in state.edges_between(node, parent_map_exit):
                        state.add_edge(edge.src, edge.src_conn, tmp_access, None, dace.Memlet(tmp_name))
                        state.remove_edge(edge)

                    for edge in state.edges_between(parent_map_entry, node):
                        memlet = copy.deepcopy(edge.data)
                        memlet.subset.offset(list(offset), negative=False)
                        child_map_entry.add_out_connector(edge.src_conn + "_")
                        state.add_edge(child_map_entry, edge.src_conn + "_", node, edge.dst_conn, memlet)
                        state.remove_edge(edge)

                for edge in edges:
                    state.add_edge(tmp_access, None, edge.dst, edge.dst_conn, dace.Memlet(tmp_name))
