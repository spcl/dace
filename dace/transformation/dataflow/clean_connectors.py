# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy as dcpy
from dace import registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from typing import List, Union
import networkx as nx
from dace.sdfg import state as dace_state
from dace.sdfg import sdfg as dace_sdfg
from dace import memlet
from dace.sdfg import graph as dace_graph
from dace.transformation.helpers import nest_state_subgraph
from dace.sdfg.graph import SubgraphView
from dace.sdfg.replace import replace_properties
from dace import dtypes
from dace import data as dace_data
import itertools


def find_duplicate_in_connectors(state: dace_state.SDFGState, nested_sdfg: nodes.NestedSDFG):
    input_duplicates = {}

    in_edges = state.in_edges(nested_sdfg)

    checked_connectors = {}

    if len(in_edges) == 1 and in_edges[0].data.is_empty():
        # empty memlet, no duplicates
        return input_duplicates

    for e in in_edges:
        access_node = e.src
        conn_name = e.dst_conn
        memlet: Memlet = e.data
        range = memlet.subset
        inp = (access_node.data, range)
        if inp in checked_connectors:
            # we found a duplicate
            original_conn = checked_connectors[inp]
            input_duplicates[original_conn].append(e)
        else:
            # not a duplicate
            checked_connectors[inp] = conn_name
            input_duplicates[conn_name] = []

    return input_duplicates


def find_duplicate_out_connectors(state: dace_state.SDFGState, nested_sdfg: nodes.NestedSDFG):
    output_duplicates = {}

    out_edges = state.out_edges(nested_sdfg)

    checked_connectors = {}

    if len(out_edges) == 1 and out_edges[0].data.is_empty():
        # empty memlet, no duplicates
        return output_duplicates

    for e in out_edges:
        access_node = e.dst
        conn_name = e.src_conn
        memlet: Memlet = e.data
        range = memlet.subset
        inp = (access_node.data, range)
        if inp in checked_connectors:
            # we found a duplicate
            original_conn = checked_connectors[inp]
            output_duplicates[original_conn].append(e)
        else:
            # not a duplicate
            checked_connectors[inp] = conn_name
            output_duplicates[conn_name] = []

    return output_duplicates


def build_access_maps(state: dace_state.SDFGState, nested_sdfg: nodes.NestedSDFG):
    """
    returns mapping from (access node data, memlet range) to connector names
    for input and output connectors separately
    """
    input_map = {}
    output_map = {}

    in_edges = state.in_edges(nested_sdfg)
    out_edges = state.out_edges(nested_sdfg)

    if len(in_edges) != 1 or not in_edges[0].data.is_empty():
        for e in in_edges:
            access_node = e.src
            conn_name = e.dst_conn
            memlet: Memlet = e.data
            range = memlet.subset
            inp = (access_node.data, range)
            input_map[inp] = (conn_name, e)

    if len(out_edges) != 1 or not out_edges[0].data.is_empty():
        for e in out_edges:
            access_node = e.dst
            conn_name = e.src_conn
            memlet: Memlet = e.data
            range = memlet.subset
            inp = (access_node.data, range)
            output_map[inp] = (conn_name, e)

    return input_map, output_map


def find_in_out_duplicates(state: dace_state.SDFGState, nested_sdfg: nodes.NestedSDFG):
    """
    find connectors that are connected to the same access node and range, but have different connector names
    that are connected to input and output.
    Return list with tuples of (in edge, out edge).
    """
    input_map, output_map = build_access_maps(state, nested_sdfg)

    in_out_duplicates = []

    for in_range, in_conn in input_map.items():
        for out_range, out_conn in output_map.items():
            if in_range != out_range:
                continue
            if in_conn[0] != out_conn[0]:
                in_out_duplicates.append((in_conn[1], out_conn[1]))

    return in_out_duplicates


def merge_dict_duplicates(d, orig, dup):
    if orig in d:
        del d[dup]


def merge_symbols(sdfg: dace_sdfg.SDFG, name: str, dup_name: str):
    if name == dup_name:
        return

    symrepl = {
        symbolic.symbol(dup_name):
        symbolic.pystr_to_symbolic(name)
        if isinstance(name, str) else name
    }

    # Replace in arrays and symbols (if a variable name)
    if dtypes.validate_name(dup_name):
        merge_dict_duplicates(sdfg._arrays, name, dup_name)
        merge_dict_duplicates(sdfg.symbols, name, dup_name)

    # Replace inside data descriptors
    for array in sdfg.arrays.values():
        replace_properties(array, symrepl, dup_name, name)

    # Replace in inter-state edges
    for edge in sdfg.edges():
        edge.data.replace(dup_name, name)

    # Replace in states
    for state in sdfg.nodes():
        state.replace(dup_name, name)


def find_read_write_states(sdfg: dace_sdfg.SDFG, array: str):
    """
    For given sdfg and name of array this function returns two sets of states:
    set with reads from array and set with writes into array.
    """

    read_states = set() # states that contain reads
    write_states = set() # states that contain writes

    for state in sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            node: nodes.AccessNode
            if node.data != array:
                continue
            for edge in state.out_edges(node):
                if edge.data.is_empty():
                    continue
                # if we reached this line, then we found read
                read_states.add(state)
            for edge in state.in_edges(node):
                if edge.data.is_empty():
                    continue
                # if we reached this line, then we found write
                write_states.add(state)
                if edge.data.wcr:
                    # if it is wcr node, then this is read-write
                    read_states.add(state)

    return read_states, write_states


def find_writes_without_reads(sdfg: dace_sdfg.SDFG, array: str):
    """
    Finds writes to array that will never be read.
    They are useless writes and can be removed.
    """
    read_states, write_states = find_read_write_states(sdfg, array)

    write_without_read = set()
    for ws in write_states - read_states:
        read_not_found = True
        for e in sdfg.bfs_edges(ws):
            if e.dst in read_states:
                read_not_found = False
        if read_not_found:
            write_without_read.add(ws)

    return write_without_read


def is_conn_write_only(nsdfg: nodes.NestedSDFG, conn: str):
    # detect states that contain reads or writes
    read_states, write_states = find_read_write_states(nsdfg.sdfg, conn)

    # treat read/write states as read-only
    write_states -= read_states

    # build set of states that potentially read global values
    if nsdfg.sdfg.start_state in write_states:
        return True

    use_global = {nsdfg.sdfg.start_state}

    def propagate_condition(src, dst, data):
        return dst not in write_states

    for e in nsdfg.sdfg.dfs_edges(nsdfg.sdfg.start_state, propagate_condition):
        use_global.add(e.dst)

    # check that all reads use can't potentially use global value
    if read_states.intersection(use_global):
        return False

    return True


def replace_accesses(state: dace_state.SDFGState, old_name: str, new_name: str):
    for node in state:
        if not isinstance(node, nodes.AccessNode):
            continue
        node: nodes.AccessNode
        if node.data != old_name:
            continue
        new_node = state.add_access(new_name)
        for e in state.in_edges(node):
            new_memlet = Memlet() if e.data.is_empty() else Memlet(data=new_name, subset=e.data.subset)
            state.add_edge(e.src, e.src_conn, new_node, None, new_memlet)
        for e in state.out_edges(node):
            new_memlet = Memlet() if e.data.is_empty() else Memlet(data=new_name, subset=e.data.subset)
            state.add_edge(new_node, None, e.dst, e.dst_conn, new_memlet)
        state.remove_node(node)


@registry.autoregister_params(singlestate=True)
class CleanNestedSDFGConnectors(transformation.Transformation):
    """
    Removes duplicate connectors that are connected to the same array range.
    Works independenty for input connectors and output connectors
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                CleanNestedSDFGConnectors.nested_sdfg,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[CleanNestedSDFGConnectors.nested_sdfg]]

        duplicate_in_connectors = find_duplicate_in_connectors(state, nested_sdfg)
        duplicate_out_connectors = find_duplicate_out_connectors(state, nested_sdfg)

        for orig_conn, edge_list in itertools.chain(duplicate_in_connectors.items(), duplicate_out_connectors.items()):
            if edge_list:
                return True

        # duplicates not found
        return False

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        nested_sdfg: dace_nodes.NestedSDFG = state.nodes()[candidate[CleanNestedSDFGConnectors.nested_sdfg]]

        duplicate_in_connectors = find_duplicate_in_connectors(state, nested_sdfg)
        duplicate_out_connectors = find_duplicate_out_connectors(state, nested_sdfg)

        # remove duplicate connectors
        for orig_conn, edge_list in duplicate_in_connectors.items():
            for e in edge_list:
                dup_conn = e.dst_conn
                state.remove_edge(e)
                nested_sdfg.remove_in_connector(dup_conn)
                merge_symbols(nested_sdfg.sdfg, orig_conn, dup_conn)

        for orig_conn, edge_list in duplicate_out_connectors.items():
            for e in edge_list:
                dup_conn = e.src_conn
                state.remove_edge(e)
                nested_sdfg.remove_out_connector(dup_conn)
                merge_symbols(nested_sdfg.sdfg, orig_conn, dup_conn)


@registry.autoregister_params(singlestate=True)
class UnifyInOutNestedSDFGConnectors(transformation.Transformation):
    """
    It finds in and out connectors that are connected to the same access node with the same memlet range and
    makes their names the same.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                CleanNestedSDFGConnectors.nested_sdfg,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[CleanNestedSDFGConnectors.nested_sdfg]]

        duplicate_in_out_connectors = find_in_out_duplicates(state, nested_sdfg)

        if duplicate_in_out_connectors:
            return True

        # duplicates not found
        return False

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        nested_sdfg: dace_nodes.NestedSDFG = state.nodes()[candidate[CleanNestedSDFGConnectors.nested_sdfg]]

        duplicate_edges = find_in_out_duplicates(state, nested_sdfg)

        # remove duplicate connectors
        for in_edge, out_edge in duplicate_edges:
            in_conn: dace_graph.MultiConnectorEdge = in_edge.dst_conn
            out_conn: dace_graph.MultiConnectorEdge = out_edge.src_conn

            # rename output connector to match input connector
            nested_sdfg.out_connectors[in_conn] = nested_sdfg.out_connectors.pop(out_conn)

            state.add_edge(out_edge.src, in_conn, out_edge.dst, out_edge.dst_conn, out_edge.data)
            state.remove_edge(out_edge)

            merge_symbols(nested_sdfg.sdfg, in_conn, out_conn)


@registry.autoregister_params(singlestate=True)
class RemoveReadSDFGConnectors(transformation.Transformation):
    """
    Detect SDFG connectors that exist both in input and output connectors, but doesn't use input value.
    In such case remove input connector and leave only output connector.
    """
    access_node = transformation.PatternNode(nodes.AccessNode)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                RemoveReadSDFGConnectors.access_node,
                RemoveReadSDFGConnectors.nested_sdfg,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg: dace_sdfg.SDFG, strict=False):
        access_node: nodes.AccessNode = state.nodes()[candidate[RemoveReadSDFGConnectors.access_node]]
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[RemoveReadSDFGConnectors.nested_sdfg]]

        from dace.transformation.dataflow.constant_propagation import detect_data_dependencies # avoid import loop
        read_deps, write_deps = detect_data_dependencies(nested_sdfg.sdfg)

        in_out_connectors = set(nested_sdfg.in_connectors).intersection(set(nested_sdfg.out_connectors))

        edge = state.edges_between(access_node, nested_sdfg)[0]

        target_conn = edge.dst_conn

        if target_conn not in in_out_connectors:
            return False

        if None in write_deps[target_conn]:

            # value written in outer scope has some reads in this SDFG, so transformation can't be applied here
            return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        access_node: nodes.AccessNode = state.nodes()[candidate[RemoveReadSDFGConnectors.access_node]]
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[RemoveReadSDFGConnectors.nested_sdfg]]

        edge = state.edges_between(access_node, nested_sdfg)[0]

        nested_sdfg.remove_in_connector(edge.dst_conn)
        state.remove_edge(edge)

        # if access node became isolated, then remove it
        if not access_node.has_reads(state) and not access_node.has_writes(state):
            state.remove_node(access_node)


@registry.autoregister_params(singlestate=True)
class NestTransients(transformation.Transformation):
    """
    If outputs of some nested sdfg go into transients that are never used anymore,
    we can remove these transients from outermost scope and put it into innermost scope.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    access_node = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestTransients.nested_sdfg,
                NestTransients.access_node,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg: dace_sdfg.SDFG, strict=False):
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[NestTransients.nested_sdfg]]
        access_node: nodes.AccessNode = state.nodes()[candidate[NestTransients.access_node]]


        data_name = access_node.data
        data_obj = sdfg.data(data_name)


        # access node should point to data object
        if not isinstance(data_obj, dace_data.Data):
            return False # probably this means that access node corresponds to symbol

        data_obj: dace_data.Data

        edges = state.edges_between(nested_sdfg, access_node)
        # if there are more than 1 edge, try applying CleanNestedSDFGConnectors transformation first
        if len(edges) != 1:
            return False

        edge: dace_graph.MultiConnectorEdge = edges[0]

        # if connector is used for both reading and writing we can't move transient inside
        if not is_conn_write_only(nested_sdfg, edge.src_conn):
            return False

        # data object should be transient
        if not data_obj.transient:
            return False

        # data object should be write only in all states starting from the current state
        states_to_check = {state}
        for e in sdfg.bfs_edges(state):
            states_to_check.add(e.dst)

        for other_state in states_to_check:

            for an in other_state.nodes():
                if not isinstance(an, nodes.AccessNode):
                    continue # skip non-access nodes

                an: nodes.AccessNode
                if an.data != data_name:
                    continue # skip access nodes that point to other data objects

                for e in other_state.out_edges(an):
                    e: dace_graph.MultiConnectorEdge
                    memlet = e.data
                    if not memlet.is_empty():
                        # we found non-empty outgoing memlet, so this data object is read at some point in future,
                        # therefore we can't apply the transformation.
                        return False

        return True

    @staticmethod
    def match_to_str(state: dace_state.SDFGState, candidate):
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[NestTransients.nested_sdfg]]
        access_node: nodes.AccessNode = state.nodes()[candidate[NestTransients.access_node]]

        return f"{str(nested_sdfg)} -> {access_node.label}"

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[NestTransients.nested_sdfg]]
        access_node: nodes.AccessNode = state.nodes()[candidate[NestTransients.access_node]]

        data_name = access_node.data
        data_obj: dace_data.Data = sdfg.data(data_name)

        # remove connectors which will be unused after transformation
        edges = state.edges_between(nested_sdfg, access_node)
        edge = edges[0]
        connector_name = edge.src_conn
        nested_sdfg.remove_out_connector(connector_name)

        # remove access node
        state.remove_node(access_node)

        # replace existing data object inside nested sdfg by the transient
        old_data_obj = nested_sdfg.sdfg.arrays[connector_name]

        new_data_obj = dace_data.Array(dtype=old_data_obj.dtype,
                                       shape=old_data_obj.shape,
                                       transient=True)

        nested_sdfg.sdfg.arrays[connector_name] = new_data_obj



@registry.autoregister_params(singlestate=True)
class CleanNestedWrites(transformation.Transformation):
    """
    Detects unused outputs of nested SDFGs and removes them
    by replacing writes with writes to unused transient.
    It is similar to NestTransients, but works even when transient can't be nested completely,
    for example, if it is both input and output to the nested SDFG.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    access_node = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                CleanNestedWrites.nested_sdfg,
                CleanNestedWrites.access_node,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg: dace_sdfg.SDFG, strict=False):
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[CleanNestedWrites.nested_sdfg]]
        access_node: nodes.AccessNode = state.nodes()[candidate[CleanNestedWrites.access_node]]

        data_name = access_node.data
        data_obj = sdfg.data(data_name)

        # access node should point to data object
        if not isinstance(data_obj, dace_data.Data):
            return False # probably this means that access node corresponds to symbol

        data_obj: dace_data.Data

        edges = state.edges_between(nested_sdfg, access_node)
        # if there are more than 1 edge, try applying CleanNestedSDFGConnectors transformation first
        if len(edges) != 1:
            return False

        edge: dace_graph.MultiConnectorEdge = edges[0]

        # data object should be transient
        if not data_obj.transient:
            return False

        # data object should not be read in all states starting from the current state
        states_to_check = set()
        for e in sdfg.bfs_edges(state):
            states_to_check.add(e.dst)

        for other_state in states_to_check:

            for an in other_state.nodes():
                if not isinstance(an, nodes.AccessNode):
                    continue # skip non-access nodes

                an: nodes.AccessNode
                if an.data != data_name:
                    continue # skip access nodes that point to other data objects

                for e in other_state.out_edges(an):
                    e: dace_graph.MultiConnectorEdge
                    memlet = e.data
                    if not memlet.is_empty():
                        # we found non-empty outgoing memlet, so this data object is read at some point in future,
                        # therefore we can't apply the transformation.
                        return False

        # find writes that doesn't have reads in the following states
        # otherwise this transformation have nothing to do
        states_with_writes_without_reads = find_writes_without_reads(nested_sdfg.sdfg, edge.src_conn)
        if not states_with_writes_without_reads:
            return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):
        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        nested_sdfg: nodes.NestedSDFG = state.nodes()[candidate[CleanNestedWrites.nested_sdfg]]
        access_node: nodes.AccessNode = state.nodes()[candidate[CleanNestedWrites.access_node]]

        data_name = access_node.data
        data_obj: dace_data.Data = sdfg.data(data_name)

        edges = state.edges_between(nested_sdfg, access_node)
        edge = edges[0]
        connector_name = edge.src_conn

        states_with_writes_without_reads = find_writes_without_reads(nested_sdfg.sdfg, connector_name)

        # create stub transient array that will be used to make useless writes
        transient_name = "t_" + connector_name
        nested_sdfg.sdfg.add_transient(name=transient_name, shape=data_obj.shape, dtype=data_obj.dtype)

        # replace useless writes with writes to stub transient array
        for state in states_with_writes_without_reads:
            replace_accesses(state, connector_name, transient_name)


@registry.autoregister_params(singlestate=True)
class RemoveDanglingAccessNodes(transformation.Transformation):

    access_node = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                RemoveDanglingAccessNodes.access_node,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        access_node: nodes.AccessNode = state.nodes()[candidate[RemoveDanglingAccessNodes.access_node]]

        if access_node.has_reads(state) or access_node.has_writes(state):
            return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        access_node: dace_nodes.AccessNode = state.nodes()[candidate[RemoveDanglingAccessNodes.access_node]]

        state.remove_node(access_node)
