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


def find_duplicate_connectors(state: dace_state.SDFGState, nested_sdfg: nodes.NestedSDFG):
    duplicate_edges_dict = {}

    in_edges = state.in_edges(nested_sdfg)
    out_edges = state.out_edges(nested_sdfg)

    checked_connectors = {}

    # we process out_edges first because they have more priority to be preserved if duplicates
    # found among inputs and outputs simultaneously
    for e in out_edges + in_edges:
        access_node = e.src if e in in_edges else e.dst
        conn_name = e.dst_conn if e in in_edges else e.src_conn
        memlet: Memlet = e.data
        range = memlet.subset
        inp = (access_node.data, range)
        if inp in checked_connectors:
            # we found a duplicate
            original_conn = checked_connectors[inp]
            duplicate_edges_dict[original_conn].append(e)
        else:
            # not a duplicate
            checked_connectors[inp] = conn_name
            duplicate_edges_dict[conn_name] = []

    return duplicate_edges_dict


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


def is_conn_write_only(nsdfg: nodes.NestedSDFG, conn: str):
    # detect states that contain reads or writes
    read_states = set() # states that contain reads
    write_states = set() # states that contain writes

    for state in nsdfg.sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            node: nodes.AccessNode
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


@registry.autoregister_params(singlestate=True)
class CleanNestedSDFGConnectors(transformation.Transformation):
    """
    Removes duplicate connectors.
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

        duplicate_connectors = find_duplicate_connectors(state, nested_sdfg)

        for orig_conn, edge_list in duplicate_connectors.items():
            if edge_list:
                return True

        # duplicates not found
        return False

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        nested_sdfg: dace_nodes.NestedSDFG = state.nodes()[candidate[CleanNestedSDFGConnectors.nested_sdfg]]

        duplicate_connectors = find_duplicate_connectors(state, nested_sdfg)

        # remove duplicate connectors
        for orig_conn, edge_list in duplicate_connectors.items():
            for e in edge_list:
                dup_conn = e.dst_conn
                state.remove_edge(e)
                if e.src == nested_sdfg:
                    nested_sdfg.remove_out_connector(dup_conn)
                else:
                    nested_sdfg.remove_in_connector(dup_conn)
                merge_symbols(nested_sdfg.sdfg, orig_conn, dup_conn)


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

        # make existing data object inside nested sdfg transient
        nested_data_obj = nested_sdfg.sdfg.data(connector_name)

        nested_data_obj.transient = True


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