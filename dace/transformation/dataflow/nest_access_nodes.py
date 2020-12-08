# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy as dcpy
from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import replace
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from typing import List, Union
import networkx as nx
from dace.sdfg import state as dace_state
from dace.sdfg import sdfg as dace_sdfg
from dace import memlet
from dace.sdfg import graph as dace_graph


@registry.autoregister_params(singlestate=True)
class NestExitAccessNode(transformation.Transformation):
    """
    Takes "Something->MapExit->AccessNode" and transforms it into "Something->AccessNode--MapExit--AccessNode".
    "->" is non-empty memlet
    "--" is empty memlet
    """

    map_exit = transformation.PatternNode(nodes.MapExit)
    access_node = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestExitAccessNode.map_exit,
                NestExitAccessNode.access_node,
            )
        ]

    @staticmethod
    def can_be_applied(state, candidate, expr_index, sdfg, strict=False):
        map_exit = state.nodes()[candidate[NestExitAccessNode.map_exit]]
        access_node = state.nodes()[candidate[NestExitAccessNode.access_node]]

        # edges from map_exit to access node
        outer_edges = state.edges_between(map_exit, access_node)
        if all(e.data.is_empty() for e in outer_edges):
            return False

        return True

    @staticmethod
    def match_to_str(state, candidate):
        map_exit: nodes.MapExit = state.nodes()[candidate[NestExitAccessNode.map_exit]]
        access_node: nodes.AccessNode = state.nodes()[candidate[NestExitAccessNode.access_node]]

        return map_exit.map.label + ": " + str(map_exit.map.params) + " -> " + access_node.label

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        map_exit: nodes.MapExit = state.nodes()[candidate[NestExitAccessNode.map_exit]]
        access_node: nodes.AccessNode = state.nodes()[candidate[NestExitAccessNode.access_node]]

        # edges from map_exit to access node
        outer_edges = state.edges_between(map_exit, access_node)

        # gather connector names without prefix
        connector_names: List[str] = []
        outer_edge: dace_graph.MultiConnectorEdge
        for outer_edge in outer_edges:
            if outer_edge.data.is_empty():
                continue

            outer_connector_name = outer_edge.src_conn
            prefix_len = len('OUT_')
            connector_names.append(outer_connector_name[prefix_len:])

            # replace outer edges with empty memlets
            state.remove_edge(outer_edge)
            state.add_nedge(outer_edge.src, outer_edge.dst, memlet.Memlet())

        # edges inside map
        for connector in connector_names:
            inner_connector_name = 'IN_' + connector
            inner_edges = state.in_edges_by_connector(map_exit, inner_connector_name)
            for inner_edge in inner_edges:
                # replace each inner edge "->" by pattern "->AccessNode--"
                new_access_node: nodes.AccessNode = state.add_access(access_node.data)
                state.remove_edge(inner_edge)
                state.add_edge(inner_edge.src, inner_edge.src_conn, new_access_node, None, inner_edge.data)
                state.add_nedge(new_access_node, inner_edge.dst, memlet.Memlet())

        # remove unused map_exit connectors
        for connector in connector_names:
            inner_connector_name = 'IN_' + connector
            outer_connector_name = 'OUT_' + connector
            map_exit.remove_in_connector(inner_connector_name)
            map_exit.remove_out_connector(outer_connector_name)


@registry.autoregister_params(singlestate=True)
class NestEntryAccessNode(transformation.Transformation):
    """
    Takes "AccessNode->MapEntry->Something" and transforms it into "AccessNode--MapEntry--AccessNode->Something".
    "->" is non-empty memlet
    "--" is empty memlet
    """

    access_node = transformation.PatternNode(nodes.AccessNode)
    map_entry = transformation.PatternNode(nodes.MapEntry)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestEntryAccessNode.access_node,
                NestEntryAccessNode.map_entry,
            )
        ]

    @staticmethod
    def can_be_applied(state, candidate, expr_index, sdfg, strict=False):
        access_node = state.nodes()[candidate[NestEntryAccessNode.access_node]]
        map_entry = state.nodes()[candidate[NestEntryAccessNode.map_entry]]

        # edges from map_exit to access node
        outer_edges = state.edges_between(access_node, map_entry)
        if all(e.data.is_empty() for e in outer_edges):
            return False

        return True

    @staticmethod
    def match_to_str(state, candidate):
        access_node = state.nodes()[candidate[NestEntryAccessNode.access_node]]
        map_entry = state.nodes()[candidate[NestEntryAccessNode.map_entry]]

        return access_node.label + " -> " + map_entry.map.label + ": " + str(map_entry.map.params)

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        access_node = state.nodes()[candidate[NestEntryAccessNode.access_node]]
        map_entry = state.nodes()[candidate[NestEntryAccessNode.map_entry]]

        # edges from map_exit to access node
        outer_edges = state.edges_between(access_node, map_entry)

        # gather connector names without prefix
        connector_names: List[str] = []
        outer_edge: dace_graph.MultiConnectorEdge
        for outer_edge in outer_edges:
            if outer_edge.data.is_empty():
                continue

            outer_connector_name = outer_edge.dst_conn
            prefix_len = len('IN_')
            connector_names.append(outer_connector_name[prefix_len:])

            # replace outer edges with empty memlets
            state.remove_edge(outer_edge)
            state.add_nedge(outer_edge.src, outer_edge.dst, memlet.Memlet())

        # edges inside map
        for connector in connector_names:
            inner_connector_name = 'OUT_' + connector
            inner_edges = state.out_edges_by_connector(map_entry, inner_connector_name)
            for inner_edge in inner_edges:
                # replace each inner edge "->" by pattern "--AccessNode->"
                new_access_node: nodes.AccessNode = state.add_access(access_node.data)
                state.remove_edge(inner_edge)
                state.add_nedge(inner_edge.src, new_access_node, memlet.Memlet())
                state.add_edge(new_access_node, None, inner_edge.dst, inner_edge.dst_conn, inner_edge.data)

        # remove unused map_exit connectors
        for connector in connector_names:
            outer_connector_name = 'IN_' + connector
            inner_connector_name = 'OUT_' + connector
            map_entry.remove_in_connector(outer_connector_name)
            map_entry.remove_out_connector(inner_connector_name)


@registry.autoregister_params(singlestate=True)
class RemoveUnusedAccessNode(transformation.Transformation):
    """
    Removes AccessNodes that are connected only to empty Memlets.
    Each pair of input and output empty memlets through AccessNode is replaced by a single empty memlet.

    +----+                      +----+                    +----+                      +----+
    |    |                      |    |                    |    +----------+----------->    |
    |    |        XXXXX         |    |                    |    |          |           |    |
    |    +------->     +-------->    |            X       |    |      +--------------->    |
    +----+      X       X       +----+            XX      +----+      |   |           +----+
                X       X                  XXXXXXXXXX                 |   |
    +----+      X       X       +----+            XX      +----+      |   |           +----+
    |    +------->     +-------->    |            X       |    |      |   +----------->    |
    |    |        XXXXX         |    |                    |    |      |               |    |
    |    |                      |    |                    |    +------+--------------->    |
    +----+                      +----+                    +----+                      +----+

    """

    access_node = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                RemoveUnusedAccessNode.access_node,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        access_node = state.nodes()[candidate[RemoveUnusedAccessNode.access_node]]

        inputs_empty = all(e.data.is_empty() for e in state.in_edges(access_node))
        outputs_empty = all(e.data.is_empty() for e in state.out_edges(access_node))
        if inputs_empty and outputs_empty:
            return True

        return False

    @staticmethod
    def match_to_str(state, candidate):
        access_node = state.nodes()[candidate[RemoveUnusedAccessNode.access_node]]

        return access_node.label

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        access_node = state.nodes()[candidate[RemoveUnusedAccessNode.access_node]]

        initial_in_edges = state.in_edges(access_node)
        initial_out_edges = state.out_edges(access_node)

        for in_edge in initial_in_edges:
            for out_edge in initial_out_edges:
                state.add_nedge(in_edge.src, out_edge.dst, memlet.Memlet())

        for in_edge in initial_in_edges:
            state.remove_edge(in_edge)

        for out_edge in initial_out_edges:
            state.remove_edge(out_edge)

        state.remove_node(access_node)