# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

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
from dace.transformation.dataflow.constant_propagation import detect_data_dependencies


@registry.autoregister_params(singlestate=True)
class AddNestedSDFGInputConnector(transformation.Transformation):
    """
    Detects the case when nested SDFG has only output connector for data descriptor that has
    read before write in this SDFG. Adds missing input connector in such case.
    """

    nsdfg = transformation.PatternNode(nodes.NestedSDFG)
    access = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                AddNestedSDFGInputConnector.nsdfg,
                AddNestedSDFGInputConnector.access,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        nsdfg: nodes.NestedSDFG = state.nodes()[candidate[AddNestedSDFGInputConnector.nsdfg]]
        access: nodes.AccessNode = state.nodes()[candidate[AddNestedSDFGInputConnector.access]]

        edge = state.edges_between(nsdfg, access)[0]
        data_desc = edge.src_conn

        if data_desc in nsdfg.in_connectors:
            # transformation is not required because the input connector is already here
            return False

        read_deps, write_deps = detect_data_dependencies(nsdfg.sdfg)

        desc_write_deps = write_deps[data_desc]
        if None not in desc_write_deps:
            # transformation is not required because there are no reads before write
            return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):
        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph

        nsdfg: nodes.NestedSDFG = state.nodes()[candidate[AddNestedSDFGInputConnector.nsdfg]]
        access: nodes.AccessNode = state.nodes()[candidate[AddNestedSDFGInputConnector.access]]

        edge = state.edges_between(nsdfg, access)[0]
        data_desc = edge.src_conn

        out_memlet: memlet.Memlet = edge.data

        # add new input access node to the same data descriptor as the output
        new_in_access = state.add_access(access.data)

        # connect access node to the outer scope if required
        nsdfg_in_edges = state.in_edges(nsdfg)
        if nsdfg.in_connectors:
            # get any (first) input edge for the check
            in_edge = nsdfg_in_edges[0]
            in_access = in_edge.src
            assert isinstance(in_access, nodes.AccessNode)
            in_access_inputs = state.in_edges(in_access)
            if in_access_inputs:
                # access node is expected to be connected to the MapEntry
                map_entry = in_access_inputs[0].src
                assert isinstance(map_entry, nodes.MapEntry)

                # connect map entry with new access node with empty memlet
                state.add_nedge(map_entry, new_in_access, memlet.Memlet())
            else:
                # sampled access node has no other inputs so new access node should not have the as well
                # no action is required
                pass
        else:
            if state.in_edges(nsdfg):
                # nested SDFG doesn't have input connectors but it is connected with empty memlet to the MapEntry
                assert len(nsdfg_in_edges) == 1
                nsdfg_in_edge = nsdfg_in_edges[0]
                assert nsdfg_in_edge.data.is_empty()
                map_entry = nsdfg_in_edge.src
                assert isinstance(map_entry, nodes.MapEntry)

                # remove empty memlet from SDFG
                state.remove_edge(nsdfg_in_edge)

                # connect map entry with new access node with empty memlet
                state.add_nedge(map_entry, new_in_access, memlet.Memlet())
            else:
                # nested SDFG doesn't have input connectors and it is not connected to anything from input side
                # no action is required
                pass

        # copy memlet range from outer edge and create input edge based on it
        state.add_edge(new_in_access, None, nsdfg, data_desc,
                       memlet.Memlet(data=out_memlet.data, subset=out_memlet.subset))

        # add input connector to nested SDFG
        nsdfg.in_connectors[data_desc] = nsdfg.out_connectors[data_desc]