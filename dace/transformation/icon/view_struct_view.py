# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import copy

from typing import Any, Dict, List
from dace.sdfg import SDFG, SDFGState
from dace.properties import DictProperty, make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.data import Structure, View
import re
from dace.transformation import pass_pipeline as ppl

@make_properties
class ReconnectVSV(ppl.Pass):
    vsv_map = DictProperty(key_type=str, value_type=list[nodes.AccessNode])

    def __init__(self, vsv_map):
        self.vsv_map = vsv_map
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _get_src_dst(self, state : SDFGState, n1 : nodes.Any, n2: nodes.Any):
        n1_to_n2 = [e.dst for e in state.out_edges(n1) if e.dst == n2]
        n2_to_n1 = [e.dst for e in state.out_edges(n2) if e.dst == n1]
        if len(n2_to_n1) == 0 and len(n1_to_n2) == 0:
            raise Exception("E1")
        elif len(n2_to_n1) != 0 and len(n1_to_n2) != 0:
            raise Exception("E2")
        elif len(n2_to_n1) == 0:
            assert (len(n1_to_n2) > 0)
            return (n1, n2)
        else:
            assert (len(n2_to_n1) > 0)
            return (n2, n1)


    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> int:
        for state in sdfg.nodes():
            nodes = state.nodes()
            for node in nodes:
                if node not in state.nodes():
                    continue
                for in_connected_nodes, out_connected_nodes in self.vsv_map.values():
                    assert len(in_connected_nodes) <= 1 and len(out_connected_nodes) <= 1
                    if len(in_connected_nodes) == 1 and len(out_connected_nodes) == 1:
                        if node in in_connected_nodes:
                            src = node
                            dst = out_connected_nodes[0]
                            for oe in state.out_edges(dst):
                                assert(oe.src_conn is None)
                                state.add_edge(src, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
                                print("xccx")
                            state.remove_node(dst)
                        elif node in out_connected_nodes:
                            continue

        return 0

    def annotates_memlets():
        return False
