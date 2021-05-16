# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from os import stat
from typing import Any, AnyStr, Dict, Set, Tuple, Union
import re

from dace import dtypes, registry, SDFG, SDFGState, symbolic
from dace.transformation import transformation as pm, helpers
from dace.sdfg import nodes, utils
from dace.sdfg.analysis import cfg


@registry.autoregister_params(singlestate=True, strict=False)
class RedundantWrite(pm.Transformation):
    """ Remove redundant writes, i.e., a write that is followed by another
    write on the exact same data, without the value of the first being used
    anywhere in the program.
    NOTE (1): This first version of the transformation targets back-to-back 
    write that follow the pattern Write -> NestedSDFG -> Write.
    NOTE (2): This first version of the transformation targets first writes
    that are written by a Tasklet with no inputs.
    """

    fwrite = pm.PatternNode(nodes.AccessNode)
    nsdfg = pm.PatternNode(nodes.NestedSDFG)
    swrite = pm.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [utils.node_path_graph(RedundantWrite.fwrite,
                                    RedundantWrite.nsdfg, RedundantWrite.swrite)]

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:

        fwrite = graph.node(candidate[RedundantWrite.fwrite])
        nsdfg = graph.node(candidate[RedundantWrite.nsdfg])
        swrite = graph.node(candidate[RedundantWrite.swrite])

        # The data must match
        if fwrite.data != swrite.data:
            return False
        
        # The value of the first write must not be used by anything other
        # than the nested SDFG
        if graph.out_degree(fwrite) > 1:
            return False
        
        # The value must not be actually read in the nested SDFG
        conn = graph.out_edges(fwrite)[0].dst_conn
        for state in nsdfg.sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == conn:
                    # Check for read access
                    if state.in_degree(node) == 0:
                        return False
                    # Check that there is a corresponding out connector
                    # NOTE: This is probably not needed since it would be
                    # invalid to write to an input connector.
                    if conn not in nsdfg.out_connectors:
                        return False

        # NOTE: We currently target Tasklet -> Write -> NSDFG -> Write
        # Enhance this to allow matching with more general subgraphs.
        for e in graph.in_edges(fwrite):
            if not isinstance(e.src, nodes.Tasklet):
                return False
            if graph.in_degree(e.src) > 0:
                return False

        return True

    def apply(self, sdfg: SDFG) -> Union[Any, None]:

        state = sdfg.node(self.state_id)
        fwrite = self.fwrite(sdfg)
        nsdfg = self.nsdfg(sdfg)

        # NOTE: We currently target Tasklet -> Write -> NSDFG -> Write
        # Enhance this to remove more general subgraphs with greater precision.
        e1 = state.edges_between(fwrite, nsdfg)[0]
        for e in state.in_edges(fwrite):
            src = state.memlet_path(e)[0].src
            state.remove_memlet_path(e, remove_orphans=True)
            if isinstance(src, nodes.CodeNode):
                state.remove_node(src)
        state.remove_memlet_path(e1, remove_orphans=True)
