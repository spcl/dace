# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import utils, graph
from dace.codegen.targets import fpga
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet

class SDFGMultiplier(transformation.Transformation):
    """
    Nests a whole SDFG and packs it into an unrolled map. 
    """

    # type=Tuple[str, str]
    outer_map_range = properties.Property(
        dtype=Tuple,
        default=("k", "32"),
        desc="Stores the range for the outer HBM map. Defaults to k = 32.")
        
    @staticmethod
    def can_be_applied(self, graph: Union[SDFG, SDFGState], candidate: Dict['PatternNode', int], expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        return True

    @staticmethod
    def expressions(self) -> List[graph.SubgraphView]:
        # Matches anything
        return [networkx.DiGraph()]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        unrollparam = self.outer_map_range

        nesting = interstate.NestSDFG(sdfg.sdfg_id, -1, {}, self.expr_index)
        nesting.apply(sdfg)
        state = sdfg.states()[0]
        nsdfg_node = list(
            filter(lambda x: isinstance(x, nd.NestedSDFG), state.nodes()))[0]
        nsdfg_node.no_inline = True

        map_enter, map_exit = state.add_map("hbm_unrolled_map",
                                            {unrollparam[0]: unrollparam[1]},
                                            dtypes.ScheduleType.Unrolled)
        for input in state.in_edges(nsdfg_node):
            state.remove_edge(input)
            state.add_memlet_path(input.src,
                                  map_enter,
                                  nsdfg_node,
                                  memlet=input.data,
                                  src_conn=input.src_conn,
                                  dst_conn=input.dst_conn)
        for output in state.out_edges(nsdfg_node):
            state.remove_edge(output)
            state.add_memlet_path(nsdfg_node,
                                  map_exit,
                                  output.dst,
                                  memlet=output.data,
                                  src_conn=output.src_conn,
                                  dst_conn=output.dst_conn)