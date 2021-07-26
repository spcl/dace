# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx
from dace import data, dtypes, properties, registry, subsets, symbolic
from dace.sdfg import utils, graph
from dace.codegen.targets import fpga
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet

@registry.autoregister
@properties.make_properties
class HbmAccess(transformation.Transformation):
    """
    ...
    """

    _access_node = nd.AccessNode("")
    _map_entry = nd.MapEntry(None)
    _map_exit = nd.MapExit(None)

    @staticmethod
    def can_be_applied(self, graph: Union[SDFG, SDFGState], candidate: Dict['PatternNode', int], expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        node: nd.AccessNode = graph.nodes()[candidate[HbmAccess._access_node]]
        map_entry: nd.MapEntry = graph.nodes()[candidate[HbmAccess._map_entry]]

        if map_entry.map.schedule != dtypes.ScheduleType.Unrolled:
            return False

        desc = sdfg.arrays[node.data]
        if not isinstance(desc, data.Array) or isinstance(desc, data.View):
            return False

        if len(map_entry.map.params) != 1:
            return False

        return True

    @staticmethod
    def expressions(self) -> List[graph.SubgraphView]:
        return [
            utils.node_path_graph(HbmAccess._access_node,
                                  HbmAccess._map_entry),
            utils.node_path_graph(HbmAccess._map_exit,
                                  HbmAccess._access_node)
        ]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        pass