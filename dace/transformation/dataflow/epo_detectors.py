# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that deted Embarassingly Parallel Operations (EPO). """

from dace.sdfg.utils import consolidate_edges
from typing import Dict, List
import dace
from dace import dtypes, registry, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import OrderedMultiDiConnectorGraph
from dace.transformation import transformation as pm


@registry.autoregister_params(singlestate=True)
class ElementWiseArrayOperation(pm.Transformation):
    """ Detects element-wise array operations.
    """

    map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(ElementWiseArrayOperation.map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[ElementWiseArrayOperation.map_entry])
        map_exit = graph.exit_node(map_entry)
        params = map_entry.map.params
        param_num = len(params)

        inputs = dict()
        for _, _, _, _, m in graph.out_edges(map_entry):
            desc = sdfg.arrays[m.data]
            if desc in inputs.keys():
                inputs[desc].add(m.subset)
            else:
                inputs[desc] = set(m.subset)
        
        for desc, accesses in inputs.items():
            if isinstance(desc, dace.data.Scalar):
                continue
            elif isinstance(desc, (dace.data.Array, dace.data.View)):
                for a in accesses:
                    if a.num_elements > 1:
                        return False
                    indices = a.min_element()
                    unmatched_indices = set(params)
                    for idx in indices:
                        if idx in unmatched_indices:
                            unmatched_indices.remove(idx)
                    if len(unmatched_indices) > 0:
                        return False
            else:
                return False

        outputs = dict()
        for _, _, _, _, m in graph.in_edges(map_exit):
            if m.wcr:
                return False
            desc = sdfg.arrays[m.data]
            if desc in outputs.keys():
                outputs[desc].add(m.subset)
            else:
                outputs[desc] = set(m.subset)
        
        for desc, accesses in outputs.items():
            if isinstance(desc, (dace.data.Array, dace.data.View)):
                for a in accesses:
                    if a.num_elements > 1:
                        return False
                    indices = a.min_element()
                    unmatched_indices = set(params)
                    for idx in indices:
                        if idx in unmatched_indices:
                            unmatched_indices.remove(idx)
                    if len(unmatched_indices) > 0:
                        return False
            else:
                return False

        return True

    @staticmethod
    def match_to_str(graph: dace.SDFGState, candidate: Dict[pm.PatternNode,
                                                            int]) -> str:
        map_entry = graph.node(candidate[ElementWiseArrayOperation.map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        pass
