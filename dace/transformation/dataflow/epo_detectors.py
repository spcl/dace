# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that deted Embarassingly Parallel Operations (EPO). """

from dace.sdfg.utils import consolidate_edges
from copy import deepcopy
from numbers import Number
from typing import Dict, List
import dace
import sympy
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
        params = [dace.symbol(p) for p in map_entry.map.params]

        inputs = dict()
        for _, _, _, _, m in graph.out_edges(map_entry):
            if not m.data:
                continue
            desc = sdfg.arrays[m.data]
            if desc not in inputs.keys():
                inputs[desc] = []
            inputs[desc].append(m.subset)
        
        for desc, accesses in inputs.items():
            if isinstance(desc, dace.data.Scalar):
                continue
            elif isinstance(desc, (dace.data.Array, dace.data.View)):
                if list(desc.shape) == [1]:
                    continue
                for a in accesses:
                    if a.num_elements() != 1:
                        # if list(a.size()) == list(desc.shape):
                        #     continue
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
            if desc not in outputs.keys():
                outputs[desc] = []
            outputs[desc].append(m.subset)
        
        for desc, accesses in outputs.items():
            if isinstance(desc, (dace.data.Array, dace.data.View)):
                for a in accesses:
                    if a.num_elements() != 1:
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


@registry.autoregister_params(singlestate=True)
class StencilOperation(pm.Transformation):
    """ Detects stencil operations.
    """

    map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(StencilOperation.map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[StencilOperation.map_entry])
        map_exit = graph.exit_node(map_entry)
        params = [dace.symbol(p) for p in map_entry.map.params]

        inputs = dict()
        for _, _, _, _, m in graph.out_edges(map_entry):
            if not m.data:
                continue
            desc = sdfg.arrays[m.data]
            if desc not in inputs.keys():
                inputs[desc] = []
            inputs[desc].append(m.subset)
        
        stencil_found = False
        for desc, accesses in inputs.items():
            if isinstance(desc, dace.data.Scalar):
                continue
            elif isinstance(desc, (dace.data.Array, dace.data.View)):
                if list(desc.shape) == [1]:
                    continue
                first_access = None
                for a in accesses:
                    if a.num_elements() != 1:
                        # if list(a.size()) == list(desc.shape):
                        #     continue
                        return False
                    if first_access:
                        new_access = deepcopy(a)
                        new_access.offset(first_access, True)
                        for idx in new_access.min_element():
                            if not isinstance(idx, Number):
                                return False
                            if idx != 0:
                                stencil_found = True
                    else:
                        first_access = a
                    indices = a.min_element()
                    unmatched_indices = set(params)
                    for idx in indices:
                        if isinstance(idx, sympy.Symbol):
                            bidx = idx
                        elif isinstance(idx, sympy.Add):
                            if len(idx.free_symbols) != 1:
                                return False
                            bidx = list(idx.free_symbols)[0]
                        else:
                            return False
                        if bidx in unmatched_indices:
                            unmatched_indices.remove(bidx)
                    if len(unmatched_indices) > 0:
                        return False
            else:
                return False

        outputs = dict()
        for _, _, _, _, m in graph.in_edges(map_exit):
            if m.wcr:
                return False
            desc = sdfg.arrays[m.data]
            if desc not in outputs.keys():
                outputs[desc] = []
            outputs[desc].append(m.subset)
        
        for desc, accesses in outputs.items():
            if isinstance(desc, (dace.data.Array, dace.data.View)):
                for a in accesses:
                    if a.num_elements() > 1:
                        return False
                    indices = a.min_element()
                    unmatched_indices = set(params)
                    for idx in indices:
                        if isinstance(idx, sympy.Symbol):
                            bidx = idx
                        elif isinstance(idx, sympy.Add):
                            if len(idx.free_symbols) != 1:
                                return False
                            bidx = list(idx.free_symbols)[0]
                        else:
                            return False
                        if bidx in unmatched_indices:
                            unmatched_indices.remove(bidx)
                    if len(unmatched_indices) > 0:
                        return False
            else:
                return False

        return stencil_found

    @staticmethod
    def match_to_str(graph: dace.SDFGState, candidate: Dict[pm.PatternNode,
                                                            int]) -> str:
        map_entry = graph.node(candidate[StencilOperation.map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        pass


@registry.autoregister_params(singlestate=True)
class OuterProductOperation(pm.Transformation):
    """ Detects outer-product operations.
    """

    map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(OuterProductOperation.map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[OuterProductOperation.map_entry])
        map_exit = graph.exit_node(map_entry)
        params = [dace.symbol(p) for p in map_entry.map.params]

        inputs = dict()
        for _, _, _, _, m in graph.out_edges(map_entry):
            if not m.data:
                continue
            desc = sdfg.arrays[m.data]
            if desc not in inputs.keys():
                inputs[desc] = []
            inputs[desc].append(m.subset)
        
        outer_product_found = False
        for desc, accesses in inputs.items():
            if isinstance(desc, dace.data.Scalar):
                continue
            elif isinstance(desc, (dace.data.Array, dace.data.View)):
                if list(desc.shape) == [1]:
                    continue
                for a in accesses:
                    indices = a.min_element()
                    unmatched_indices = set(params)
                    for idx in indices:
                        if not isinstance(idx, sympy.Symbol):
                            return False
                        if idx in unmatched_indices:
                            unmatched_indices.remove(idx)
                    if len(unmatched_indices) == 0:
                        return False
                    outer_product_found = True
            else:
                return False

        outputs = dict()
        for _, _, _, _, m in graph.in_edges(map_exit):
            if m.wcr:
                return False
            desc = sdfg.arrays[m.data]
            if desc not in outputs.keys():
                outputs[desc] = []
            outputs[desc].append(m.subset)
        
        for desc, accesses in outputs.items():
            if isinstance(desc, (dace.data.Array, dace.data.View)):
                for a in accesses:
                    if a.num_elements() != 1:
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

        return outer_product_found

    @staticmethod
    def match_to_str(graph: dace.SDFGState, candidate: Dict[pm.PatternNode,
                                                            int]) -> str:
        map_entry = graph.node(candidate[OuterProductOperation.map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        pass
