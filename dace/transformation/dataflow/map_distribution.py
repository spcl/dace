# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that distribute Map computations """

from dace.sdfg.utils import consolidate_edges
from copy import deepcopy
from numbers import Number
from typing import Dict, List
import dace
import sympy
from dace import data, dtypes, registry, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import OrderedMultiDiConnectorGraph
from dace.transformation import transformation as pm


@registry.autoregister_params(singlestate=True)
class ElementWiseArrayOperation(pm.Transformation):
    """ Distributes element-wise array operations.
    """

    _map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(ElementWiseArrayOperation._map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[ElementWiseArrayOperation._map_entry])
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
        map_entry = graph.node(candidate[ElementWiseArrayOperation._map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[self._map_entry]]
        map_exit = graph.exit_node(map_entry)

        if len(map_entry.map.params) > 1:
            raise NotImplementedError

        sz = dace.symbol('commsize', dtype=dace.int32)

        root_name = sdfg.temp_data_name()
        sdfg.add_scalar(root_name, dace.int32, transient=True)
        root_node = graph.add_access(root_name)
        root_tasklet = graph.add_tasklet('_set_root_', {}, {'__out'},
                                         '__out = 0')
        graph.add_edge(root_tasklet, '__out', root_node, None,
                       dace.Memlet.simple(root_name, '0'))

        from dace.libraries.mpi import Scatter, Gather

        inputs = set()
        for src, _, _, _, m in graph.in_edges(map_entry):
            if not isinstance(src, nodes.AccessNode):
                raise NotImplementedError
            desc = src.desc(sdfg)
            if not isinstance(desc, data.Array):
                raise NotImplementedError
            if list(desc.shape) != m.src_subset.size_exact():
                raise NotImplementedError
            inputs.add(src)

        for inp in inputs:
            desc = inp.desc(sdfg)
            if isinstance(desc, data.Scalar):
                raise NotImplementedError
            if isinstance(desc, data.Array):
                if len(desc.shape) > 1:
                    raise NotImplementedError
                local_name, local_arr = sdfg.add_temp_transient(
                    [desc.shape[0] // sz], dtype=desc.dtype, storage=desc.storage)
                local_access = graph.add_access(local_name)
                scatter_node = Scatter('_Scatter_')
                graph.add_edge(inp, None, scatter_node, '_inbuffer',
                               dace.Memlet.from_array(inp.data, desc))
                graph.add_edge(root_node, None, scatter_node, '_root',
                               dace.Memlet.simple(root_name, '0'))
                graph.add_edge(scatter_node, '_outbuffer', local_access, None,
                               dace.Memlet.from_array(local_name, local_arr))
                for e in graph.edges_between(inp, map_entry):
                    graph.add_edge(local_access, None, map_entry, e.dst_conn,
                                   dace.Memlet.from_array(local_name, local_arr))
                    graph.remove_edge(e)
                for e in graph.out_edges(map_entry):
                    if e.data.data == inp.data:
                        e.data.data = local_name
                continue
            raise NotImplementedError

        outputs = set()
        for _, _, dst, _, m in graph.out_edges(map_exit):
            if not isinstance(dst, nodes.AccessNode):
                raise NotImplementedError
            desc = dst.desc(sdfg)
            if not isinstance(desc, data.Array):
                raise NotImplementedError
            if list(desc.shape) != m.dst_subset.size_exact():
                raise NotImplementedError
            outputs.add(dst)

        for out in outputs:
            desc = out.desc(sdfg)
            if isinstance(desc, data.Scalar):
                raise NotImplementedError
            if isinstance(desc, data.Array):
                if len(desc.shape) > 1:
                    raise NotImplementedError
                local_name, local_arr = sdfg.add_temp_transient(
                    [desc.shape[0] // sz], dtype=desc.dtype, storage=desc.storage)
                local_access = graph.add_access(local_name)
                scatter_node = Gather('_Gather_')
                graph.add_edge(local_access, None, scatter_node, '_inbuffer',
                               dace.Memlet.from_array(local_name, local_arr))
                graph.add_edge(root_node, None, scatter_node, '_root',
                               dace.Memlet.simple(root_name, '0'))
                graph.add_edge(scatter_node, '_outbuffer', out, None,
                               dace.Memlet.from_array(out.data, desc))
                for e in graph.edges_between(map_exit, out):
                    graph.add_edge(map_exit, e.src_conn, local_access, None,
                                   dace.Memlet.from_array(local_name, local_arr))
                    graph.remove_edge(e)
                for e in graph.in_edges(map_exit):
                    if e.data.data == out.data:
                        e.data.data = local_name
                continue
            raise NotImplementedError

        new_ranges = [(0, (e + 1) / sz - 1, 1)
                      for _, e, _ in map_entry.map.range]
        map_entry.range = subsets.Range(new_ranges)


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


@registry.autoregister_params(singlestate=True)
class Reduction1Operation(pm.Transformation):
    """ Detects reduction1 operations.
    """

    map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(Reduction1Operation.map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[Reduction1Operation.map_entry])
        map_exit = graph.exit_node(map_entry)
        params = [dace.symbol(p) for p in map_entry.map.params]

        outputs = dict()
        for _, _, _, _, m in graph.out_edges(map_exit):
            if not m.wcr:
                return False
            desc = sdfg.arrays[m.data]
            if desc not in outputs.keys():
                outputs[desc] = []
            outputs[desc].append(m.subset)
        
        for desc, accesses in outputs.items():
            if isinstance(desc, dace.data.Scalar):
                continue
            elif isinstance(desc, (dace.data.Array, dace.data.View)):
                for a in accesses:
                    if a.num_elements() != 1:
                        return False
            else:
                return False

        return True

    @staticmethod
    def match_to_str(graph: dace.SDFGState, candidate: Dict[pm.PatternNode,
                                                            int]) -> str:
        map_entry = graph.node(candidate[Reduction1Operation.map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        pass


@registry.autoregister_params(singlestate=True)
class ReductionNOperation(pm.Transformation):
    """ Detects reductionN operations.
    """

    map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(ReductionNOperation.map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[ReductionNOperation.map_entry])
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

        outputs = dict()
        for _, _, _, _, m in graph.in_edges(map_exit):
            desc = sdfg.arrays[m.data]
            if not m.wcr:
                if desc not in inputs.keys():
                    return False
                access_found = False
                for a in inputs[desc]:
                    if a == m.subset:
                        access_found = True
                        break
                if not access_found:
                    return False
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
                    if len(unmatched_indices) == len(params):
                        return False
            else:
                return False

        return True

    @staticmethod
    def match_to_str(graph: dace.SDFGState, candidate: Dict[pm.PatternNode,
                                                            int]) -> str:
        map_entry = graph.node(candidate[ReductionNOperation.map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        pass