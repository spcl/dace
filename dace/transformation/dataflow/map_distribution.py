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
from dace.transformation.subgraph.helpers import subgraph_from_maps
from functools import reduce


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

        if "commsize" in map_entry.map.range.free_symbols:
            return False
        if "Px" in map_entry.map.range.free_symbols:
            return False
        if "Py" in map_entry.map.range.free_symbols:
            return False

        # If the map iterators are used in the code of a Tasklet,
        # then we cannot flatten them (currently).
        # See, for example, samples/simple/mandelbrot.py
        for node in subgraph_from_maps(sdfg, graph, [map_entry]):
            if isinstance(node, dace.nodes.CodeNode):
                for p in params:
                    if str(p) in node.free_symbols:
                        return False

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

        sz = dace.symbol('commsize', dtype=dace.int32)

        def _prod(sequence):
            return reduce(lambda a, b: a * b, sequence, 1)

        # NOTE: Maps with step in their ranges are currently not supported
        if len(map_entry.map.params) == 1:
            params = map_entry.map.params
            ranges = [(0, (e-b+1)/sz - 1, 1) for b, e, _ in map_entry.map.range]
            strides = [1]
        else:
            params = ['__iflat']
            sizes = map_entry.map.range.size_exact()
            total_size = _prod(sizes)
            ranges = [(0, (total_size)/sz - 1, 1)]
            strides = [_prod(sizes[i + 1:]) for i in range(len(sizes))]

        root_name = sdfg.temp_data_name()
        sdfg.add_scalar(root_name, dace.int32, transient=True)
        root_node = graph.add_access(root_name)
        root_tasklet = graph.add_tasklet('_set_root_', {}, {'__out'},
                                         '__out = 0')
        graph.add_edge(root_tasklet, '__out', root_node, None,
                       dace.Memlet.simple(root_name, '0'))

        from dace.libraries.mpi import Bcast, Scatter, Gather

        inputs = set()
        for src, _, _, _, m in graph.in_edges(map_entry):
            if not isinstance(src, nodes.AccessNode):
                raise NotImplementedError
            desc = src.desc(sdfg)
            if not isinstance(desc, (data.Scalar, data.Array)):
                raise NotImplementedError
            if list(desc.shape) != m.src_subset.size_exact():
                # Second attempt
                # TODO: We need a solution for symbols not matching
                if str(list(desc.shape)) != str(m.src_subset.size_exact()):
                    raise NotImplementedError
            inputs.add(src)

        for inp in inputs:
            desc = inp.desc(sdfg)

            if isinstance(desc, data.Scalar):
                local_access = graph.add_access(inp.data)
                bcast_node = Bcast('_Bcast_')
                graph.add_edge(inp, None, bcast_node, '_inbuffer',
                               dace.Memlet.from_array(inp.data, desc))
                graph.add_edge(root_node, None, bcast_node, '_root',
                               dace.Memlet.simple(root_name, '0'))
                graph.add_edge(bcast_node, '_outbuffer', local_access, None,
                               dace.Memlet.from_array(inp.data, desc))
                for e in graph.edges_between(inp, map_entry):
                    graph.add_edge(local_access, None, map_entry, e.dst_conn,
                                   dace.Memlet.from_array(inp.data, desc))
                    graph.remove_edge(e)

            elif isinstance(desc, data.Array):

                local_name, local_arr = sdfg.add_temp_transient(
                    [(desc.total_size) // sz], dtype=desc.dtype,
                    storage=desc.storage)
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
                        e.data = dace.Memlet.simple(local_name, params[0])

            else:
                raise NotImplementedError

        outputs = set()
        for _, _, dst, _, m in graph.out_edges(map_exit):
            if not isinstance(dst, nodes.AccessNode):
                raise NotImplementedError
            desc = dst.desc(sdfg)
            if not isinstance(desc, data.Array):
                raise NotImplementedError
            try:
                if list(desc.shape) != m.dst_subset.size_exact():
                    # Second attempt
                    # TODO: We need a solution for symbols not matching
                    if str(list(desc.shape)) != str(m.dst_subset.size_exact()):
                        raise NotImplementedError
            except AttributeError:
                if list(desc.shape) != m.subset.size_exact():
                    # Second attempt
                    # TODO: We need a solution for symbols not matching
                    if str(list(desc.shape)) != str(m.subset.size_exact()):
                        raise NotImplementedError
            outputs.add(dst)

        for out in outputs:
            desc = out.desc(sdfg)
            if isinstance(desc, data.Scalar):
                raise NotImplementedError
            elif isinstance(desc, data.Array):
                local_name, local_arr = sdfg.add_temp_transient(
                    [(desc.total_size) // sz], dtype=desc.dtype,
                    storage=desc.storage)
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
                        e.data = dace.Memlet.simple(local_name, params[0])
            else:
                raise NotImplementedError

        map_entry.map.params = params
        map_entry.map.range = subsets.Range(ranges)


@registry.autoregister_params(singlestate=True)
class ElementWiseArrayOperation2D(pm.Transformation):
    """ Distributes element-wise array operations.
    """

    _map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(ElementWiseArrayOperation2D._map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[ElementWiseArrayOperation2D._map_entry])
        map_exit = graph.exit_node(map_entry)
        params = [dace.symbol(p) for p in map_entry.map.params]
        if len(params) != 2:
            return False

        if "commsize" in map_entry.map.range.free_symbols:
            return False
        if "Px" in map_entry.map.range.free_symbols:
            return False
        if "Py" in map_entry.map.range.free_symbols:
            return False

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
                if len(desc.shape) != 2:
                    return False
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
                if len(desc.shape) != 2:
                    return False
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
        map_entry = graph.node(candidate[ElementWiseArrayOperation2D._map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)


    def apply(self, sdfg: dace.SDFG):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[self._map_entry]]
        map_exit = graph.exit_node(map_entry)

        sz = dace.symbol('commsize', dtype=dace.int32, integer=True, positive=True)
        Px = dace.symbol('Px', dtype=dace.int32, integer=True, positive=True)
        Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)

        def _prod(sequence):
            return reduce(lambda a, b: a * b, sequence, 1)

        # NOTE: Maps with step in their ranges are currently not supported
        if len(map_entry.map.params) == 2:
            params = map_entry.map.params
            ranges = [None] * 2
            b, e, _ = map_entry.map.range[0]
            ranges[0] = (0, (e-b+1)/Px - 1, 1)
            b, e, _ = map_entry.map.range[1]
            ranges[1] = (0, (e-b+1)/Py - 1, 1)
            strides = [1]
        else:
            params = ['__iflat']
            sizes = map_entry.map.range.size_exact()
            total_size = _prod(sizes)
            ranges = [(0, (total_size)/sz - 1, 1)]
            strides = [_prod(sizes[i + 1:]) for i in range(len(sizes))]

        root_name = sdfg.temp_data_name()
        sdfg.add_scalar(root_name, dace.int32, transient=True)
        root_node = graph.add_access(root_name)
        root_tasklet = graph.add_tasklet('_set_root_', {}, {'__out'},
                                         '__out = 0')
        graph.add_edge(root_tasklet, '__out', root_node, None,
                       dace.Memlet.simple(root_name, '0'))

        from dace.libraries.mpi import Bcast
        from dace.libraries.pblas import BlockCyclicScatter, BlockCyclicGather

        inputs = set()
        for src, _, _, _, m in graph.in_edges(map_entry):
            if not isinstance(src, nodes.AccessNode):
                raise NotImplementedError
            desc = src.desc(sdfg)
            if not isinstance(desc, (data.Scalar, data.Array)):
                raise NotImplementedError
            if list(desc.shape) != m.src_subset.size_exact():
                # Second attempt
                # TODO: We need a solution for symbols not matching
                if str(list(desc.shape)) != str(m.src_subset.size_exact()):
                    raise NotImplementedError
            inputs.add(src)

        for inp in inputs:
            desc = inp.desc(sdfg)

            if isinstance(desc, data.Scalar):
                local_access = graph.add_access(inp.data)
                bcast_node = Bcast('_Bcast_')
                graph.add_edge(inp, None, bcast_node, '_inbuffer',
                               dace.Memlet.from_array(inp.data, desc))
                graph.add_edge(root_node, None, bcast_node, '_root',
                               dace.Memlet.simple(root_name, '0'))
                graph.add_edge(bcast_node, '_outbuffer', local_access, None,
                               dace.Memlet.from_array(inp.data, desc))
                for e in graph.edges_between(inp, map_entry):
                    graph.add_edge(local_access, None, map_entry, e.dst_conn,
                                   dace.Memlet.from_array(inp.data, desc))
                    graph.remove_edge(e)

            elif isinstance(desc, data.Array):

                local_name, local_arr = sdfg.add_temp_transient(
                    [(desc.shape[0]) // Px, (desc.shape[1]) // Py],
                    dtype=desc.dtype, storage=desc.storage)
                local_access = graph.add_access(local_name)
                bsizes_name, bsizes_arr = sdfg.add_temp_transient(
                    (2,), dtype=dace.int32)
                bsizes_access = graph.add_access(bsizes_name)
                bsizes_tasklet = nodes.Tasklet(
                    '_set_bsizes_',
                    {}, {'__out'},
                    "__out[0] = {x}; __out[1] = {y}".format(
                        x=(desc.shape[0]) // Px, y=(desc.shape[1]) // Py))
                graph.add_edge(bsizes_tasklet, '__out', bsizes_access, None,
                               dace.Memlet.from_array(bsizes_name, bsizes_arr))
                gdesc_name, gdesc_arr = sdfg.add_temp_transient(
                    (9,), dtype=dace.int32)
                gdesc_access = graph.add_access(gdesc_name)
                ldesc_name, ldesc_arr = sdfg.add_temp_transient(
                    (9,), dtype=dace.int32)
                ldesc_access = graph.add_access(ldesc_name)
                scatter_node = BlockCyclicScatter('_Scatter_')
                graph.add_edge(inp, None, scatter_node, '_inbuffer',
                               dace.Memlet.from_array(inp.data, desc))
                graph.add_edge(bsizes_access, None, scatter_node, '_block_sizes',
                               dace.Memlet.from_array(bsizes_name, bsizes_arr))
                graph.add_edge(scatter_node, '_outbuffer', local_access, None,
                               dace.Memlet.from_array(local_name, local_arr))
                graph.add_edge(scatter_node, '_gdescriptor', gdesc_access, None,
                               dace.Memlet.from_array(gdesc_name, gdesc_arr))
                graph.add_edge(scatter_node, '_ldescriptor', ldesc_access, None,
                               dace.Memlet.from_array(ldesc_name, ldesc_arr))
                for e in graph.edges_between(inp, map_entry):
                    graph.add_edge(local_access, None, map_entry, e.dst_conn,
                                   dace.Memlet.from_array(local_name, local_arr))
                    graph.remove_edge(e)
                for e in graph.out_edges(map_entry):
                    if e.data.data == inp.data:
                        e.data.data = local_name

            else:
                raise NotImplementedError

        outputs = set()
        for _, _, dst, _, m in graph.out_edges(map_exit):
            if not isinstance(dst, nodes.AccessNode):
                raise NotImplementedError
            desc = dst.desc(sdfg)
            if not isinstance(desc, data.Array):
                raise NotImplementedError
            try:
                if list(desc.shape) != m.dst_subset.size_exact():
                    # Second attempt
                    # TODO: We need a solution for symbols not matching
                    if str(list(desc.shape)) != str(m.dst_subset.size_exact()):
                        raise NotImplementedError
            except AttributeError:
                if list(desc.shape) != m.subset.size_exact():
                    # Second attempt
                    # TODO: We need a solution for symbols not matching
                    if str(list(desc.shape)) != str(m.subset.size_exact()):
                        raise NotImplementedError
            outputs.add(dst)

        for out in outputs:
            desc = out.desc(sdfg)
            if isinstance(desc, data.Scalar):
                raise NotImplementedError
            elif isinstance(desc, data.Array):
                local_name, local_arr = sdfg.add_temp_transient(
                    [(desc.shape[0]) // Px, (desc.shape[1]) // Py],
                    dtype=desc.dtype, storage=desc.storage)
                local_access = graph.add_access(local_name)
                bsizes_name, bsizes_arr = sdfg.add_temp_transient(
                    (2,), dtype=dace.int32)
                bsizes_access = graph.add_access(bsizes_name)
                bsizes_tasklet = nodes.Tasklet(
                    '_set_bsizes_',
                    {}, {'__out'},
                    "__out[0] = {x}; __out[1] = {y}".format(
                        x=(desc.shape[0]) // Px, y=(desc.shape[1]) // Py))
                graph.add_edge(bsizes_tasklet, '__out', bsizes_access, None,
                               dace.Memlet.from_array(bsizes_name, bsizes_arr))
                scatter_node = BlockCyclicGather('_Gather_')
                graph.add_edge(local_access, None, scatter_node, '_inbuffer',
                               dace.Memlet.from_array(local_name, local_arr))
                graph.add_edge(bsizes_access, None, scatter_node, '_block_sizes',
                               dace.Memlet.from_array(bsizes_name, bsizes_arr))
                graph.add_edge(scatter_node, '_outbuffer', out, None,
                               dace.Memlet.from_array(out.data, desc))

                for e in graph.edges_between(map_exit, out):
                    graph.add_edge(map_exit, e.src_conn, local_access, None,
                                   dace.Memlet.from_array(local_name, local_arr))
                    graph.remove_edge(e)
                for e in graph.in_edges(map_exit):
                    if e.data.data == out.data:
                        e.data.data = local_name
            else:
                raise NotImplementedError

        map_entry.map.params = params
        map_entry.map.range = subsets.Range(ranges)


@registry.autoregister_params(singlestate=True, strict=False)
class RedundantComm2D(pm.Transformation):
    """ Implements the redundant communication removal transformation,
        applied when data are scattered and immediately gathered,
        but never used anywhere else. """

    in_array = pm.PatternNode(nodes.AccessNode)
    gather = pm.PatternNode(nodes.Tasklet)
    mid_array = pm.PatternNode(nodes.AccessNode)
    scatter = pm.PatternNode(nodes.Tasklet)
    out_array = pm.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(RedundantComm2D.in_array,
                                   RedundantComm2D.gather,
                                   RedundantComm2D.mid_array,
                                   RedundantComm2D.scatter,
                                   RedundantComm2D.out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        gather = graph.nodes()[candidate[RedundantComm2D.gather]]
        if '_block_sizes' not in gather.in_connectors:
            return False
        scatter = graph.nodes()[candidate[RedundantComm2D.scatter]]
        if '_gdescriptor' not in scatter.out_connectors:
            return False
        in_array = graph.nodes()[candidate[RedundantComm2D.in_array]]
        out_array = graph.nodes()[candidate[RedundantComm2D.out_array]]
        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)
        if len(in_desc.shape) != 2:
            return False
        if in_desc.shape == out_desc.shape:
            return True
        return False

    @staticmethod
    def match_to_str(graph, candidate):
        in_array = graph.nodes()[candidate[RedundantComm2D.in_array]]

        return "Remove " + str(in_array)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        in_array = self.in_array(sdfg)
        gather = self.gather(sdfg)
        mid_array = self.mid_array(sdfg)
        scatter = self.scatter(sdfg)
        out_array = self.out_array(sdfg)

        in_desc = sdfg.arrays[in_array.data]
        out_desc = sdfg.arrays[out_array.data]

        for e in graph.in_edges(gather):
            if e.src != in_array:
                if graph.in_degree(e.src) == 0 and graph.out_degree(e.src) == 1:
                    graph.remove_edge(e)
                    graph.remove_node(e.src)
        for e in graph.out_edges(scatter):
            if e.dst != out_array:
                if graph.in_degree(e.dst) == 1 and graph.out_degree(e.dst) == 0:
                    graph.remove_edge(e)
                    graph.remove_node(e.dst)
        
        for e in graph.out_edges(out_array):
            path = graph.memlet_tree(e)
            for e2 in path:
                if e2.data.data == out_array.data:
                    e2.data.data = in_array.data
            graph.remove_edge(e)
            graph.add_edge(in_array, None, e.dst, e.dst_conn,
                           dace.Memlet.from_array(in_array, in_desc))

        graph.remove_node(gather)
        graph.remove_node(mid_array)
        graph.remove_node(scatter)
        graph.remove_node(out_array)


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
