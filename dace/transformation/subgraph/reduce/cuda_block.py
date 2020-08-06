""" This module contains classes that implement the cuda-block transformation.
"""

from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, utils
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg

from dace.frontend.operations import detect_reduction_type
from dace.transformation.subgraph import helpers

from dace.transformation.dataflow.local_storage import LocalStorage

from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib

import timeit



@registry.autoregister_params(singlestate=True)
@make_properties
class CUDABlockAllReduce(pattern_matching.Transformation):
    """ Implements the CUDABlockAllReduce transformation.
        Takes a cuda block reduce node, transforms it to a block reduce node,
        warps it in outer maps and creates an if-output of thread0
        to a newly created shared memory container
    """

    _reduce = stdlib.Reduce()

    debug = Property(desc="Debug Info",
                     dtype = bool,
                     default = True)
    collapse = Property(desc = "Collapse Reduction for better viewability",
                        dtype = bool,
                        default = False)

    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(CUDABlockAllReduce._reduce)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict = False):
        # TODO: Work in progress
        reduce_node = graph.nodes()[candidate[CUDABlockAllReduce._reduce]]
        inedge = graph.in_edges(reduce_node)[0]
        scope_dict = graph.scope_dict()
        if reduce_node.implementation != 'CUDA (block)':
            return False
        if sdfg.data(inedge.data.data).total_size == 1:
            return False
        if not scope_dict[reduce_node]:
            # reduce node must not be top level
            # but inside a block already
            return False

        # finally, check whether data_dims of incoming
        # memlet is equal to length of data axes
        if inedge.data.subset.data_dims() != len(reduce_node.axes):
            return False

        # good to go
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        reduce = candidate[CUDABlockAllReduce._reduce]
        return str(reduce)

    def redirect_edge(self, graph, edge, new_src = None, new_src_conn = None ,
                                         new_dst = None, new_dst_conn = None, new_data = None ):

        data = new_data if new_data else edge.data
        if new_src and new_dst:
            ret = graph.add_edge(new_src, new_src_conn, new_dst, new_dst_conn, data)
            graph.remove_edge(edge)
        elif new_src:
            ret = graph.add_edge(new_src, new_src_conn, edge.dst, edge.dst_conn, data)
            graph.remove_edge(edge)
        elif new_dst:
            ret = graph.add_edge(edge.src, edge.src_conn, new_dst, new_dst_conn, data)
            graph.remove_edge(edge)
        else:
            print("WARNING: redirect_edge has been called in vain")
        return ret

    def apply(self, sdfg, strict = False):
        """ Create a map around the BlockReduce node
            with in and out transients in registers
            and an if tasklet that redirects the output
            of thread 0 to a shared memory transient
        """

        ### define some useful vars
        graph = sdfg.nodes()[self.state_id]
        reduce_node = graph.nodes()[self.subgraph[CUDABlockAllReduce._reduce]]
        in_edge = graph.in_edges(reduce_node)[0]
        out_edge = graph.out_edges(reduce_node)[0]

        axes = reduce_node.axes

        ### add a map that encloses the reduce node
        (new_entry, new_exit) = graph.add_map(
                      name = 'inner_reduce_block',
                      ndrange = {'i'+str(i): f'{rng[0]}:{rng[1]+1}:{rng[2]}'
                                for (i,rng) in enumerate(in_edge.data.subset)
                                if i in axes},
                      schedule = dtypes.ScheduleType.Default)

        if self.debug:
            print(f"CUDABlockAllReduce on range(s) {new_entry.map.range}")
        map = new_entry.map
        self.redirect_edge(graph, in_edge, new_dst = new_entry)
        self.redirect_edge(graph, out_edge, new_src = new_exit)
        subset_in = subsets.Range([in_edge.data.subset[i] if i not in axes
                                   else (new_entry.map.params[0],new_entry.map.params[0],1)
                                   for i in range(len(in_edge.data.subset))])
        memlet_in = Memlet(data = in_edge.data.data,
                           volume = 1,
                           subset = subset_in)
        memlet_out = dcpy(out_edge.data)
        graph.add_edge(u = new_entry, u_connector = None,
                       v = reduce_node,v_connector = None,
                       memlet = memlet_in)
        graph.add_edge(u = reduce_node, u_connector = None,
                       v = new_exit, v_connector = None,
                       memlet = memlet_out)

        ### add in and out local storage

        in_local_storage_subgraph = {
            LocalStorage._node_a: graph.nodes().index(new_entry),
            LocalStorage._node_b: graph.nodes().index(reduce_node)
        }
        out_local_storage_subgraph = {
            LocalStorage._node_a: graph.nodes().index(reduce_node),
            LocalStorage._node_b: graph.nodes().index(new_exit)
        }

        local_storage = LocalStorage(sdfg.sdfg_id,
                                     self.state_id,
                                     in_local_storage_subgraph,
                                     0)

        local_storage.array = in_edge.data.data
        local_storage.apply(sdfg)
        in_transient = local_storage._data_node
        sdfg.data(in_transient.data).storage = dtypes.StorageType.Register

        local_storage = LocalStorage(sdfg.sdfg_id,
                                     self.state_id,
                                     out_local_storage_subgraph,
                                     0)
        local_storage.array = out_edge.data.data
        local_storage.apply(sdfg)
        out_transient = local_storage._data_node
        sdfg.data(out_transient.data).storage = dtypes.StorageType.Register

        # hack: swap edges as local_storage does not work correctly here
        # TODO: beautify
        e1 = graph.in_edges(out_transient)[0]
        e2 = graph.out_edges(out_transient)[0]
        e1.data.data = dcpy(e2.data.data)
        e1.data.subset = dcpy(e2.data.subset)

        ### add an if tasket and diverge
        code = 'if '
        for (i,param) in enumerate(new_entry.map.params):
            code += (param + '==0')
            if i < len(axes) - 1:
                code += ' and '
        code += ':\n'
        code += '\tout=inp'

        tasklet_node = graph.add_tasklet(name = 'block_reduce_write',
                                         inputs = ['inp'],
                                         outputs = ['out'],
                                         code = code)

        edge_out_outtrans = graph.out_edges(out_transient)[0]
        edge_out_innerexit = graph.out_edges(new_exit)[0]
        self.redirect_edge(graph, edge_out_outtrans,
                           new_dst = tasklet_node, new_dst_conn = 'inp')
        e = graph.add_edge(u = tasklet_node, u_connector = 'out',
                           v = new_exit, v_connector = None,
                           memlet = dcpy(edge_out_innerexit.data))
        # set dynamic with volume 0 FORNOW
        e.data.volume = 0
        e.data.dynamic = True

        ### set reduce_node axes to all (needed)
        reduce_node.axes = None

        if self.collapse:
            new_entry.is_collapsed = True
        # fill scope connectors, done.
        sdfg.fill_scope_connectors()
        return
