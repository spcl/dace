# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement the reduce-map transformation.
"""

from dace import dtypes
from dace.sdfg import SDFG, nodes, utils, graph
from dace.memlet import Memlet
from dace.sdfg.scope import ScopeTree
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.data import View

from dace.frontend.operations import detect_reduction_type
from dace.sdfg.propagation import propagate_memlets_scope

from copy import deepcopy as dcpy
from typing import List

import numpy as np

import timeit


@make_properties
class ReduceExpansion(transformation.SingleStateTransformation):
    """ Implements the ReduceExpansion transformation.
        Expands a Reduce node into inner and outer map components,
        where the outer map consists of the axes not being reduced.
        A new reduce node is created inside the inner map.
        Special cases where e.g reduction identities are not defined
        and arrays being reduced to already exist are handled
        on the fly.
    """

    import dace.libraries.standard as stdlib  # Avoid slow imports

    reduce = transformation.PatternNode(stdlib.Reduce)

    debug = Property(desc="Debug Info", dtype=bool, default=False)

    create_in_transient = Property(desc="Create local in-transient" "in registers", dtype=bool, default=False)

    create_out_transient = Property(desc="Create local out-transient" "in registers", dtype=bool, default=False)

    reduce_implementation = Property(desc="Reduce implementation of inner reduce. If specified,"
                                     "overrides any existing implementations",
                                     dtype=str,
                                     default=None,
                                     choices=[
                                         'pure', 'OpenMP', 'CUDA (device)', 'CUDA (block)', 'CUDA (block allreduce)',
                                         'CUDA (warp)', 'CUDA (warp allreduce)'
                                     ],
                                     allow_none=True)

    reduction_type_update = {
        dtypes.ReductionType.Max: 'out = max(reduction_in, array_in)',
        dtypes.ReductionType.Min: 'out = min(reduction_in, array_in)',
        dtypes.ReductionType.Sum: 'out = reduction_in + array_in',
        dtypes.ReductionType.Product: 'out = reduction_in * array_in',
        dtypes.ReductionType.Bitwise_And: 'out = reduction_in & array_in',
        dtypes.ReductionType.Bitwise_Or: 'out = reduction_in | array_in',
        dtypes.ReductionType.Bitwise_Xor: 'out = reduction_in ^ array_in',
        dtypes.ReductionType.Logical_And: 'out = reduction_in and array_in',
        dtypes.ReductionType.Logical_Or: 'out = reduction_in or array_in',
        dtypes.ReductionType.Logical_Xor: 'out = reduction_in != array_in'
    }

    reduction_type_identity = {
        dtypes.ReductionType.Sum: 0,
        dtypes.ReductionType.Product: 1,
        dtypes.ReductionType.Bitwise_Or: 0,
        dtypes.ReductionType.Logical_And: True,
        dtypes.ReductionType.Logical_Or: False
    }

    @classmethod
    def expressions(cls):
        return [utils.node_path_graph(cls.reduce)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        reduce_node = self.reduce
        inedge = graph.in_edges(reduce_node)[0]
        input_dims = inedge.data.subset.dims()
        axes = reduce_node.axes
        # we cannot expand upon full reduction
        if axes is None:
            return False

        # we cannot expand if there are not outer dimensions
        for (i, sz) in enumerate(inedge.data.subset.size()):
            if i not in reduce_node.axes and sz != 1:
                break
        else:
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        """ Splits the data dimension into an inner and outer dimension,
            where the inner dimension are the reduction axes and the
            outer axes the complement. Pushes the reduce inside a new
            map consisting of the complement axes.
        """
        reduce_node = self.reduce
        self.expand(sdfg, graph, reduce_node)

    def expand(self, sdfg: SDFG, graph: SDFGState, reduce_node):
        """ Splits the data dimension into an inner and outer dimension,
            where the inner dimension are the reduction axes and the
            outer axes the complement. Pushes the reduce inside a new
            map consisting of the complement axes.

        """

        # get out storage node, might be hidden behind view node
        out_data = graph.out_edges(reduce_node)[0].data
        out_storage_node = reduce_node
        while not isinstance(out_storage_node, nodes.AccessNode):
            out_storage_node = graph.out_edges(out_storage_node)[0].dst

        if isinstance(sdfg.data(out_storage_node.data), View):
            out_storage_node = graph.out_edges(out_storage_node)[0].dst
            while not isinstance(out_storage_node, nodes.AccessNode):
                out_storage_node = graph.out_edges(out_storage_node)[0].dst

        # get other useful quantities from the original reduce node
        wcr = reduce_node.wcr
        identity = reduce_node.identity
        implementation = reduce_node.implementation

        # remove the reduce identity, will get reassigned after expansion
        reduce_node.identity = None
        # expand the reduce node
        in_edge = graph.in_edges(reduce_node)[0]
        nsdfg = self._expand_reduce(sdfg, graph, reduce_node)
        # find the new nodes in the nested sdfg created
        nstate = nsdfg.sdfg.nodes()[0]
        for node, scope in nstate.scope_dict().items():
            if isinstance(node, nodes.MapEntry):
                if scope is None:
                    outer_entry = node
                else:
                    inner_entry = node
            if isinstance(node, nodes.Tasklet):
                tasklet_node = node

        inner_exit = nstate.exit_node(inner_entry)
        outer_exit = nstate.exit_node(outer_entry)

        # find earliest parent read-write occurrence of array onto which the reduction is performed: BFS

        if self.create_out_transient:
            queue = [nsdfg]
            enqueued = set()
            array_closest_ancestor = None

            while len(queue) > 0:
                current = queue.pop()
                if isinstance(current, nodes.AccessNode):
                    if current.data == out_storage_node.data:
                        # it suffices to find the first node
                        # no matter what access (ReadWrite or Read)
                        array_closest_ancestor = current
                        break
                for in_edge in graph.in_edges(current):
                    if in_edge.src not in enqueued:
                        queue.append(in_edge.src)
                        enqueued.add(in_edge.src)

            if self.debug and array_closest_ancestor:
                print(f"ReduceExpansion::Closest ancestor={array_closest_ancestor}")
            elif self.debug:
                print("ReduceExpansion::No closest ancestor found")

        if self.create_out_transient:
            # create an out transient between inner and outer map exit
            array_out = nstate.out_edges(outer_exit)[0].data.data

            from dace.transformation.dataflow.local_storage import LocalStorage
            local_storage_subgraph = {
                LocalStorage.node_a: nsdfg.sdfg.nodes()[0].nodes().index(inner_exit),
                LocalStorage.node_b: nsdfg.sdfg.nodes()[0].nodes().index(outer_exit)
            }
            nsdfg_id = nsdfg.sdfg.sdfg_list.index(nsdfg.sdfg)
            nstate_id = 0
            local_storage = LocalStorage(nsdfg_id, nstate_id, local_storage_subgraph, 0)
            local_storage.array = array_out
            local_storage.apply(graph, nsdfg.sdfg)
            out_transient_node_inner = local_storage._data_node

            # push to register
            nsdfg.sdfg.data(out_transient_node_inner.data).storage = dtypes.StorageType.Register

            # remove WCRs from all edges where possible if there is no
            # prior occurrence
            if array_closest_ancestor is None:
                nstate.out_edges(outer_exit)[0].data.wcr = None
                nstate.out_edges(out_transient_node_inner)[0].data.wcr = None
                nstate.out_edges(out_transient_node_inner)[0].data.volume = 1
        else:

            # remove WCR from outer exit
            nstate.out_edges(outer_exit)[0].data.wcr = None

        if self.create_in_transient:
            # create an in-transient between inner and outer map entry
            array_in = nstate.in_edges(outer_entry)[0].data.data

            from dace.transformation.dataflow.local_storage import LocalStorage
            local_storage_subgraph = {
                LocalStorage.node_a: nsdfg.sdfg.nodes()[0].nodes().index(outer_entry),
                LocalStorage.node_b: nsdfg.sdfg.nodes()[0].nodes().index(inner_entry)
            }

            nsdfg_id = nsdfg.sdfg.sdfg_list.index(nsdfg.sdfg)
            nstate_id = 0
            local_storage = LocalStorage(nsdfg_id, nstate_id, local_storage_subgraph, 0)
            local_storage.array = array_in
            local_storage.apply(graph, nsdfg.sdfg)
            in_transient_node_inner = local_storage._data_node

            # push to register
            nsdfg.sdfg.data(in_transient_node_inner.data).storage = dtypes.StorageType.Register

        # inline fuse back our nested SDFG
        from dace.transformation.interstate import InlineSDFG
        inline_sdfg = InlineSDFG(sdfg.sdfg_list.index(sdfg),
                                 sdfg.nodes().index(graph), {InlineSDFG._nested_sdfg: graph.nodes().index(nsdfg)}, 0)
        inline_sdfg.apply(graph, sdfg)

        new_schedule = dtypes.ScheduleType.Default
        new_implementation = self.reduce_implementation \
                             if self.reduce_implementation is not None \
                             else implementation
        new_axes = dcpy(reduce_node.axes)

        reduce_node_new = graph.add_reduce(wcr=wcr, axes=new_axes, schedule=new_schedule, identity=identity)
        reduce_node_new.implementation = new_implementation
        # replace inner map with new reduction node
        edge_tmp = graph.in_edges(inner_entry)[0]
        memlet_src_reduce = dcpy(edge_tmp.data)
        graph.add_edge(edge_tmp.src, edge_tmp.src_conn, reduce_node_new, None, memlet_src_reduce)

        edge_tmp = graph.out_edges(inner_exit)[0]
        memlet_reduce_dst = Memlet(data=edge_tmp.data.data, volume=1, subset=edge_tmp.data.subset)

        graph.add_edge(reduce_node_new, None, edge_tmp.dst, edge_tmp.dst_conn, memlet_reduce_dst)

        identity_tasklet = graph.out_edges(inner_entry)[0].dst
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)
        graph.remove_node(identity_tasklet)

        # propagate scope for correct volumes
        scope_tree = ScopeTree(outer_entry, outer_exit)
        scope_tree.parent = ScopeTree(None, None)
        propagate_memlets_scope(sdfg, graph, scope_tree)
        sdfg.validate()

        # create variables for outside access
        self._reduce = reduce_node_new
        self._outer_entry = outer_entry

        if identity is None and self.create_out_transient:
            if self.debug:
                print("ReduceExpansion::Trying to infer reduction WCR type due to out transient created")
            # set the reduction identity accordingly so that the correct
            # blank result is written to the out_transient node
            # we use default values deducted from the reduction type
            reduction_type = detect_reduction_type(wcr)
            try:
                reduce_node_new.identity = self.reduction_type_identity[reduction_type]
            except KeyError:

                if reduction_type == dtypes.ReductionType.Min:
                    reduce_node_new.identity = dtypes.max_value(sdfg.arrays[out_storage_node.data].dtype)
                elif reduction_type == dtypes.ReductionType.Max:
                    reduce_node_new.identity = dtypes.min_value(sdfg.arrays[out_storage_node.data].dtype)
                else:
                    raise ValueError(f"Cannot infer reduction identity."
                                     "Please specify the identity of node"
                                     "{reduce_node_new}")

        return

    def _expand_reduce(self, sdfg, state, node):
        # expands a reduce into two nested maps
        # taken from legacy expand_reduce.py

        node.validate(sdfg, state)
        inedge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        outedge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        input_dims = len(inedge.data.subset)
        output_dims = len(outedge.data.subset)
        input_data = sdfg.arrays[inedge.data.data]
        output_data = sdfg.arrays[outedge.data.data]

        # Standardize axes
        axes = node.axes if node.axes else [i for i in range(input_dims)]

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        nsdfg.add_array('_in',
                        inedge.data.subset.size(),
                        input_data.dtype,
                        strides=input_data.strides,
                        storage=input_data.storage)

        nsdfg.add_array('_out',
                        outedge.data.subset.size(),
                        output_data.dtype,
                        strides=output_data.strides,
                        storage=output_data.storage)

        if node.identity is not None:
            raise ValueError("Node identity has to be None at this point.")
        else:
            nstate = nsdfg.add_state()
        # END OF INIT

        # (If axes != all) Add outer map, which corresponds to the output range
        if len(axes) != input_dims:
            # Interleave input and output axes to match input memlet
            ictr, octr = 0, 0
            input_subset = []
            for i in range(input_dims):
                if i in axes:
                    input_subset.append('_i%d' % ictr)
                    ictr += 1
                else:
                    input_subset.append('_o%d' % octr)
                    octr += 1

            output_size = outedge.data.subset.size()

            ome, omx = nstate.add_map(
                'reduce_output', {'_o%d' % i: '0:%s' % symstr(sz)
                                  for i, sz in enumerate(outedge.data.subset.size())})
            outm = Memlet.simple('_out', ','.join(['_o%d' % i for i in range(output_dims)]), wcr_str=node.wcr)
            inmm = Memlet.simple('_in', ','.join(input_subset))
        else:
            ome, omx = None, None
            outm = Memlet.simple('_out', '0', wcr_str=node.wcr)
            inmm = Memlet.simple('_in', ','.join(['_i%d' % i for i in range(len(axes))]))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        ime, imx = nstate.add_map(
            'reduce_values',
            {'_i%d' % i: '0:%s' % symstr(inedge.data.subset.size()[axis])
             for i, axis in enumerate(sorted(axes))})

        # Add identity tasklet for reduction
        t = nstate.add_tasklet('identity', {'inp'}, {'out'}, 'out = inp')

        # Connect everything
        r = nstate.add_read('_in')
        w = nstate.add_read('_out')

        if ome:
            nstate.add_memlet_path(r, ome, ime, t, dst_conn='inp', memlet=inmm)
            nstate.add_memlet_path(t, imx, omx, w, src_conn='out', memlet=outm)
        else:
            nstate.add_memlet_path(r, ime, t, dst_conn='inp', memlet=inmm)
            nstate.add_memlet_path(t, imx, w, src_conn='out', memlet=outm)

        # Rename outer connectors and add to node
        inedge._dst_conn = '_in'
        outedge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')

        nsdfg = state.add_nested_sdfg(nsdfg,
                                      sdfg,
                                      node.in_connectors,
                                      node.out_connectors,
                                      schedule=node.schedule,
                                      name=node.name)

        utils.change_edge_dest(state, node, nsdfg)
        utils.change_edge_src(state, node, nsdfg)
        state.remove_node(node)

        return nsdfg
