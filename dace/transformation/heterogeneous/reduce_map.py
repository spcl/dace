""" This module contains classes that implement the reduce-map transformation.
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


from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib

import timeit


@registry.autoregister_params(singlestate=True)
@make_properties
class ReduceMap(pattern_matching.Transformation):
    """ Implements the Reduce-Map transformation.
        Expands a Reduce node into inner and outer map components,
        then introduces a transient for its intermediate output,
        deletes the WCR on the outer map exit
        and transforms the inner map back into a reduction.
    """

    _reduce = stdlib.Reduce()


    debug = Property(desc="Debug Info",
                     dtype = bool,
                     default = True)

    map_transient_to_registers = Property(desc="Push out-transient created inside"
                                                "the reduction into register",
                                                dtype = bool,
                                                default = True)

    create_in_transient = Property(desc = "Create local in-transient",
                                   dtype = bool,
                                   default = False)

    reduce_implementation = Property(desc = "Reduce implementation of inner reduce",
                                     dtype = str,
                                     default = 'pure')

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
        dtypes.ReductionType.Logical_Xor: 'out = reduction_in xor array_in'
    }


    @staticmethod
    def expressions():
        return[
            utils.node_path_graph(ReduceMap._reduce)
        ]
    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict = False):
        reduce_node = candidate[ReduceMap._reduce]
        inedge = graph.in_edges(reduce_node)[0]
        input_dims = inedge.data.subset.data_dims()
        axes = reduce_node.axes
        if axes is None:
            # axes = None -> full reduction, can't expand
            return False
        if len(axes) == input_dims:
            # axes = all  -> full reduction, can't expand
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        reduce = candidate[ReduceMap._reduce]
        return str(reduce)

    def apply(self, sdfg: SDFG, strict = False):
        """ Splits the data dimension into an inner and outer dimension,
            where the inner dimension are the reduction axes and the
            outer axes the complement. Pushes the reduce inside a new
            map consisting of the complement axes.
        """

        graph = sdfg.nodes()[self.state_id]
        reduce_node = graph.nodes()[self.subgraph[ReduceMap._reduce]]
        self.expand(sdfg, graph, reduce_node)

    def expand(self, sdfg, graph, reduce_node):
        """ Splits the data dimension into an inner and outer dimension,
            where the inner dimension are the reduction axes and the
            outer axes the complement. Pushes the reduce inside a new
            map consisting of the complement axes.
        """

        out_storage_node = graph.out_edges(reduce_node)[0].dst
        in_storage_node = graph.in_edges(reduce_node)[0].src
        wcr = reduce_node.wcr
        identity = reduce_node.identity
        schedule = reduce_node.schedule
        implementation = reduce_node.implementation


        # remove the reduce identity
        # we will reassign it later after expanding
        reduce_node.identity = None
        # expand the reduce node
        try:
            in_edge = graph.in_edges(reduce_node)[0]
            nsdfg = self._expand_reduce(sdfg, graph, reduce_node)
        except Exception:
            print(f"Aborting: Could not execute expansion in {reduce_node}")
            raise TypeError("EXPANSION_ABORT")

        # find the new nested sdfg

        nstate = nsdfg.sdfg.nodes()[0]
        for node, scope in nstate.scope_dict().items():
            if isinstance(node, nodes.MapEntry):
                if scope is None:
                    outer_entry = node
                else:
                    inner_entry = node

        inner_exit = nstate.exit_node(inner_entry)
        outer_exit = nstate.exit_node(outer_entry)

        ###### create an out transient between inner and outer map exit
        array_out = nstate.out_edges(outer_exit)[0].data.data

        from dace.transformation.dataflow.local_storage import LocalStorage
        local_storage_subgraph = {
            LocalStorage._node_a: nsdfg.sdfg.nodes()[0].nodes().index(inner_exit),
            LocalStorage._node_b: nsdfg.sdfg.nodes()[0].nodes().index(outer_exit)
        }
        nsdfg_id = nsdfg.sdfg.sdfg_list.index(nsdfg.sdfg)
        nstate_id = 0


        local_storage = LocalStorage(nsdfg_id,
                                     nstate_id,
                                     local_storage_subgraph,
                                     0)
        local_storage.array = array_out
        local_storage.apply(nsdfg.sdfg)
        out_transient_node_inner = local_storage._data_node



        if self.create_in_transient:
            # create an in-transient as well.
            array_in = nstate.in_edges(outer_entry)[0].data.data
            local_storage_subgraph = {
                LocalStorage._node_a: nsdfg.sdfg.nodes()[0].nodes().index(outer_entry),
                LocalStorage._node_b: nsdfg.sdfg.nodes()[0].nodes().index(inner_entry)
            }

            local_storage = LocalStorage(nsdfg_id,
                                         nstate_id,
                                         local_storage_subgraph,
                                         0)
            local_storage.array = array_in
            local_storage.apply(nsdfg.sdfg)
            in_transient_node_inner = local_storage._data_node

            # FORNOW
            nsdfg.sdfg.data(in_transient_node_inner.data).storage = dtypes.StorageType.Default


        if self.map_transient_to_registers:
            nsdfg.sdfg.data(out_transient_node_inner.data).storage = dtypes.StorageType.Register


        # find earliest parent read-write occurrence of array onto which
        # we perform the reduction:
        # do BFS, best complexity O(V+E)

        queue = [nsdfg]
        array_closest_ancestor = None
        while len(queue) > 0:
            current = queue.pop(0)
            if isinstance(current, nodes.AccessNode):
                if current.data == out_storage_node.data:
                    # it suffices to find the first node
                    # no matter what access (ReadWrite or Read)
                    # can't be Read only, else state would be ambiguous
                    array_closest_ancestor = current
                    break
            queue.extend([in_edge.src for in_edge in graph.in_edges(current)])

        # if it doesnt exist:
        #           if non-transient: create data node accessing it
        #           if transient: ancestor_node = none, set_zero on outer node

        shortcut = False
        if (not array_closest_ancestor and sdfg.data(out_storage_node.data).transient) \
                                        or identity is not None:
            if self.debug:
                print("ReduceMap::Shortcut applied")
            # we are lucky
            shortcut = True
            nstate.out_edges(out_transient_node_inner)[0].data.wcr = None
            nstate.out_edges(out_transient_node_inner)[0].data.num_accesses = 1
            nstate.out_edges(outer_exit)[0].data.wcr = None


        else:
            if self.debug:
                print("ReduceMap::No shortcut, operating with ancestor", array_closest_ancestor)
            array_closest_ancestor = nodes.AccessNode(out_storage_node.data,
                                        access = dtypes.AccessType.ReadOnly)
            graph.add_node(array_closest_ancestor)



        # array_closest_ancestor now points to the node we want to connect
        # to the map entry

        # first, inline fuse back our NSDFG
        from dace.transformation.interstate import InlineSDFG
        # debug
        if InlineSDFG.can_be_applied(graph, \
                                     {InlineSDFG._nested_sdfg: graph.nodes().index(nsdfg)}, \
                                     0, sdfg) is False:
            print("ERROR: This should not appear")
        inline_sdfg = InlineSDFG(sdfg.sdfg_list.index(sdfg),
                                 sdfg.nodes().index(graph),
                                 {InlineSDFG._nested_sdfg: graph.nodes().index(nsdfg)},
                                 0)
        inline_sdfg.apply(sdfg)


        if not shortcut:
            # TODO: also for other types of reductions
            deduction_type = detect_reduction_type(wcr)
            if deduction_type in ReduceMap.reduction_type_update:
                code = ReduceMap.reduction_type_update[deduction_type]
            else:
                raise RuntimeError("Not yet implemented for custom reduction")


            new_tasklet = graph.add_tasklet(name = "reduction_transient_update",
                                            inputs = {"reduction_in", "array_in"},
                                            outputs = {"out"},
                                            code = code)

            edge_to_remove = graph.out_edges(out_transient_node_inner)[0]

            new_memlet_array_inner =    Memlet(data = out_storage_node.data,
                                            num_accesses = 1,
                                            subset = edge_to_remove.data.subset,
                                            vector_length = 1)
            new_memlet_array_outer =    Memlet(data = array_closest_ancestor.data,
                                            num_accesses = graph.in_edges(outer_entry)[0].data.num_accesses,
                                            subset = subsets.Range.from_array(sdfg.data(out_storage_node.data)),
                                            vector_length = 1)

            new_memlet_reduction =      Memlet(data = graph.out_edges(inner_exit)[0].data.data,
                                            num_accesses = 1,
                                            subset = graph.out_edges(inner_exit)[0].data.subset,
                                            vector_length = 1)
            new_memlet_out_inner =      Memlet(data = edge_to_remove.data.data,
                                            num_accesses = 1,
                                            subset = edge_to_remove.data.subset,
                                            vector_length = 1)
            new_memlet_out_outer =      dcpy(new_memlet_array_outer)

            # remove old edges

            outer_edge_to_remove = None
            for edge in graph.out_edges(outer_exit):
                if edge.src == edge_to_remove.dst:
                    outer_edge_to_remove = edge

            # debug
            if outer_edge_to_remove is None:
                print("ERROR: No outer_edge_to_remove found")

            graph.remove_edge_and_connectors(edge_to_remove)
            graph.remove_edge_and_connectors(outer_edge_to_remove)


            graph.add_edge(out_transient_node_inner,
                           None,
                           new_tasklet,
                           "reduction_in",
                           new_memlet_reduction)

            graph.add_edge(outer_entry,
                           None,
                           new_tasklet,
                           "array_in",
                           new_memlet_array_inner)
            graph.add_edge(array_closest_ancestor,
                           None,
                           outer_entry,
                           None,
                           new_memlet_array_outer)
            graph.add_edge(new_tasklet,
                           "out",
                           outer_exit,
                           None,
                           new_memlet_out_inner)
            graph.add_edge(outer_exit,
                           None,
                           out_storage_node,
                           None,
                           new_memlet_out_outer)

            # fill map scope connectors
            graph.fill_scope_connectors()
            graph._clear_scopedict_cache()
            # wcr is already removed

        # FORNOW: choose default schedule and implementation
        new_schedule = dtypes.ScheduleType.Default
        new_implementation = implementation if implementation else self.reduce_implementation
        new_axes = reduce_node.axes

        reduce_node_new = graph.add_reduce(wcr = wcr,
                                           axes = new_axes,
                                           schedule = new_schedule,
                                           identity = identity)
        reduce_node_new.implementation = new_implementation


        edge_tmp = graph.in_edges(inner_entry)[0]
        memlet_src_reduce = dcpy(edge_tmp.data)
        graph.add_edge(edge_tmp.src, edge_tmp.src_conn, reduce_node_new, None, memlet_src_reduce)

        edge_tmp = graph.out_edges(inner_exit)[0]
        memlet_reduce_dst = Memlet(data = edge_tmp.data.data,
                                   num_accesses = 1,
                                   subset = edge_tmp.data.subset,
                                   vector_length = edge_tmp.data.veclen)

        graph.add_edge(reduce_node_new, None, edge_tmp.dst, edge_tmp.dst_conn, memlet_reduce_dst)

        identity_tasklet = graph.out_edges(inner_entry)[0].dst
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)
        graph.remove_node(identity_tasklet)

        sdfg.validate()

        # setzero stuff

        if identity is None:
            # set transient_inner to set_zero = True
            # TODO: create identities for other reductions
            out_transient_node_inner.setzero = True

        # create variables for outside access
        self._new_reduce = reduce_node_new
        self._outer_entry = outer_entry
        return


    def _expand_reduce(self, sdfg, state, node):

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

        # If identity is defined, add an initialization state
        if node.identity is not None:
            print("ERROR: Identity must be overridden to None first")
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
                'reduce_output', {
                    '_o%d' % i: '0:%s' % symstr(sz)
                    for i, sz in enumerate(outedge.data.subset.size())
                })
            outm = Memlet.simple(
                '_out',
                ','.join(['_o%d' % i for i in range(output_dims)]),
                wcr_str=node.wcr)
            inmm = Memlet.simple('_in', ','.join(input_subset))
        else:
            ome, omx = None, None
            outm = Memlet.simple('_out', '0', wcr_str=node.wcr)
            inmm = Memlet.simple(
                '_in', ','.join(['_i%d' % i for i in range(len(axes))]))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        ime, imx = nstate.add_map(
            'reduce_values', {
                '_i%d' % i: '0:%s' % symstr(inedge.data.subset.size()[axis])
                for i, axis in enumerate(sorted(axes))
            })

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

        if node.schedule != dtypes.ScheduleType.Default:
            topnodes = nstate.scope_dict(node_to_children = True)[None]
            for topnode in topnodes:
                if isinstance(topnode, (nodes.EntryNode, nodes.LibraryNode)):
                    topnode.schedule = node.schedule


        nsdfg = state.add_nested_sdfg(nsdfg,
                                      sdfg,
                                      node.in_connectors,
                                      node.out_connectors,
                                      name = node.name)

        utils.change_edge_dest(state,node,nsdfg)
        utils.change_edge_src(state,node,nsdfg)
        state.remove_node(node)

        return nsdfg
