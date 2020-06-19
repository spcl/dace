""" This module contains classes that implement subgraph fusion
"""
import dace

from dace import dtypes, registry, symbolic, subsets, data
from dace.sdfg import nodes, utils
from dace.memlet import Memlet, EmptyMemlet
from dace.sdfg import replace, SDFG
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlet

from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib




@make_properties
class SubgraphFusion():
    """ Implements the SubgraphFusion transformation.
        Fuses the maps specified together with their outer maps
        as a global outer map, creating transients and new connections
        where necessary.
        Use MultiExpansion first before fusing a graph with SubgraphFusion
        Applicability checks have not been implemented yet.

        This is currently not implemented as a transformation template,
        as we want to input an arbitrary subgraph / collection of maps.

    """

    register_trans = Property(desc="Make connecting transients with size 1 inside"
                                    "the global map registers",
                              dtype = bool,
                              default = True)

    debug = Property(desc = "Show debug info",
                     dtype = bool,
                     default = True)

    def __init__(self, debug = None, register_trans = None):
        if debug is not None:
            self.debug = True
        if register_trans is not None:
            self.register_trans = True


    @staticmethod
    def can_be_applied(sdfg, graph, maps):

        # TODO: Not really a priority right now

        # Fusable if
        # 1. Maps have the same access sets and ranges in order
        # 2. Any nodes in between two maps are AccessNodes only without WCR
        #    There is at most one AccessNode only on a path between two maps,
        #    no other nodes are allowed
        # 3. Any exiting memlet's subset to an intermediate edge must cover
        #    the respective incoming memlets subset into the next map

        # 4  Every array that is in between two maps can only appear once
        #    in a write node within the maps subgraph

        return True

    def redirect_edge(self, graph, edge, new_src = None, new_src_conn = None ,
                                         new_dst = None, new_dst_conn = None, new_data = None ):
        if not(new_src or new_dst) or new_src and new_dst:
            raise RuntimeError("Redirect Edge has been used wrongly")
        data = new_data if new_data else edge.data
        if new_src:
            ret = graph.add_edge(new_src, new_src_conn, edge.dst, edge.dst_conn, data)
            graph.remove_edge(edge)
        if new_dst:
            ret = graph.add_edge(edge.src, edge.src_conn, new_dst, new_dst_conn, data)
            graph.remove_edge(edge)

        return ret


    def _create_transients(self,sdfg, graph, in_nodes, out_nodes, intermediate_nodes, \
                           map_entries, map_exits, do_not_override = []):
        # better preprocessing which allows for non-transient in between nodes
        ''' creates transients for every in-between node that has exit connections or that is non-transient (or both)
            the resulting transient can then later be pushed into the map
        '''

        def redirect(redirect_node, original_node):
            # redirect all traffic to original node to redirect node
            # and then create a path from redirect to original
            # outgoing edges to other maps in our subset should be originated from the clone
            # similar to utils.change_edge_dest(graph, original_node, redirect_node)
            # but only when edge comes from a map exit
            edges = list(graph.in_edges(original_node))
            for e in edges:
                if e.src in map_exits:
                    graph.remove_edge(e)
                    if isinstance(e, dace.sdfg.graph.MultiConnectorEdge):
                        graph.add_edge(e.src, e.src_conn, redirect_node, e.dst_conn, e.data)
                    else:
                        graph.add_edge(e.src, redirect_node, e.data)


            graph.add_edge(redirect_node,
                           None,
                           original_node,
                           None,
                           EmptyMemlet())
            for edge in graph.out_edges(original_node):
                if edge.dst in map_entries:
                    #edge.src = redirect_node
                    self.redirect_edge(graph, edge, new_src = redirect_node)


        transient_dict = {}
        for node in (intermediate_nodes):
            if node in out_nodes \
               or node in do_not_override \
               or not sdfg.data(node.data).transient:

                data_ref = sdfg.data(node.data)
                trans_data_name = node.data + '__trans'

                data_trans = sdfg.add_transient(name=trans_data_name,
                                                shape= data_ref.shape,
                                                dtype= data_ref.dtype,
                                                storage= data_ref.storage,
                                                offset= data_ref.offset)
                node_trans = graph.add_access(trans_data_name)
                redirect(node_trans, node)
                transient_dict[node_trans] = node

        return transient_dict


    def _create_transients_TRANS_ONLY(self, sdfg, graph, in_nodes, out_nodes, intermediate_nodes, \
                                      map_entries, map_exits, do_not_override = []):
        # old version that does not work with non-transients

        def redirect(redirect_node, original_node):
            # redirect all traffic to original node to redirect node
            # and then create a path from redirect to original
            utils.change_edge_dest(graph, original_node, redirect_node)
            graph.add_edge(redirect_node,
                           None,
                           original_node,
                           None,
                           EmptyMemlet())
            for edge in graph.out_edges(original_node):
                if edge.dst in map_entries:
                    #edge.src = redirect_node
                    self.redirect_edge(graph, edge, new_src = redirect_node)


        transient_dict = {}
        for node in (intermediate_nodes & out_nodes):
            data_ref = sdfg.data(node.data)
            trans_data_name = node.data + '__trans'

            data_trans = sdfg.add_transient(name=trans_data_name,
                                            shape= data_ref.shape,
                                            dtype= data_ref.dtype,
                                            storage= data_ref.storage,
                                            offset= data_ref.offset)
            node_trans = graph.add_access(trans_data_name)
            redirect(node_trans, node)
            transient_dict[node_trans] = node

        return transient_dict


    def fuse(self, sdfg, graph, map_entries, **kwargs):
        """ takes the map_entries specified and tries to fuse maps.

            all maps have to be extended into outer and inner map
            (use MapExpansion as a pre-pass)

            Extra transients get created for non-transient arrays that
            lie in between two maps in order to get pushed into the map.
            (and also other arrays to be specified in do_not_override)

            For every output respective connections are crated automatically.

            See can_be_applied for requirements.

            [Work in Progress] Features and corner cases not supported yet:
            - border memlets with subset changes (memlet.other_subset)
              are not supported currently
            - Transients that get pushed into the global map
              always persist, even if they have size one (unlike MapFusion)

            :param sdfg: SDFG
            :param graph: State
            :param map_entries: Map Entries (class MapEntry) of the outer maps
                                which we want to fuse
            :param do_not_override: List of AccessNodes that are transient but
                                  should not be directly modified when pushed
                                  into the global map. Instead a transient copy
                                  is created and linked to it.
        """

        # if there are no maps, return immediately
        if len(map_entries) == 0:
            return

        maps = [map_entry.map for map_entry in map_entries]
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]

        # Nodes that flow into one or several maps but no data is flowed to them from any map
        in_nodes = set()

        # Nodes into which data is flowed but that no data flows into any map from them
        out_nodes = set()

        # Nodes that act as intermediate node - data flows from a map into them and then there
        # is an outgoing path into another map
        intermediate_nodes = set()

        """ NOTE:
        - in_nodes, out_nodes, intermediate_nodes refer to the configuration of the final fused map
        - in_nodes and out_nodes are trivially disjoint
        - Intermediate_nodes and out_nodes are not necessarily disjoint
        - Intermediate_nodes and in_nodes are disjoint by design.
          There could be a node that has both incoming edges from a map exit
          and from outside, but it is just treated as intermediate_node and handled
          automatically.
        """

        for map_entry, map_exit in zip(map_entries, map_exits):
            for edge in graph.in_edges(map_entry):
                in_nodes.add(edge.src)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if len(graph.out_edges(current_node)) == 0:
                    out_nodes.add(current_node)
                else:
                    for dst_edge in graph.out_edges(current_node):
                        if dst_edge.dst in map_entries:
                            intermediate_nodes.add(current_node)
                        else:
                            out_nodes.add(current_node)

        # any intermediate_nodes currently in in_nodes shouldnt be there
        in_nodes -= intermediate_nodes

        if self.debug:
            print("SubgraphFusion::In_nodes", in_nodes)
            print("SubgraphFusion::Out_nodes", out_nodes)
            print("SubgraphFusion::Intermediate_nodes", intermediate_nodes)

        # all maps are assumed to have the same params and range in order
        global_map = nodes.Map(label = "outer_fused",
                               params = maps[0].params,
                               ndrange = maps[0].range)
        global_map_entry = nodes.MapEntry(global_map)
        global_map_exit  = nodes.MapExit(global_map)

        # assign correct schedule to global_map_entry
        schedule = map_entries[0].schedule
        if not all([entry.schedule == schedule for entry in map_entries]):
            raise RuntimeError("Not all the maps have the same schedule. Cannot fuse.")

        global_map_entry.schedule = schedule

        # if we make new transients of any objects, we are to save them
        # into this dict. This allows for easy redirection

        try:
            # in-between transients that should be duplicated nevertheless
            do_not_override = kwargs['do_not_override']
        except KeyError:
            do_not_override = []

        transient_dict = self._create_transients(sdfg, graph, in_nodes, out_nodes, intermediate_nodes, map_entries, map_exits, do_not_override)
        inconnectors_dict = {}
        # {access_node: (edge, in_conn, out_conn)}

        graph.add_node(global_map_entry)
        graph.add_node(global_map_exit)

        for map_entry, map_exit in zip(map_entries, map_exits):
            # handle inputs
            # TODO: dynamic map range -- this is fairly unrealistic in such a setting
            for edge in graph.in_edges(map_entry):
                src = edge.src
                mmt = graph.memlet_tree(edge)
                out_edges = [child.edge for child in mmt.root().children]

                if src in in_nodes:

                    in_conn = None; out_conn = None

                    if src in inconnectors_dict:
                        if not inconnectors_dict[src][0].data.subset.covers(edge.data.subset):
                            if debug:
                                print("SubgraphFusion::Extend range")
                            inconnectors_dict[edge.data.data][0].subset = edge.data.subset

                        in_conn = inconnectors_dict[src][1]
                        out_conn = inconnectors_dict[src][2]
                        graph.remove_edge(edge)

                    else:
                        next_conn = global_map_entry.next_connector()
                        in_conn = 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map_entry.add_in_connector(in_conn)
                        global_map_entry.add_out_connector(out_conn)

                        inconnectors_dict[src] = (edge, in_conn, out_conn)

                        # reroute in edge via global_map_entry
                        self.redirect_edge(graph, edge, new_dst = global_map_entry, \
                                                        new_dst_conn = in_conn)
                        #edge.dst = global_map_entry
                        #edge.dst_conn = in_conn

                    # map out edges to new map
                    for out_edge in out_edges:
                        self.redirect_edge(graph, out_edge, new_src = global_map_entry, \
                                                            new_src_conn = out_conn)
                        #out_edge.src = global_map_entry
                        #out_edge.src_conn = out_conn

                else:

                    # connect directly
                    # also make sure memlet data gets changed correctly
                    old_name = edge.data.data
                    if src in transient_dict:
                        new_name = old_name + '__trans'
                    else:
                        new_name = old_name

                    queue = []
                    for out_edge in out_edges:
                        mm = Memlet(data = new_name,
                                    num_accesses = out_edge.data.num_accesses,
                                    subset = out_edge.data.subset,
                                    vector_length = out_edge.data.veclen,
                                    other_subset = out_edge.data.other_subset
                                    )

                        self.redirect_edge(graph, out_edge, new_src = src, new_data = mm)
                        queue.append(out_edge.dst)
                        #out_edge.src = src

                    graph.remove_edge(edge)


                    while len(queue) > 0:
                        current = queue.pop(0)
                        if isinstance(current, nodes.MapEntry):
                            for oedge in graph.out_edges(current):
                                if oedge.data.data == old_name:
                                    oedge.data.data = new_name
                                    queue.append(oedge.dst)
            for edge in graph.out_edges(map_entry):
                # special case: for nodes that have no data connections
                if not edge.src_conn:
                    self.redirect_edge(graph, edge, new_src = global_map_entry)
            for edge in graph.in_edges(map_exit):
                if not edge.dst_conn:
                    # no destination connector, path ends here.
                    self.redirect_edge(graph, edge, new_dst = global_map_exit)
                    continue
                # find corresponding out_edges for current edge, cannot use mmt anymore
                out_edges = [oedge for oedge in graph.out_edges(map_exit)
                                      if oedge.src_conn[3:] == edge.dst_conn[2:]]
                port_created = None

                for out_edge in out_edges:
                    dst = out_edge.dst
                    transient_created = None
                    try:
                        dst_original = transient_dict[dst]
                        transient_created = True
                    except KeyError:
                        transient_created = False
                        dst_original = dst


                    if transient_created:
                        # transients only get created for itntermediate_nodes
                        # that are either in out_nodes or non-transient
                        next_conn = global_map_exit.next_connector()
                        in_conn= 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map_exit.add_in_connector(in_conn)
                        global_map_exit.add_out_connector(out_conn)

                        graph.add_edge(dst, None,
                                       global_map_exit, in_conn,
                                       dcpy(edge.data))
                        graph.add_edge(global_map_exit, out_conn,
                                       dst_original, None,
                                       dcpy(out_edge.data))

                        edge_to_remove = None
                        for e in graph.out_edges(dst):
                            if e.dst == dst_original:
                                edge_to_remove = e
                                break
                        if edge_to_remove:
                            graph.remove_edge(edge_to_remove)
                        else:
                            print("ERROR: Transient support edge should be removable, not found")


                    # handle separately: intermediate_nodes and pure out nodes
                    if dst_original in intermediate_nodes:

                        # change upwards adjacent memlet data name to this data name
                        # downwards was handled before in in_edges
                        new_name = dst.data
                        old_name = dst_original.data

                        mm = Memlet(data = new_name,
                                    num_accesses = edge.data.num_accesses,
                                    subset = edge.data.subset,
                                    vector_length = edge.data.veclen,
                                    other_subset = edge.data.other_subset)

                        self.redirect_edge(graph, out_edge, new_src = edge.src,
                                                            new_src_conn = edge.src_conn,
                                                            new_data = mm)


                        queue = [edge.src]
                        while len(queue) > 0:
                            current = queue.pop(0)
                            if isinstance(current, nodes.MapExit):
                                for iedge in graph.in_edges(current):
                                    if iedge.data.data == old_name:
                                        iedge.data.data = new_name
                                        queue.append(iedge.src)

                        # TODO
                        # remove data node if size is 1 and only 1 in_edge
                        # can just do a direct flow
                        #if new_data_totalsize == 1 and len(graph.in_edges(dst)) == 1:
                        #......

                    if dst_original in (out_nodes - intermediate_nodes):
                        if edge.dst != global_map_exit:
                            next_conn = global_map_exit.next_connector()
                            in_conn= 'IN_' + next_conn
                            out_conn = 'OUT_' + next_conn
                            global_map_exit.add_in_connector(in_conn)
                            global_map_exit.add_out_connector(out_conn)
                            self.redirect_edge(graph, edge, new_dst = global_map_exit,
                                                                   new_dst_conn = in_conn)
                            port_created = (in_conn, out_conn)
                            #edge.dst = global_map_exit
                            #edge.dst_conn = in_conn

                        else:
                            conn_nr = edge.dst_conn[3:]
                            in_conn = port_created.st
                            out_conn = port_created.nd


                        # map
                        graph.add_edge(global_map_exit,
                                       out_conn,
                                       dst,
                                       None,
                                       dcpy(out_edge.data))
                        graph.remove_edge(out_edge)

                # remove the edge if it has not been used by any pure out node
                if not port_created:
                    graph.remove_edge(edge)



            graph.remove_node(map_entry)
            graph.remove_node(map_exit)


        # do one pass to adjust in-transients and their corresponding memlets
        transient_dict_rev = {v:k for k,v in transient_dict.items()}
        for transient_node in intermediate_nodes:
            try:
                transient_node = transient_dict_rev[transient_node]
            except KeyError:
                pass

            '''
            if len(graph.in_edges(transient_node)) > 1:
                raise NotImplementedError("[WIP] Not implemented yet "
                                          "for transient node with multiple incoming connections")
            '''

            # for each dimension, determine base set
            in_edges = graph.in_edges(transient_node)
            cont_edges = []
            out_edges = []
            for e in graph.out_edges(transient_node):
                if e.dst == global_map_exit:
                    out_edges.append(e)
                else:
                    cont_edges.append(e)


            # general case: multiple in_edges per in-between transient
            # e.g different subsets written to it
            in_edges_iter = iter(in_edges)
            in_edge = next(in_edges_iter)
            target_subset = dcpy(in_edge.data.subset)
            base_offset = in_edge.data.subset.min_element()
            ###### Work in Progress
            while True: # entered if there are multiple in_edges
                try:
                    in_edge = next(in_edges_iter)
                    target_subset_curr = dcpy(in_edge.data.subset)
                    base_offset_curr = in_edge.data.subset.min_element()

                    target_subset = subsets.bounding_box_union(target_subset, \
                                                               target_subset_curr)
                    base_offset = [min(base_offset[i], base_offset_curr[i]) for i in range(len(base_offset))]
                except StopIteration:
                    break
            ######

            ### augment transients

            sizes = target_subset.bounding_box_size()
            new_data_shape = [sz for (sz, s) in zip(sizes, target_subset)]
            # in case it is just a scalar
            new_data_strides = [data._prod(new_data_shape[i+1:])
                                for i in range(len(new_data_shape))]

            new_data_totalsize = data._prod(new_data_shape)
            new_data_offset = [0]*len(new_data_shape)

            transient_to_transform = sdfg.data(transient_node.data)
            transient_to_transform.shape   = new_data_shape
            transient_to_transform.strides = new_data_strides
            transient_to_transform.total_size = new_data_totalsize
            transient_to_transform.offset  = new_data_offset
            transient_to_transform.lifetime = dtypes.AllocationLifetime.Scope

            if self.register_trans and new_data_totalsize == 1:
                transient_to_transform.storage = dtypes.StorageType.Register
            else:
                transient_to_transform.storage = dtypes.StorageType.Default

            ###


            # offset every memlet where necessary
            for iedge in in_edges:
                for edge in graph.memlet_tree(iedge):
                    edge.data.subset.offset(base_offset, True)

            for cedge in cont_edges:
                for edge in graph.memlet_tree(cedge):
                    edge.data.subset.offset(base_offset, True)

            # of more than one entry: other_subset
            if len(in_edges) > 1:
                for oedge in out_edges:
                    oedge.data.other_subset = dcpy(oedge.data.subset)
                    oedge.data.other_subset.offset(base_offset, True)


            # TODO: other_subset handling
            # TODO: Waiting for Tal's API
            #for edge in out_edges:
            #    edge.data.other_subset = dcpy(in_edge.data.subset)
            #    ...



        # do one last pass to correct outside memlets adjacent to global map
        for out_connector in global_map_entry.out_connectors:
            # find corresponding in_connector
            in_connector = 'IN' + out_connector[3:]
            for iedge in graph.in_edges(global_map_entry):
                if iedge.dst_conn == in_connector:
                    in_edge = iedge
            for oedge in graph.out_edges(global_map_entry):
                if oedge.src_conn == out_connector:
                    out_edge = oedge
            # do memlet propagation
            memlet_out = propagate_memlet(dfg_state = graph,
                                          memlet = out_edge.data,
                                          scope_node = global_map_entry,
                                          union_inner_edges = True)

            # override number of accesses
            in_edge.data.num_accesses = memlet_out.num_accesses

        # create a hook for outside the class to access global_map
        self._global_map_entry = global_map_entry
