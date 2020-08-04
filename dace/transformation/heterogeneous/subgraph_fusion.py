""" This module contains classes that implement subgraph fusion
"""
import dace

from dace import dtypes, registry, symbolic, subsets, data
from dace.sdfg import nodes, utils, replace, SDFG, scope_contains_scope
from dace.sdfg.graph import SubgraphView
from dace.memlet import Memlet
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlet
from dace.transformation.heterogeneous import helpers


from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib

from collections import defaultdict
from itertools import chain

'''
TODO:

- revamp                                [OK]
- other_subset                          [OK]
- can_be_applied()                      [OK]
- StorageType Inference                 [OK]   cancelled
- cover intermediate nodes with
  incoming edges from outside           (*) (TODO)
- counter fix                           [OK]
- subset in_dict fix                    [OK]
- maybe: one intermediate data
         counter with out-conn:
         -> direct connection           (**) (TODO)
- rewrite tests                         TODO (pull branch)
- maybe add more tests                  TODO (pull branch)
- stencils                              next

'''

@make_properties
class SubgraphFusion(pattern_matching.SubgraphTransformation):
    """ Implements the SubgraphFusion transformation.
        Fuses the maps specified together with their outer maps
        as a global outer map, creating transients and new connections
        where necessary.
        Use MultiExpansion first before fusing a graph with SubgraphFusion
        Applicability checks have not been implemented yet.

        This is currently not implemented as a transformation template,
        as we want to input an arbitrary subgraph / collection of maps.

    """

    debug = Property(desc = "Show debug info",
                     dtype = bool,
                     default = True)

    cpu_transient_allocation = Property(desc = "Storage Location to push"
                                               "transients to in CPU environment",
                                        dtype = str,
                                        default = "default",
                                        choices = ["auto", "register", "heap", "threadlocal", "default"])

    cuda_transient_allocation = Property(desc = "Storage Location to push"
                                                "transients to in GPU environment",
                                         dtype = str,
                                         default = "local",
                                         choices = ["auto", "shared", "local", "default"])
    @staticmethod
    def match(sdfg, subgraph):

        # Fusable if
        # 1. Maps have the same access sets and ranges in order
        # 2. Any nodes in between two maps are AccessNodes only, without WCR
        #    There is at most one AccessNode only on a path between two maps,
        #    no other nodes are allowed
        # 3. The exiting memlets' subsets to an intermediate edge must cover
        #    the respective incoming memlets' subset into the next map

        # get graph
        graph = subgraph.graph
        for node in subgraph.nodes():
            if node not in graph.nodes():
                return False

        # next, get all the maps
        map_entries = helpers.get_lowest_scope_maps(sdfg, graph, subgraph)
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]
        maps = [map_entry.map for map_entry in map_entries]

        # 1. check whether all map ranges and indices are the same
        if len(maps) <= 1:
            return False
        base_map = maps[0]
        for map in maps:
            if map.get_param_num() != base_map.get_param_num():
                return False
            if not all([p1 == p2 for (p1,p2) in zip(map.params, base_map.params)]):
                return False
            if not map.range == base_map.range:
                return False

        # 1.1 check whether all map entries have the same schedule
        schedule = map_entries[0].schedule
        if not all([entry.schedule == schedule for entry in map_entries]):
            return False


        # 2. check intermediate feasiblility
        # see map_fusion.py for similar checks
        # we are being more relaxed here

        # 2.1 do some preparation work first:
        # calculate all out_nodes and intermediate_nodes
        # definition see in apply()
        intermediate_nodes = set()
        out_nodes = set()
        for map_entry, map_exit in zip(map_entries, map_exits):
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

        # 2.2 topological feasibility:
        # For each intermediate and out node: must never reach any map
        # entry if it is not connected to map entry immediately
        visited = set()
        # for memoization purposes
        def visit_descendants(graph, node, visited, map_entries):
            # if we have already been at this node
            if node in visited:
                return True
            # not necessary to add if there aren't any other in connections
            if len(graph.in_edges(node)) > 1:
                visited.add(node)
            for oedge in graph.out_edges(node):
                if not visit_descendants(graph, oedge.dst, visited, map_entries):
                    return False
            return True

        for node in intermediate_nodes | out_nodes:
            # these nodes must not lead to a map entry
            nodes_to_check = set()
            for oedge in graph.out_edges(node):
                if oedge.dst not in map_entries:
                    nodes_to_check.add(oedge.dst)

            for forbidden_node in nodes_to_check:
                if not visit_descendants(graph, forbidden_node, visited, map_entries):
                    return False

        del visited

        # 2.3 memlet feasibility
        # For each intermediate node, look at whether inner adjacent
        # memlets of the exiting map cover inner adjacent memlets
        # of the next entering map.
        # We also check for any WCRs on the fly.

        for node in intermediate_nodes:
            upper_subsets = set()
            lower_subsets = set()
            # First, determine which dimensions of the memlet ranges
            # change with the map, we do not need to care about the other dimensions.
            total_dims = len(sdfg.data(node.data).shape)
            dims_to_discard = SubgraphFusion.get_invariant_dimensions(sdfg, graph, map_entries, map_exits, node)

            # find upper_subsets
            for in_edge in graph.in_edges(node):
                # first check for WCRs
                if in_edge.data.wcr:
                    return False
                if in_edge.src in map_exits:
                    edge = graph.memlet_path(in_edge)[-2]
                    subset_to_add = dcpy(edge.data.subset\
                                         if edge.data.data == node.data\
                                         else edge.data.other_subset)
                    subset_to_add.pop(dims_to_discard)
                    upper_subsets.add(subset_to_add)
                else:
                    raise NotImplementedError("TODO (*)")

            # find lower_subsets
            for out_edge in graph.out_edges(node):
                if out_edge.dst in map_entries:
                    # cannot use memlet tree here as there could be
                    # not just one map succedding. Do it manually
                    for oedge in graph.out_edges(out_edge.dst):
                        if oedge.src_conn[3:] == out_edge.dst_conn[2:]:
                            subset_to_add = dcpy(oedge.data.subset \
                                                 if edge.data.data == node.data \
                                                 else edge.data.other_subset)
                            subset_to_add.pop(dims_to_discard)
                            lower_subsets.add(subset_to_add)


            #print("upper_subsets:", upper_subsets)
            #print("lower_subsets:", lower_subsets)

            upper_iter = iter(upper_subsets)
            union_upper = next(upper_iter)

            # TODO: add this check at a later point
            # We assume that upper_subsets for each data array
            # are contiguous
            # or do the full check if possible (intersection needed)
            '''
            # check whether subsets in upper_subsets are adjacent.
            # this is a requriement for the current implementation
            #try:
            # O(n^2*|dims|) but very small amount of subsets anyway
            try:
                for dim in range(total_dims - len(dims_to_discard)):
                    ordered_list = [(-1,-1,-1)]
                    for upper_subset in upper_subsets:
                        lo = upper_subset[dim][0]
                        hi = upper_subset[dim][1]
                        for idx,element in enumerate(ordered_list):
                            if element[0] <= lo and element[1] >= hi:
                                break
                            if element[0] > lo:
                                ordered_list.insert(idx, (lo,hi))
                    ordered_list.pop(0)


                    highest = ordered_list[0][1]
                    for i in range(len(ordered_list)):
                        if i < len(ordered_list)-1:
                            current_range = ordered_list[i]
                            if current_range[1] > highest:
                                hightest = current_range[1]
                            next_range = ordered_list[i+1]
                            if highest < next_range[0] - 1:
                                return False
            except TypeError:
                #return False
            '''

            # now take union of upper subsets
            for subs in upper_iter:
                union_upper = subsets.union(union_upper, subs)
                if not union_upper:
                    # something went wrong using union -- we'd rather abort
                    return False

            # finally check coverage
            for lower_subset in lower_subsets:
                if not union_upper.covers(lower_subset):
                    return False

        return True

    def storage_type_inference(self, node):
        ''' on a fused graph, looks in which memory intermediate node
            needs to be pushed to
        '''
        raise NotImplementedError("WIP")

    @staticmethod
    def get_invariant_dimensions(sdfg, graph, map_entries, map_exits, node):
        '''
        on a non-fused graph, return a set of
        indices that correspond to array dimensions that
        do not change when we are entering maps
        for an access node
        '''
        variate_dimensions = set()
        subset_length = -1

        for in_edge in graph.in_edges(node):
            if in_edge.src in map_exits:
                other_edge = graph.memlet_path(in_edge)[-2]
                other_subset = other_edge.data.subset \
                               if other_edge.data.data == node.data \
                               else other_edge.data.other_subset

                for (idx, (ssbs1, ssbs2)) \
                    in enumerate(zip(in_edge.data.subset, other_subset)):
                    if ssbs1 != ssbs2:
                        #print("(Up) Added to dims_to_inspect:")
                        #print(sbs1,"|", sbs2, ":", in_edge.src, "->", in_edge.dst)
                        variate_dimensions.add(idx)
            else:
                raise NotImplementedError("TODO(*)")

            if subset_length < 0:
                subset_length = other_subset.dims()
            else:
                assert other_subset.dims() == subset_length

        for out_edge in graph.out_edges(node):
            if out_edge.dst in map_entries:
                for other_edge in graph.out_edges(out_edge.dst):
                    if other_edge.src_conn[3:] == out_edge.dst_conn[2:]:
                        other_subset = other_edge.data.subset \
                                       if other_edge.data.data == node.data \
                                       else other_edge.data.other_subset
                        for (idx, (ssbs1, ssbs2)) in enumerate(zip(out_edge.data.subset, other_subset)):
                            if ssbs1 != ssbs2:
                                    #print("(Down) Added to dims_to_inspect:")
                                    #print(sbs1,"|", sbs2, ":", out_edge.src, "->", out_edge.dst)
                                    variate_dimensions.add(idx)
                        assert other_subset.dims() == subset_length

        invariant_dimensions = set([i for i in range(subset_length)]) - variate_dimensions
        return invariant_dimensions


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

    def prepare_intermediate_nodes(self, sdfg, graph, in_nodes, out_nodes,
                                    intermediate_nodes, map_entries, map_exits,
                                    do_not_override = []):

        def redirect(redirect_node, original_node):
            # redirect all outgoing traffic which
            # does not enter fusion scope again
            # from original_node to redirect_node
            # and then create a path from original_node to redirect_node.

            edges = list(graph.out_edges(original_node))
            for edge in edges:
                if edge.dst not in map_entries:
                    self.redirect_edge(graph, edge, new_src = redirect_node)

            graph.add_edge(original_node, None,
                           redirect_node, None,
                           Memlet())

        # first search whether intermediate_nodes appear outside of subgraph
        # and store it in dict
        data_counter = defaultdict(int)
        data_counter_subgraph = defaultdict(int)

        data_intermediate = set([node.data for node in intermediate_nodes])


        # do a full global search and count each data from each intermediate node
        scope_dict = graph.scope_dict()
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in data_intermediate:
                    # add them to the counter set in all cases
                    data_counter[node.data] += 1
                    # see whether we are inside the subgraph scope
                    # if so, add to data_counter_subgraph
                    # DO NOT add if it is in out_nodes
                    if state == graph and \
                       (node in intermediate_nodes or scope_dict[node] in map_entries):
                        data_counter_subgraph[node.data] += 1


        # next up: If intermediate_counter and global counter match and if the array
        # is declared transient, it is fully contained by the subgraph

        subgraph_contains_data = {data: data_counter[data] == data_counter_subgraph[data] \
                                        and sdfg.data(data).transient \
                                        and data not in do_not_override \
                                  for data in data_intermediate}

        transients_created = {}
        for node in intermediate_nodes & out_nodes:
            # create new transient at exit replacing the array
            # and redirect all traffic
            data_ref = sdfg.data(node.data)
            out_trans_data_name = node.data + '_OUT'
            data_trans = sdfg.add_transient(name = out_trans_data_name,
                                            shape = data_ref.shape,
                                            dtype = data_ref.dtype,
                                            storage= data_ref.storage,
                                            offset = data_ref.offset)
            node_trans = graph.add_access(out_trans_data_name)
            redirect(node_trans, node)
            transients_created[node] = node_trans

        # finally, create dict for every array that for which
        # subgraph_contains_data is true that lists invariant axes.
        invariant_dimensions = {}
        for node in intermediate_nodes:
            if subgraph_contains_data[node.data]:
                # only need to check in this case
                # else the array doesn't get modified and we don't
                # need invariate dimensions
                data = node.data
                inv_dims = SubgraphFusion.get_invariant_dimensions(sdfg, graph, map_entries, map_exits, node)
                if node in invariant_dimensions:
                    # do a check -- we want the same result for each
                    # node containing the same data
                    if not inv_dims == invariant_dimensions[node]:
                        print(f"WARNING: Invariant dimensions differ for {node.data}")
                        invariant_dimensions[data] |= inv_dims

                else:
                    invariant_dimensions[data] = inv_dims


        return (subgraph_contains_data, transients_created, invariant_dimensions)


    def apply(self, sdfg, subgraph, **kwargs):
        self.subgraph = subgraph

        graph = subgraph.graph

        map_entries = helpers.get_lowest_scope_maps(sdfg, graph, subgraph)
        self.fuse(sdfg, graph, map_entries, **kwargs)

    def fuse(self, sdfg, graph, map_entries, **kwargs):
        """ takes the map_entries specified and tries to fuse maps.

            all maps have to be extended into outer and inner map
            (use MapExpansion as a pre-pass)

            Arrays that don't exist outside the subgraph get pushed
            into the map and their data dimension gets cropped.
            Otherwise the original array is taken.

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

        # get maps and map exits
        maps = [map_entry.map for map_entry in map_entries]
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]

        # re-construct the map subgraph if necessary
        try:
            self.subgraph
        except AttributeError:
            subgraph_nodes = set()
            scope_dict = graph.scope_dict(node_to_children = True)
            for node in chain(map_entries, map_exits):
                subgraph_nodes.add(node)
                # add all border arrays
                for e in chain(graph.in_edges(node), graph.out_edges(node)):
                    subgraph_nodes.add(e.src)
                    subgraph_nodes.add(e.dst)
                try:
                    subgraph_nodes |= set(scope_dict[node])
                except KeyError:
                    pass
            self.subgraph = SubgraphView(graph, subgraph_nodes)


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
                            # add to intermediate_nodes
                            intermediate_nodes.add(current_node)

                        else:
                            # add to out_nodes
                            out_nodes.add(current_node)
                for e in graph.in_edges(current_node):
                    if e.src not in map_exits:
                        raise NotImplementedError("TODO: Not implemented yet (*)")

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

        ### assign correct schedule to global_map_entry
        schedule = map_entries[0].schedule
        if not all([entry.schedule == schedule for entry in map_entries]):
            raise RuntimeError("Not all the maps have the same schedule. Cannot fuse.")

        global_map_entry.schedule = schedule
        graph.add_node(global_map_entry)
        graph.add_node(global_map_exit)


        try:
            # in-between transients whose data should not be modified
            do_not_override = kwargs['do_not_override']
        except KeyError:
            do_not_override = []

        ### next up, for any intermediate node, find whether it only appears
        ### in the subgraph or also somewhere else / as an input
        ### create new transients for nodes that are in out_nodes and
        ### intermediate_nodes simultaneously
        ### also check which dimensions of each transient data element correspond
        ### to map axes and write this information into a dict.
        node_info = self.prepare_intermediate_nodes(sdfg, graph, in_nodes, out_nodes, \
                                                    intermediate_nodes,\
                                                    map_entries, map_exits, \
                                                    do_not_override)

        (subgraph_contains_data, transients_created, invariant_dimensions) = node_info
        if self.debug:
            print("SubgraphFusion::Intermediate_nodes: subgraph_contains_data")
            print(subgraph_contains_data)

        inconnectors_dict = {}
        # Dict for saving incoming nodes and their assigned connectors
        # Format: {access_node: (edge, in_conn, out_conn)}

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
                        # no need to augment subset of outer edge.
                        # will do this at the end in one pass.

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
                        #edge.dst = global_map_entry, edge.dst_conn = in_conn

                    # map out edges to new map
                    for out_edge in out_edges:
                        self.redirect_edge(graph, out_edge, new_src = global_map_entry, \
                                                            new_src_conn = out_conn)
                        #out_edge.src = global_map_entry
                        #out_edge.src_conn = out_conn

                else:
                    # connect directly
                    for out_edge in out_edges:
                        mm = dcpy(out_edge.data)
                        self.redirect_edge(graph, out_edge, new_src = src, new_data = mm)

                    graph.remove_edge(edge)


            for edge in graph.out_edges(map_entry):
                # special case: for nodes that have no data connections
                if not edge.src_conn:
                    self.redirect_edge(graph, edge, new_src = global_map_entry)

            ######################################

            for edge in graph.in_edges(map_exit):
                if not edge.dst_conn:
                    # no destination connector, path ends here.
                    self.redirect_edge(graph, edge, new_dst = global_map_exit)
                    continue
                # find corresponding out_edges for current edge, cannot use mmt anymore
                out_edges = [oedge for oedge in graph.out_edges(map_exit)
                                      if oedge.src_conn[3:] == edge.dst_conn[2:]]

                # Tuple to store in/out connector port that might be created
                port_created = None

                for out_edge in out_edges:
                    dst = out_edge.dst

                    if dst in intermediate_nodes & out_nodes:
                        # create connection thru global map from
                        # dst to dst_transient that was created
                        dst_transient = transients_created[dst]
                        next_conn = global_map_exit.next_connector()
                        in_conn= 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map_exit.add_in_connector(in_conn)
                        global_map_exit.add_out_connector(out_conn)

                        e = graph.add_edge(dst, None,
                                           global_map_exit, in_conn,
                                           dcpy(edge.data))

                        e = graph.add_edge(global_map_exit, out_conn,
                                           dst_transient, None,
                                           dcpy(out_edge.data))

                        # remove edge from dst to dst_transient that was created
                        # in intermediate preparation.
                        for e in graph.out_edges(dst):
                            if e.dst == dst_transient:
                                graph.remove_edge(e)
                                removed = True
                                break

                        if self.debug:
                            assert removed == True

                    # handle separately: intermediate_nodes and pure out nodes
                    # case 1: intermediate_nodes: can just redirect edge
                    if dst in intermediate_nodes:
                        self.redirect_edge(graph, out_edge, new_src = edge.src,
                                                            new_src_conn = edge.src_conn,
                                                            new_data = dcpy(edge.data))

                    # case 2: pure out node: connect to outer array node
                    if dst in (out_nodes - intermediate_nodes):
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

            # maps are now ready to be discarded
            graph.remove_node(map_entry)
            graph.remove_node(map_exit)

            # end main loop.




        # TODO: do one pass for special case (*)

        # create a mapping from data arrays to offsets
        # for later memlet adjustments later
        min_offsets = dict()


        ### do one pass to augment all transient arrays
        data_intermediate = set([node.data for node in intermediate_nodes])
        for data_name in data_intermediate:
            if subgraph_contains_data[data_name]:
                all_nodes = [n for n in self.subgraph.nodes() if isinstance(n, nodes.AccessNode) and n.data == data_name]
                in_edges = list(chain(*(graph.in_edges(n) for n in all_nodes)))

                in_edges_iter = iter(in_edges)
                in_edge = next(in_edges_iter)
                target_subset = dcpy(in_edge.data.subset)
                target_subset.pop(invariant_dimensions[data_name])
                ######
                while True:
                    try: # executed if there are multiple in_edges
                        in_edge = next(in_edges_iter)
                        target_subset_curr = dcpy(in_edge.data.subset)
                        target_subset_curr.pop(invariant_dimensions[data_name])
                        target_subset = subsets.union(target_subset, \
                                                      target_subset_curr)
                    except StopIteration:
                        break

                min_offsets_cropped = target_subset.min_element()
                # calculate the new transient array size.
                target_subset.offset(min_offsets_cropped, True)

                # re-add invariant dimensions with offset 0 and save to min_offsets
                min_offset = []
                index = 0
                for i in range(len(sdfg.data(data_name).shape)):
                    if i in invariant_dimensions[data_name]:
                        min_offset.append(0)
                    else:
                        min_offset.append(min_offsets_cropped[index])
                        index += 1

                min_offsets[data_name] = min_offset

                # determine the shape of the new array.
                new_data_shape = []
                index = 0
                for i, sz in enumerate(sdfg.data(data_name).shape):
                    if i in invariant_dimensions[data_name]:
                        new_data_shape.append(sz)
                    else:
                        new_data_shape.append(target_subset.size()[index])
                        index += 1

                new_data_strides = [data._prod(new_data_shape[i+1:])
                                    for i in range(len(new_data_shape))]

                new_data_totalsize = data._prod(new_data_shape)
                new_data_offset = [0]*len(new_data_shape)
                # augment.
                transient_to_transform = sdfg.data(data_name)
                transient_to_transform.shape   = new_data_shape
                transient_to_transform.strides = new_data_strides
                transient_to_transform.total_size = new_data_totalsize
                transient_to_transform.offset  = new_data_offset

                if schedule == dtypes.ScheduleType.GPU_Device:
                    print("Global Schedule: GPU")
                    if self.cuda_transient_allocation == 'local':
                        transient_to_transform.storage = dtypes.StorageType.Register
                    if self.cuda_transient_allocation == 'shared':
                        transient_to_transform.storage = dtypes.StorageType.GPU_Shared
                    if self.cuda_transient_allocation == 'default':
                        transient_to_transform.storage = dtypes.StorageType.Default
                    if self.cuda_transient_allocation == 'auto':
                        transient_to_transform.storage = self.storage_type_inference(node)
                else:
                    print("Global Schedule: CPU")
                    if self.cpu_transient_allocation == 'register':
                        transient_to_transform.storage = dtypes.StorageType.Register
                    if self.cpu_transient_allocation == 'threadlocal':
                        transient_to_transform.storage = dtypes.StorageType.CPU_ThreadLocal
                    if self.cpu_transient_allocation == 'heap':
                        transient_to_transform.storage = dtypes.StorageType.CPU_Heap
                    if self.cpu_transient_allocation == 'default':
                        transient_to_transform.storage = dtypes.StorageType.Default
                    if self.cpu_transient_allocation == 'auto':
                        transient_to_transform.storage = self.storage_type_inference(node)


            else:
                # don't modify data container - array is needed outside
                # of subgraph.

                # hack: set lifetime to State if allocation has only been
                # scope so far to avoid scopoing issues (Tal)
                if sdfg.data(data_name).lifetime == dtypes.AllocationLifetime.Scope:
                    sdfg.data(data_name).lifetime = dtypes.AllocationLifetime.State


        ### do one pass to adjust and the memlets of in-between transients
        for node in intermediate_nodes:
            # all incoming edges to node
            in_edges = graph.in_edges(node)
            # outgoing edges going to another fused part
            inter_edges = []
            # outgoing edges that exit global map
            out_edges = []
            for e in graph.out_edges(node):
                if e.dst == global_map_exit:
                    out_edges.append(e)
                else:
                    inter_edges.append(e)

            # offset memlets where necessary
            if subgraph_contains_data[node.data]:
                # get min_offset
                min_offset = min_offsets[node.data]
                # re-add invariant dimensions with offset 0
                for iedge in in_edges:
                    for edge in graph.memlet_tree(iedge):
                        if edge.data.data == node.data:
                            edge.data.subset.offset(min_offset, True)
                        elif edge.data.other_subset:
                            edge.data.other_subset.offset(min_offset, True)

                for cedge in inter_edges:
                    for edge in graph.memlet_tree(cedge):
                        if edge.data.data == node.data:
                            edge.data.subset.offset(min_offset, True)
                        elif edge.data.other_subset:
                            edge.data.other_subset.offset(min_offset, True)

                # if in_edges has several entries:
                # put other_subset into out_edges for correctness
                if len(in_edges) > 1:
                    for oedge in out_edges:
                        oedge.data.other_subset = dcpy(oedge.data.subset)
                        oedge.data.other_subset.offset(min_offset, True)


            # also correct memlets of created transient
            if node in transients_created:
                transient_in_edges = graph.in_edges(transients_created[node])
                transient_out_edges = graph.out_edges(transients_created[node])
                for edge in chain(transient_in_edges, transient_out_edges):
                    for e in graph.memlet_tree(edge):
                        if e.data.data == node.data:
                            e.data.data += '_OUT'


        ### do one last pass to correct outside memlets adjacent to global map
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
            in_edge.data.volume = memlet_out.volume
            in_edge.data.subset = memlet_out.subset

        ### create a hook for outside access to global_map
        self._global_map_entry = global_map_entry
