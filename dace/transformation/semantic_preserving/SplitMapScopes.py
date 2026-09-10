"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the SplitMapScopes transformation.

"""

import copy
import dace

class SplitMapScopes():
    """
    This class implements the SplitMapScopes transformation. It searches for map scopes in the SDFG and splits them. The transformation is only applied if the map scope is splittable.        
    """

    def __init__(self):
        self.__name = 'SplitMapScopes'
        self.checked = False

    @property
    def name(self):
        return self.__name

    def isInnermost(self, state, map_entry, traversed=None):
        if traversed is None:
            traversed = set()
        for edge in state.out_edges(map_entry):
            if isinstance(edge.dst, dace.nodes.MapEntry):
                return False
            if isinstance(edge.dst, dace.nodes.MapExit):
                continue
            if not edge.dst in traversed:
                if not self.isInnermost(state, edge.dst, traversed):
                    return False
        return True

    def traverse(self, state, map_entry, tasklets=None, visited=None):
        if tasklets is None:
            tasklets = set()
        if visited is None:
            visited = set()

        visited.add(map_entry)
        next_edges = state.out_edges(map_entry)
        for edge in next_edges:
            if isinstance(edge.dst, dace.nodes.Tasklet) and not edge.dst in tasklets:
                tasklets.add((state, edge.dst))
            elif isinstance(edge.dst, dace.nodes.MapExit) or isinstance(edge.dst, dace.nodes.Tasklet):
                continue
            if not edge.dst in visited:
                self.traverse(state, edge.dst, tasklets, visited)
        
        return list(tasklets)

    def find_2(self, sdfg):
        splitable_targets = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if not self.isInnermost(state, node):
                        continue
                    splitable_targets.extend(self.traverse(state, node))
        result = []
        for state, target_tasklet in splitable_targets:
            for edge in state.in_edges(target_tasklet):
                if not isinstance(edge.src, dace.nodes.MapEntry):
                    result.append((state, target_tasklet))
                    break
        self.checked = True
        return result
    
    def find(self, sdfg):
        """
        This method searches for map scopes in the SDFG and returns a list of tuples containing the map scope and the tasklet to split.

        The map scope is splittable if the following conditions are met:
            - The map scope is the innermost map scope
            - The map scope contains more than one tasklet

        Args:
            sdfg (dace.SDFG): The SDFG to search for splittable map scopes.
        
        Returns:
            list: A list of tuples, where each tuple contains a the state and the tasklet to split.
        """

        splitable_targets = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if not self.isInnermost(state, node):
                        continue
                    splitable_targets.extend(self.traverse(state, node))
        
        result = []
    
        # for tigthening the application space
        for state, node in splitable_targets:
            scope_nodes = []
            reject = False
            self.find_scope_nodes(state, node, scope_nodes)
            for scope_node in scope_nodes[1:]:
                inputs = [edge.src for edge in state.in_edges(scope_node)]
                for ins in inputs:
                    if isinstance(ins, dace.nodes.MapEntry):
                        reject = True
                        break 
                    if ins not in scope_nodes:
                        reject = True 
                        break
                    
            if not reject:
                result.append(node)
        self.checked = True
        return result
                                            
    def apply(self, sdfg, state, target_tasklet):
        """
        This method applies the SplitMapScopes transformation to the given SDFG. It splits the given map scope at the given tasklet.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            state (dace.SDFGState): The state the map scope is in.
            target_tasklet (dace.nodes.Tasklet): The tasklet to split the map scope at.
        """
        if not self.checked and not (state, target_tasklet) in self.find(sdfg):
            return
        self.checked = False

        map_entry = state.entry_node(target_tasklet)
        map1_input1 = []
        map1_input2 = []
        map1_input1 = [edge for edge in state.in_edges(map_entry)]
        map1_input2 = [edge for edge in state.out_edges(map_entry)]

        #add a new map
        new_map_entry, new_map_exit = state.add_map(f'{map_entry.label}', {map_entry.map.params[0]: f'0:{map_entry.map.range.size()[0]}'})
        
        # reconnect old map entry edges to new one
        for edge in map1_input1:
            in_conn = edge.dst_conn
            if in_conn:
                map_entry.remove_in_connector(in_conn)
            new_map_entry.add_in_connector(in_conn)
            dace.transformation.helpers.redirect_edge(state, edge, new_dst = new_map_entry, new_dst_conn = in_conn)
        for edge in map1_input2:
            out_conn = edge.src_conn
            if out_conn:
                map_entry.remove_out_connector(out_conn)
            new_map_entry.add_out_connector(out_conn)
            dace.transformation.helpers.redirect_edge(state, edge, new_src = new_map_entry, new_src_conn = out_conn)
        
        # find the paths that have to be redirected because of the new map
        inbetween_nodes = []
        inbetween_nodes = list(set([edge.src for edge in state.in_edges(target_tasklet)]))
        path_nodes = []
        for node in inbetween_nodes:
            if isinstance(node, dace.nodes.AccessNode):
                in_edges = state.in_edges(node)
                out_edges = state.out_edges(node)
                for edge in in_edges:
                    for edge2 in out_edges:
                        if not edge2.dst == target_tasklet:
                            continue
                        path_nodes.append((edge, node, edge2))
            if isinstance(node, dace.nodes.MapEntry):
                in_edges = {edge:False for edge in state.in_edges(node)}
                out_edges = {edge:False for edge in state.out_edges(node)}
                for edge, used in in_edges.items():
                    if used:
                        continue
                    for edge2, used2 in out_edges.items():
                        if not edge2.dst == target_tasklet:
                            continue
                        if used2:
                            continue
                        if edge.dst_conn[3:] == edge2.src_conn[4:]:
                            in_edges[edge] = True
                            out_edges[edge2] = True
                            path_nodes.append((edge, node, edge2))
                            break

        # redirect the edges of the input nodes of the tasklet
        # make a copy if input is used somewhere else aswell
        already_mapped = set()
        keep = set()
        for (edge1, btw_node, edge2) in path_nodes:
            if isinstance(btw_node, dace.nodes.AccessNode):
                out_edges = {edge.dst for edge in state.out_edges(btw_node) if not isinstance(edge.dst, dace.nodes.MapEntry)}
                if len(out_edges) == 1:
                    memlet = edge1.data
                    memlet2 = copy.deepcopy(edge1.data)
                    src = edge1.src
                    dst = edge2.dst
                    if isinstance(src, dace.nodes.Tasklet) and not src in already_mapped:
                        already_mapped.add(src)
                        state.add_memlet_path(src, new_map_exit, btw_node, memlet=memlet, src_conn=edge1.src_conn, dst_conn=edge1.dst_conn)
                    state.add_memlet_path(btw_node, map_entry, dst, memlet=memlet2, dst_conn=edge2.dst_conn, src_conn=edge2.src_conn)
                else:
                    new_node = copy.deepcopy(btw_node)
                    memlet = copy.deepcopy(edge1.data)
                    keep.add(edge1)
                    state.add_memlet_path(new_node, map_entry, edge2.dst, memlet=memlet, dst_conn=edge2.dst_conn)
                    
                    #add in and out connector manually if not already doen by add_memlet_path
                    out_edge = state.out_edges(new_node)[0]
                    if not out_edge.dst_conn in map_entry.in_connectors:
                        map_entry.add_in_connector(out_edge.dst_conn)
                    in_edges = state.in_edges(new_node)
                    in_edge = None
                    for edge in in_edges:
                        if edge.src_conn[4:] == out_edge.dst_conn[3:]:
                            in_edge = edge
                            break
                    if in_edge and not in_edge.src_conn in map_entry.out_connectors:
                        new_node.add_out_connector(in_edge.src_conn)

            elif isinstance(btw_node, dace.nodes.MapEntry):
                memlet = edge1.data
                tmp = edge1.dst_conn
                if tmp:
                    btw_node.remove_in_connector(tmp)
                tmp = edge2.src_conn
                if tmp:
                    map_entry.remove_out_connector(tmp)
                in_conn = edge1.src_conn
                dst_conn = edge2.dst_conn
                new_map_entry.remove_out_connector(edge2.src_conn)
                new_map_entry.remove_in_connector(edge1.dst_conn)
                state.add_memlet_path(edge1.src, map_entry, edge2.dst, memlet=memlet, src_conn=in_conn, dst_conn=dst_conn)

        # remove the edges that are not needed anymore
        for edge1, btw_node, edge2 in path_nodes:
            try:
                if not edge1 in keep:
                    state.remove_edge(edge1)
            except:
                pass
            try:
                state.remove_edge(edge2)
            except:
                pass

        # add empty edge if first map exit is not connected anymore
        if len(state.in_edges(new_map_exit)) == 0:
            if keep:
                dangeling_node = keep.pop().dst
                state.add_nedge(dangeling_node, new_map_exit, dace.memlet.Memlet())
                # enforce first map scope is executed before the second one
                if len(state.out_edges(map_entry)) == 0:
                    in_nodes = [edge.src for edge in state.in_edges(map_entry)]
                    for node in in_nodes:
                        if node.label == dangeling_node.label:
                            state.add_nedge(new_map_exit, node, dace.memlet.Memlet())
                            break

        # find paths in current sdfg that have to be redirected
        scope_nodes = []
        self.find_scope_nodes(state, map_entry, scope_nodes)
        fixable_paths = []
        redirection_maps = []
        for node in scope_nodes[1:]:
            in_edges = state.in_edges(node)
            for edge in in_edges:
                if edge.src not in scope_nodes:
                    if isinstance(edge.src, dace.nodes.MapEntry):
                        redirection_maps.append((edge, node))
                    else:
                        fixable_paths.append((edge, node))
        
        fixable_maps = []
        for edge, node in redirection_maps:
            in_edges = state.in_edges(edge.src)
            for edge2 in in_edges:
                if edge2.dst_conn[3:] == edge.src_conn[4:]:
                    fixable_maps.append((edge2, edge, node))
        
        # tmp variables that are produced in first map scope and used in second map scope have to be redirected
        # if node has to outgoing edges, node is copied, else redirected
        for edge, node in fixable_paths:
            if isinstance(edge.src, dace.nodes.AccessNode):
                input_edge = state.in_edges(edge.src)[0]
                src_conn = input_edge.src_conn
                dst_conn = edge.dst_conn
                new_node = copy.deepcopy(edge.src)
                out_set = {edge.dst for edge in state.out_edges(edge.src)}
                if len(out_set) == 1:
                    state.add_memlet_path(input_edge.src, new_map_exit, new_node, memlet=input_edge.data, src_conn=src_conn, dst_conn=dst_conn)
                    state.remove_node(edge.src)
                state.add_memlet_path(new_node, map_entry, node, memlet=edge.data, src_conn=edge.src_conn, dst_conn=edge.dst_conn)
                try:
                    state.remove_edge(edge)
                except:
                    pass
            elif isinstance(edge.src, dace.nodes.Tasklet):
                # should not happen
                pass
        
        # output edges of previous map scope that are only used in first map scope have to be redirected to new map exit
        exit_node = state.exit_node(map_entry)
        exit_input = [(edge, edge.src) for edge in state.in_edges(exit_node)]
        for edge, node in exit_input:
            if not node in scope_nodes:
                memlet = edge.data
                dst_edge = None
                for edge2 in state.out_edges(exit_node):
                    if edge2.src_conn[4:] == edge.dst_conn[3:]:
                        dst_edge = edge2
                        break
                if dst_edge:
                    exit_node.remove_in_connector(edge.dst_conn)
                    exit_node.remove_out_connector(dst_edge.src_conn)
                    state.add_memlet_path(node, new_map_exit, dst_edge.dst, memlet=memlet, src_conn=edge.src_conn, dst_conn=dst_edge.dst_conn)
                    state.remove_edge(edge)
                    state.remove_edge(dst_edge)
                
        # input of previous map scope that are only used in second new map scope can directly be redirected to the new map scope
        for edge2, edge, node in fixable_maps:
            new_map_entry.remove_in_connector(edge2.dst_conn)
            new_map_entry.remove_out_connector(edge.src_conn)
            state.add_memlet_path(edge2.src, map_entry, node, memlet=edge.data, src_conn=edge2.src_conn, dst_conn=edge.dst_conn)
            state.remove_edge(edge2)
            state.remove_edge(edge)
        
        # maybe fuse input nodes of second map if the are the same
        input_nodes = [edge.src for edge in state.in_edges(map_entry)]
        in_map = {}
        for node1 in input_nodes:
            if node1.label not in in_map:
                in_map[node1.label] = set()
            in_map[node1.label].add(node1)
        for key, value in in_map.items():
            if len(value) > 1:
                chosen_node = None
                for node in value:
                    if state.in_edges(node):
                        chosen_node = node
                        break
                if not chosen_node:
                    chosen_node = value.pop()
                for node in value:
                    if node == chosen_node:
                        continue
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        dace.transformation.helpers.redirect_edge(state, edge, new_src = chosen_node)
                    state.remove_node(node)
        
        # some nodes from the first map scope that are hanging without any connection can be reconnect potentially
        scope_1_nodes = []
        self.find_scope_nodes(state, new_map_entry, scope_1_nodes)
        for node in scope_1_nodes[1:]:
            if isinstance(node, dace.nodes.AccessNode):
                out_edges = state.out_edges(node)
                if not out_edges or (out_edges[0] and not out_edges[0].data.data):
                    dst_node = [edge.src for edge in state.in_edges(map_entry) if edge.src.label == node.label][0]
                    if dst_node and state.in_edges(dst_node) and state.in_edges(dst_node)[0].data.data:
                        continue
                    elif dst_node and (not state.in_edges(dst_node) or (state.in_edges(dst_node)[0] and not state.in_edges(dst_node)[0].data.data)):
                        src_edge = state.in_edges(node)[0]
                        state.add_memlet_path(src_edge.src, new_map_exit, dst_node, memlet=src_edge.data, src_conn=src_edge.src_conn, dst_conn=src_edge.dst_conn)
                        state.remove_node(node)
                        if state.in_edges(dst_node)[0] and not state.in_edges(dst_node)[0].data.data:
                            state.remove_edge(state.in_edges(dst_node)[0])
                        continue

    def find_scope_nodes(self, state, node, scope_nodes, visited=None):
        if visited is None:
            visited = set()

        visited.add(node)
        scope_nodes.append(node)
        next_edges = [edge for edge in state.out_edges(node)]
        for edge in next_edges:
            if isinstance(edge.dst, dace.nodes.MapExit):
                continue
            elif not edge.dst in visited:
                self.find_scope_nodes(state, edge.dst, scope_nodes, visited)
        return    