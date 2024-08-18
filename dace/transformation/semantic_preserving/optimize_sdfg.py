import sys
import copy
from dace.transformation.dataflow.buffer_tiling import BufferTiling
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.dataflow.map_fission import MapFission
from dace import registry, sdfg as sd, memlet as mm, subsets, data as dt
from dace.symbolic import pystr_to_symbolic
from copy import deepcopy as dcpy
from dace.sdfg.propagation import propagate_memlets_state
from dace.transformation import transformation
import dace
import numpy as np
import argparse
import math
import sympy
parser = argparse.ArgumentParser(description='Test kernels.')
parser.add_argument('--validate', action='store_true', help='see if the generated code is correct')
parser.add_argument('--print', action='store_true', help='show printed code')
parser.add_argument('--opt', type=str, help='optimization to apply', default='')
parser.add_argument('--kernel', type=str, help='kernel to apply the optimization to', default='gemm')
parser.add_argument('--sym', action='store_true', help='Symbolic or Constant Loop Sizes', default=True)
args = parser.parse_args()



class SwapScope():

    def find_swappable_maps(self, sdfg):
        swappable_maps = []
        for state in sdfg.nodes():
            for edges in state.edges():
                if isinstance(edges.src, dace.nodes.MapEntry) and isinstance(edges.dst, dace.nodes.MapEntry):
                    src = edges.src.map
                    dst = edges.dst.map
                    for i, param in enumerate(src.params):
                        if param in dst.params:
                            continue
                    swappable_maps.append((edges.src, edges.dst))
        return swappable_maps
    
    def find_exit_maps(self, sdfg, map1, map2):
        map1 = map1.map
        map2 = map2.map
        exit1 = None
        exit2 = None
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapExit):
                    if node.map == map1:
                        exit1 = node
                    if node.map == map2:
                        exit2 = node
        return exit1, exit2

    def apply(self, sdfg, maps):
        map_entry1, map_entry2 = maps
        assert isinstance(map_entry1, dace.sdfg.nodes.MapEntry)
        assert isinstance(map_entry2, dace.sdfg.nodes.MapEntry)
        if not (map_entry1, map_entry2) in self.find_swappable_maps(sdfg):
            return
        exit1, exit2 = self.find_exit_maps(sdfg, map_entry1, map_entry2)
        assert exit1 is not None
        assert exit2 is not None

        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.dst == map_entry1:
                    dace.transformation.helpers.redirect_edge(state, edge, new_dst = map_entry2)
                elif edge.src == map_entry2:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = map_entry1)
                elif edge.src == map_entry1 and edge.dst == map_entry2:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = map_entry2, new_dst = map_entry1)
                elif edge.dst == exit2:
                    dace.transformation.helpers.redirect_edge(state, edge, new_dst = exit1)
                elif edge.src == exit1:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = exit2)
                elif edge.src == exit2 and edge.dst == exit1:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = exit1, new_dst = exit2)
        
class SwapMapScope():

    def find_swappable_maps(self, sdfg):
        result = []
        map_scopes = self.find_map_scopes(sdfg)
        for i in range(len(map_scopes)):
            entry1, exit1, state1 = map_scopes[i]
            next = False
            for j in range(i+1, len(map_scopes)):
                entry2, exit2, state2 = map_scopes[j]
                if not state.entry_node(entry1) == state.entry_node(entry2):
                    continue
                if not state1 == state2:
                    continue
                if next:
                    break
                next = True
                output_1 = {edge.data.data for edge in state.out_edges(exit1)}
                input_2 = {edge.data.data for edge in state.in_edges(entry2)}
                if input_2 & output_1:
                    continue
                result.append((entry1, entry2))
               
        return result
    
    def find_map_scopes(self, sdfg):
        map_scopes = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    #map_scopes.append(dace.sdfg.scope._scope_subgraph(state, node, True, True))
                    map_scopes.append((node, state.exit_node(node), state))
                    
        return map_scopes
                    
    def apply(self, sdfg, entry1, entry2):
        if not (entry1, exit2) in self.find_swappable_ops(sdfg):
            return
        state = find_state(sdfg, entry2)
        exit2 = state.exit_node(entry2)
        dst_node = [edge.dst for edge in state(sdfg).out_edges(exit2)][0]
        state.add_nedge(dst_node, entry1, mm.Memlet())

class SwapOpertions():
    def find_swappable_ops(self, sdfg):
        result = []
        tasklets = self.find_tasklets(sdfg)
        for i in range(len(tasklets)):
            tasklet1, state1 = tasklets[i]
            next = False
            for j in range(i+1, len(tasklets)):
                tasklet2, state2 = tasklets[j]
                if not state1 == state2:
                    continue
                if not state1.entry_node(tasklet1) == state2.entry_node(tasklet2):
                    continue
                if next:
                    break
                next = True
                output_1 = {edge.data.data for edge in state1.out_edges(tasklet1)}
                input_2 = {edge.data.data for edge in state2.in_edges(tasklet2)}
                if input_2 & output_1:
                    continue
                result.append((tasklet1, tasklet2))
        return result
    
    def find_tasklets(self, sdfg):
        result = []
        for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.Tasklet):
                        result.append((node, state))
        return result
    
    def apply(self, sdfg, op1, op2):
        if not (op1, op2) in self.find_swappable_ops(sdfg):
            return
        state = find_state(sdfg, op2)
        out_node = [edge.dst for edge in state.out_edges(op2)][0]
        state.add_nedge(out_node, op1, mm.Memlet())

class LoopTile():

    def find_tileable_maps(self, sdfg):
        tileable_maps = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    map = node.map
                    dim = map.range.size()[0]
                    if isinstance(dim, sympy.core.numbers.Integer) and int(dim) > 1:
                        dim = int(dim)
                        divisors = [i for i in range(2, math.floor(math.sqrt(dim))) if dim % i == 0]
                        tileable_maps.extend([(node, tile_size) for tile_size in divisors])
                    if isinstance(dim, sympy.core.symbol.Symbol):
                        tileable_maps.extend([(node, tile_size) for tile_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]])
        return tileable_maps
    
    def apply(self, sdfg, map_entry, tile_size):
        if not (map_entry, tile_size) in self.find_tileable_maps(sdfg):
            return
        assert isinstance(map_entry, dace.sdfg.nodes.MapEntry)
        assert len(map_entry.map.params) == 1
        tile_dict = {map_entry.map.params[0]: tile_size}
        if isinstance(map_entry.range.size()[0], sympy.core.symbol.Symbol):
            dace.transformation.helpers.tile(sdfg=sdfg, map_entry=map_entry, divides_evenly=False, skew=True, **tile_dict)
        else:
            dace.transformation.helpers.tile(sdfg=sdfg, map_entry=map_entry, divides_evenly=True, skew=True, **tile_dict)

class JoinMapScopes():

    def find_joinable_mapscope(self, sdfg):
        pass
    def apply(self):
        pass

class SplitMapScopes():

    def traverse_scope(self, state, entry):
        map = entry.map
        ops = []
        def traverse(node, sec=False):
            if node in ops:
                return
            if isinstance(node, dace.nodes.MapExit) and node.map == map:
                return
            if isinstance(node, dace.nodes.Tasklet):
                if sec:
                    ops.append(node)
                else: 
                    sec = True
            for edge in state.edges():
                if edge.src == node:
                    traverse(edge.dst, sec)
        traverse(entry)

        return ops
    
    def exit_node(self, state, entry):
        def traverse(node, exit=False):
            if isinstance(node, dace.nodes.MapExit) and node.map == entry.map:
                exit = True
            if isinstance(node, dace.nodes.AccessNode) and exit:
                return node
            for edge in state.edges():
                if edge.src == node:
                    return traverse(edge.dst, exit)
    
        return traverse(entry)

        
    def find_splitable_scopes(self, sdfg):
        splitable_scopes = []
        entry_set = set()
        for state in sdfg.nodes():
            for edge in state.edges():
                if isinstance(edge.src, dace.nodes.MapEntry) and isinstance(edge.dst, dace.nodes.Tasklet):
                    map = edge.src
                    if not map in entry_set:
                        entry_set.add(map)
                        scope_ops = self.traverse_scope(state, map)
                        if scope_ops:
                            splitable_scopes.extend(scope_ops)
                            #splitable_scopes.append(self.exit_node(state, map))
        return splitable_scopes
    
    #unused
    def find_exit_maps(self, sdfg, map):
        map = map.map
        exit = None
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapExit):
                    if node.map == map:
                        exit = node
        return exit
    
    def find_input_nodes(self, sdfg, node):
        input_nodes = []
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.dst == node:
                    input_nodes.append(edge.src)
        return input_nodes
    
    def find_ingoing_edges(self, sdfg, node):
        ingoing_edges = []
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.dst == node:
                    ingoing_edges.append(edge)
        return ingoing_edges
    
    def find_outgoing_edges(self, sdfg, node):
        outgoing_edges = []
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.src == node:
                    outgoing_edges.append(edge)
        return outgoing_edges
    
    def find_edges(state, entry, target):
        edges = []
        def traverse(node, entry, target, passed=False):
            if node == target:
                passed = True
            for edge in state.edges():
                if edge.src == node:
                    if edge.dst == target:
                        edges.append(edge)
                        return
                    traverse(edge.dst, entry, target)
        for edge in state.edges():
            if edge.src == entry:
                traverse(edge.dst, entry, target)
        return edges

    def copy_map_properties(self, map):
        new_map = dace.nodes.Map(map.label, map.params, map.range)
        new_map.unroll = map.unroll
        new_map.schedule = map.schedule
        return new_map
    
    def traverse_upwards(self, state, start_node, stop_map, visited = None, maps = None):
        if visited is None:
            visited = set()
        if maps is None:
            maps = set()
    
        # Stop condition if the start_node is a MapEntry with the specified name
        if isinstance(start_node, dace.nodes.MapExit) and not start_node.map.label == stop_map:
            return 

        # Add the current node to the visited set
        visited.add(id(start_node))
        if isinstance(start_node, dace.nodes.MapEntry):
            exit_map = state.exit_node(start_node)
            maps.add((start_node, exit_map))

        # Traverse upwards from the current node
        for edge in state.in_edges(start_node):
            src_node = edge.src
            if id(src_node) not in visited:
                self.traverse_upwards(state, src_node, stop_map, visited, maps)
    

    def traverse_downwards(self, state, start_node, stop_map,visited = None, maps = None, start=True):
        if visited is None:
            visited = set()
        if maps is None:
            maps = set()
        print(f'start node {start_node}')
        # Stop condition if the start_node is a MapEntry with the specified name
        if isinstance(start_node, dace.nodes.MapEntry) and not start_node.map.label == stop_map:
            return

        # Add the current node to the visited set
        visited.add(id(start_node))
        
        # fuse exit and entry nodes if they are connected by a single access node
        if isinstance(start_node, dace.nodes.AccessNode) and not start:
            in_nodes = [edge.src for edge in state.in_edges(start_node)]
            out_nodes = [edge.dst for edge in state.out_edges(start_node)]
            in_map = None
            out_map = None
            for node in in_nodes:
                if isinstance(node, dace.nodes.MapExit):
                    in_map = node
                    break
            for node in out_nodes:
                if isinstance(node, dace.nodes.MapEntry):
                    out_map = node
                    break
            if in_map and out_map:
                self.checkMap(state, in_map, out_map)
                transform = dace.transformation.dataflow.map_fusion.MapFusion()
                transform.first_map_exit = in_map
                transform.second_map_entry = out_map
                if transform.can_be_applied(state, 0, sdfg):
                    print(f'fusion on {start_node}')
                    transform.apply(state, sdfg)
                    sdfg.view()
                    self.traverse_downwards(state, in_map, stop_map, visited, maps, True)
                    return

        # Traverse downwars from the current node
        out_edges = state.out_edges(start_node)
        for edge in out_edges:
            dst_node = edge.dst
            #if id(dst_node) not in visited:
            self.traverse_downwards(state, dst_node, stop_map, visited, maps, False)
        

    def find_fusion_map(self, state, node, end_nodes, target_nodes, failed, visited = None):
        if node in end_nodes:
            return None
        
        if visited is None:
            visited = set()
        
        visited.add(id(node))

        if isinstance(node, dace.nodes.AccessNode) and not node in target_nodes and not node in failed:
            in_nodes = [edge.src for edge in state.in_edges(node)]
            out_nodes = [edge.dst for edge in state.out_edges(node)]
            in_map = None
            out_map = None
            for in_node in in_nodes:
                if isinstance(in_node, dace.nodes.MapExit):
                    in_map = in_node
                    break
            for out_node in out_nodes:
                if isinstance(out_node, dace.nodes.MapEntry):
                    out_map = out_node
                    break
            if in_map and out_map:
                if self.checkMap(state, in_map, out_map):
                    return in_map, node, out_map
            
        for edge in state.out_edges(node):
            if id(edge.dst) not in visited:
                return self.find_fusion_map(state, edge.dst, end_nodes, target_nodes, visited)
        
    def checkMap(self, state, map1, map2):
        assert map1.map.params == map2.map.params
        assert map1.map.range == map2.map.range
        if  map1.label == map2.label:
            return True
        return False

    def apply4(self, sdfg, target):
        if not target in self.find_splitable_scopes(sdfg):
            return
        assert(isinstance(target, dace.sdfg.nodes.Tasklet))

        state = find_state(sdfg, target)

        in_edge = state.in_edges(target)
        in_nodes = [edge.src for edge in in_edge]

        current_node = target
        innermost_map, exit_map = self.find_scope_map(state, target)
        
        print(f'innermost scope {innermost_map}')
        print(f'in nodes {in_nodes}')

        map_label = innermost_map.map.label  + '_fission'
        start_node = state.in_edges(innermost_map)[0].src
        end_nodes = [edge.dst for edge in state.out_edges(exit_map)]

        transform = dace.transformation.dataflow.map_fission.MapFission()
        transform.map_entry = innermost_map
        assert transform.can_be_applied(state, 0, sdfg)
        transform.apply(state, sdfg)
        sdfg.view()

        fusion_entry, node, fusion_exit = None,None,None
        transform = dace.transformation.dataflow.map_fusion.MapFusion()
        failed = []
        while True:
            fusion_exit, node, fusion_entry = self.find_fusion_map(state, start_node, end_nodes, in_nodes, failed)
            if not fusion_entry:
                break
            transform.first_map_exit = fusion_exit
            transform.second_map_entry = fusion_entry
            fusion_entry = None
            print(f'fusion on {node} ->', end=' ')
            try:
                transform.apply(state, sdfg)
                print(f'success')
                #sdfg.view()
            except:
                print(f'failed')
                failed.append(node)
        
                
            
        """  
        v1, m1, v2, m2 = set(), set(), set(), set()
        for node in in_nodes:
            self.traverse_upwards(state, node, map_label, v1, m1)
            self.traverse_downwards(state, node, map_label, v2, m2)
        """
        #print(f'm1 {m1}')
        #print(f'm2 {m2}')
        """
        for entry, exit in m2:
            entry.map.label = entry.map.label + '_2'
            exit.map.label = exit.map.label + '_2'
        
        
        transform = dace.transformation.dataflow.map_fusion.MapFusion()
        m2 = list(m2)
        transform.first_map_exit = m2[0][1]
        transform.second_map_entry = m2[1][0]
        transform.apply(state, sdfg)
        
        transform = dace.transformation.dataflow.map_fusion.MapFusion()
        break_i = False
        for entry, exit in m2:
            out_edges = state.out_edges(exit)
            out_nodes = [edge.dst for edge in out_edges]
            for entry2, exit2 in m2:
                in_edges = state.in_edges(entry2)
                in_nodes = [edge.src for edge in in_edges]
                for node in out_nodes:
                    if node in in_nodes:
                        transform.first_map_exit = exit
                        transform.second_map_entry = entry2
                        #transform.array = node
                        #if transform.can_be_applied(state,0, sdfg):
                        try:
                            transform.apply(state, sdfg)
                        except:
                            print(f'failed on {node}')
                        
                        m2.remove((entry, exit))
                        m2.remove((entry2, exit2))
                        m2.add((entry, exit2))
                        
        """

        
    def apply(self, sdfg, target):
        if not target in self.find_splitable_scopes(sdfg):
            return
        assert isinstance(target, dace.sdfg.nodes.Tasklet)
       
        state = find_state(sdfg, target)
       
        current_node = target
        innermost_scope = None
        while current_node is not None:
            # Get the scope parent of the current node
            scope_parent = state.scope_dict()[current_node]
            
            # If there is no scope parent, break the loop
            if scope_parent is None:
                break
            
            # Update the innermost scope and the current node for the next iteration
            innermost_scope = scope_parent
            current_node = scope_parent
        
        entry = innermost_scope
        exit = state.exit_node(entry)

        input_nodes = [edge.src for edge in state.in_edges(target)]
        tasklet_edges = state.in_edges(target)  
        input_edges = [edge for node in input_nodes for edge in state.in_edges(node)]
        print("input nodes", input_nodes)
        print("tasklet edges", tasklet_edges)
        print("input edges", input_edges)

        new_map = self.copy_map_properties(entry.map)
        new_map_entry = dace.nodes.MapEntry(new_map)
        new_map_exit = dace.nodes.MapExit(new_map)

        output_edges = []
        for node in input_nodes:
            output_edges.extend(state.out_edges(node))

        inbetween_edges = list(set(output_edges) | set(tasklet_edges))


        # redirect edges that have nothin to do with the split
        tmp = []
        for edge in state.edges():
            if edge.src == entry:
                if (not self.passes_through(state, edge, input_nodes)):
                    tmp.append((edge.data.data, edge))

        special_edges = []
        for edge in state.edges():
            if edge.dst == entry:
                for data, snd_edge in tmp:
                    if edge.data.data == data:
                        tmp.remove((data, snd_edge))
                        special_edges.append((edge, snd_edge))

        print(f'inbetween edges {inbetween_edges}')
        print(f'special edges {special_edges}')

        # redirect edges
        for edge in input_edges:
            dace.transformation.helpers.reconnect_edge_through_map(state, edge, new_map_exit, False)
        for edge in inbetween_edges:
            dace.transformation.helpers.reconnect_edge_through_map(state, edge, new_map_entry, True)
        for edge, snd_edge in special_edges:
            new_edge1 = sdfg.add_edge(edge.src, new_map_entry, edge)
            sdfg.remove_edge(edge)
            new_edge2 = sdfg.add_edge(new_map_entry, snd_edge.dst, snd_edge)
            sdfg.remove_edge(snd_edge)
            #new_edge = dace.transformation.helpers.redirect_edge(state, edge, new_src = edge.src, new_dst = snd_edge.dst)
            #dace.transformation.helpers.reconnect_edge_through_map(state, new_edge, new_map_entry, True)
    
    def find_scope_map(self, state, target):
        current_node = target
        innermost_scope = None
        while current_node is not None:
            # Get the scope parent of the current node
            scope_parent = state.scope_dict()[current_node]
            
            # If there is no scope parent, break the loop
            if scope_parent is None:
                break
            
            # Update the innermost scope and the current node for the next iteration
            innermost_scope = scope_parent
            current_node = scope_parent
        if innermost_scope:
            return innermost_scope, state.exit_node(innermost_scope)
        return None, None

    def apply2(self, sdfg, target):
        if not target in self.find_splitable_scopes(sdfg):
            return
        state = find_state(sdfg, target)
        entry, exit = self.find_scope_map(state, target)
        assert entry and exit

        subgraph = state.scope_subgraph(entry, include_entry=False, include_exit=False)
        modified_arrays = set()
        outer_map = entry
        mapsize = outer_map.range.size()

        components = [(target, target)]
        for _,cout in components:   
            sources = subgraph.source_nodes()
            sinks = subgraph.sink_nodes()

        external_edges_entry = list(state.out_edges(entry))
        external_edges_exit = list(state.in_edges(exit))

        edge_to_outer = {}
        for edge in external_edges_entry:
            path = state.memlet_path(edge)
            eindex = path.index(edge)
            edge_to_outer[edge] = path[eindex-1]
        for edge in external_edges_exit:
            path = state.memlet_path(edge)
            eindex = path.index(edge)
            edge_to_outer[edge] = path[eindex+1]

        arrays = MapFission._border_arrays(sdfg, state, subgraph)
        scalars = {}
        for _,cout in components:
            for e in subgraph.out_edges(cout):
                if isinstance(e.dst, dace.nodes.CodeNode):
                    if not e.data.data in scalars:
                        scalars[e.data.data] = [e]
                    else:
                        scalars[e.data.data].append(e)

        for scalar, edges in scalars.items():
            desc = sdfg.arrays[scalar]
            del sdfg.arrays[scalar]
            name, newdesc = sdfg.add_transient(scalar,
                                                     mapsize,
                                                     desc.dtype,
                                                     desc.storage,
                                                     lifetime=desc.lifetime,
                                                     debuginfo=desc.debuginfo,
                                                     allow_conflicts=desc.allow_conflicts,
                                                     find_new_name=True)
            for edge in edges:
                anode = state.add_access(name)
                sbs = subsets.Range.from_string(','.join(outer_map.params))
                # Offset memlet by map range begin (to fit the transient)
                sbs.offset([r[0] for r in outer_map.range], True)
                state.add_edge(edge.src, edge.src_conn, anode, None,
                                mm.Memlet.simple(name, sbs, num_accesses=outer_map.range.num_elements()))
                state.add_edge(anode, None, edge.dst, edge.dst_conn,
                                mm.Memlet.simple(name, sbs, num_accesses=outer_map.range.num_elements()))
                state.remove_edge(edge)

        new_map_entries = []
        for component_in, component_out in components:
            me, mx = state.add_map(outer_map.label + '_fission', [(p, '0:1') for p in outer_map.params],
                                       outer_map.schedule,
                                       unroll=outer_map.unroll,
                                       debuginfo=outer_map.debuginfo)
            
            for conn in entry.in_connectors:
                if not conn.startswith('IN_'):
                    me.add_in_connector(conn)
            
            me.map.range = dcpy(outer_map.range)
            new_map_entries.append(me)

            conn_idx = 0
            for e in state.in_edges(component_in):
                    if e.data.data:
                        in_conn = f"IN_{conn_idx}"
                        out_conn = f"OUT_{conn_idx}"
                        conn_idx += 1
                        me.add_in_connector(in_conn)
                        me.add_out_connector(out_conn)
                    else:
                        in_conn = None
                        out_conn = None
                    state.add_edge(me, out_conn, e.dst, e.dst_conn, dcpy(e.data))
                    # Reconnect inner edges at source directly to external nodes
                    if e in external_edges_entry:
                        state.add_edge(edge_to_outer[e].src, edge_to_outer[e].src_conn, me, in_conn,
                                       dcpy(edge_to_outer[e].data))
                    else:
                        state.add_edge(e.src, e.src_conn, me, in_conn, dcpy(e.data))
                    state.remove_edge(e)
            # Empty memlet edge in nested SDFGs
            if state.in_degree(component_in) == 0:
                state.add_edge(me, None, component_in, None, mm.Memlet())

            conn_idx = 0
            for e in state.out_edges(component_out):
                if e.data.data:
                    in_conn = f"IN_{conn_idx}"
                    out_conn = f"OUT_{conn_idx}"
                    conn_idx += 1
                    mx.add_in_connector(in_conn)
                    mx.add_out_connector(out_conn)
                else:
                    in_conn = None
                    out_conn = None
                state.add_edge(e.src, e.src_conn, mx, in_conn, dcpy(e.data))
                # Reconnect inner edges at sink directly to external nodes
                if e in external_edges_exit:
                    state.add_edge(mx, out_conn, edge_to_outer[e].dst, edge_to_outer[e].dst_conn,
                                    dcpy(edge_to_outer[e].data))
                else:
                    state.add_edge(mx, out_conn, e.dst, e.dst_conn, dcpy(e.data))
                state.remove_edge(e)
            # Empty memlet edge in nested SDFGs
            if state.out_degree(component_out) == 0:
                state.add_edge(component_out, None, mx, None, mm.Memlet())
            
        for node in sources:
            if isinstance(node, dace.nodes.AccessNode):
                for edge in state.in_edges(node):
                    outer_edge = edge_to_outer[edge]
                    memlet = dcpy(edge.data) 
                    memlet.subset = subsets.Range(outer_map.range.ranges + memlet.subset.ranges)
                    state.add_edge(outer_edge.src, outer_edge.src_conn, edge.dst, edge.dst_conn, memlet)
        for node in sinks:
            if isinstance(node, dace.nodes.AccessNode):
                for edge in state.out_edges(node):
                    outer_edge = edge_to_outer[edge]
                    state.add_edge(edge.src, edge.src_conn, outer_edge.dst, outer_edge.dst_conn,
                                    dcpy(outer_edge.data))
            # Augment arrays by prepending map dimensions
        for array in arrays:
            if array in modified_arrays:
                continue
            desc = sdfg.arrays[array]
            if isinstance(desc, dt.Scalar):  # Scalar needs to be augmented to an array
                desc = dt.Array(desc.dtype, desc.shape, desc.transient, desc.allow_conflicts, desc.storage,
                                desc.location, desc.strides, desc.offset, False, desc.lifetime, 0, desc.debuginfo,
                                desc.total_size, desc.start_offset)
                sdfg.arrays[array] = desc
            for sz in reversed(mapsize):
                desc.strides = [desc.total_size] + list(desc.strides)
                desc.total_size = desc.total_size * sz

            desc.shape = mapsize + list(desc.shape)
            desc.offset = [0] * len(mapsize) + list(desc.offset)
            modified_arrays.add(array)

        # Fill scope connectors so that memlets can be tracked below
        state.fill_scope_connectors()

        for node in subgraph.nodes():
                if isinstance(node, dace.nodes.AccessNode) and node.data in arrays:
                    for edge in state.all_edges(node):
                        for e in state.memlet_tree(edge):
                            # Prepend map dimensions to memlet
                            # NOTE: Do this only for the subset corresponding to `node.data`. If the edge is copying
                            # to/from another AccessNode, the other data may not need extra dimensions. For example, see
                            # `test.transformations.mapfission_test.MapFissionTest.test_array_copy_outside_scope`.
                            if e.data.data == node.data:
                                if e.data.subset:
                                    e.data.subset = subsets.Range([(pystr_to_symbolic(d) - r[0],
                                                                    pystr_to_symbolic(d) - r[0], 1)
                                                                   for d, r in zip(outer_map.params, outer_map.range)] +
                                                                  e.data.subset.ranges)
                            else:
                                if e.data.other_subset:
                                    e.data.other_subset = subsets.Range(
                                        [(pystr_to_symbolic(d) - r[0], pystr_to_symbolic(d) - r[0], 1)
                                         for d, r in zip(outer_map.params, outer_map.range)] +
                                        e.data.other_subset.ranges)

        state.remove_nodes_from([entry, exit])
        propagate_memlets_state(sdfg,state)

    def passes_through(self, state, cur_edge, nodes):
        if cur_edge.dst in nodes:
            return True
        else:
            for edge in state.edges():
                if edge.src == cur_edge.dst:
                    if self.passes_through(state, edge, nodes):
                        return True
        return False
    
    def apply3(self,sdfg,target):
        if not target in self.find_splitable_scopes(sdfg):
            return
        state = find_state(sdfg, target)

        entry, exit = self.find_scope_map(state, target)
        new_map = self.copy_map_properties(entry.map)
        new_map_entry = dace.nodes.MapEntry(new_map)
        new_map_exit = dace.nodes.MapExit(new_map)

        subgraph =  state.scope_subgraph(entry, include_entry=True, include_exit=True)
        map_in_node = [(edge.src,edge) for node in subgraph.source_nodes() for edge in state.in_edges(node)]
        input_nodes = [edge.src for edge in state.in_edges(target)]
        scope1_exit_edge = [edge for node in input_nodes for edge in state.in_edges(node)]
        #scope1_entry_edge = [edge for node in subgraph.source_nodes() for edge in state.in_edges(node)]
        scope2_entry_edge = [edge for node in input_nodes for edge in state.out_edges(node)]
        #scope2_exit_edge = [edge for node in subgraph.sink_nodes() for edge in state.out_edges(node)]

        for edge in scope1_exit_edge:
            dace.transformation.helpers.reconnect_edge_through_map(state, edge, new_map_exit, False)
        for edge in scope2_entry_edge:
            dace.transformation.helpers.reconnect_edge_through_map(state, edge, new_map_entry, True)
        subgraph1 = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        subgraph2 = state.scope_subgraph(new_map_entry, include_entry=False, include_exit=True)
        print(f'subgraph1 {subgraph1.nodes()}')

        print(subgraph1.edges())
        for edge in subgraph1.edges():
           
            if not self.passes_through(subgraph1, edge, [new_map_exit]):
                print(f'edge {edge}, {edge.data.data}')
                if isinstance(edge.src, dace.nodes.MapEntry):
                    red_edge = None
                    for node,edge2 in map_in_node:
                        print(f'\t{edge2}, {edge2.data.data}')
                        if edge2.data.data == edge.data.data:
                            red_edge = edge2
                            break
                    if red_edge:
                        dace.transformation.helpers.redirect_edge(state, red_edge, new_dst=new_map_entry, new_dst_conn=f'IN_{edge.data.data}')
                        dace.transformation.helpers.redirect_edge(state, edge, new_src = new_map_entry, new_src_conn=f'OUT_{edge.data.data}')
                else:
                    if isinstance(edge.src,dace.nodes.AccessNode):
                        pass

    
        propagate_memlets_state(sdfg,state)
        
class UnrollScope():
    max_unroll_factor = 10
    def find_unrollable_maps(self, sdfg):
        """
        Find maps that can be unrolled
        Only last level maps with constant size can be unrolled
        """
        unrollable_maps = []
        for state in sdfg.nodes():
            for edge in state.edges():
                if isinstance(edge.src, dace.nodes.MapEntry) and isinstance(edge.dst, dace.nodes.Tasklet):
                    dim = edge.src.map.range.size()[0]
                    if isinstance(dim, sympy.core.numbers.Integer) and int(dim) > 1:
                        divisors = [i for i in range(2, max(self.max_unroll_factor, dim)) if dim % i == 0]
                        unrollable_maps.extend([(edge.src, div) for div in divisors])
        return unrollable_maps
    
    def apply(self, sdfg, map_entry, unroll_factor):
        if not (map_entry, unroll_factor) in self.find_unrollable_maps(sdfg):
            return
        assert isinstance(map_entry, dace.sdfg.nodes.MapEntry)
        #dace.transformation.dataflow.hbm_transform.HbmTransform.unroll_map(sdfg, state, map_entry, unroll_factor)
        #map_entry.map.unroll = True
        para = map_entry.map.params[0]
        unroll_dict = {para: unroll_factor}
        dace.transformation.helpers.tile(sdfg=sdfg, map_entry=map_entry, divides_evenly=True, skew=True, **unroll_dict)
        for state in sdfg.nodes():
            for edge in state.edges():
                if isinstance(edge.dst, dace.nodes.MapEntry) and edge.dst.map.params[0] == para:
                    new_map = edge.dst.map
                    new_map.unroll = True
                    new_map.schedule = dace.ScheduleType.Unrolled
                    break

class addArrayCopies():
    def find_copyable_arrays(self, sdfg):
        copyable_arrays = set()
        for state in sdfg.nodes():
           for node in state.nodes():
               if isinstance(node, dace.nodes.AccessNode):
                   copyable_arrays.add(node.data)

        return sorted(list(copyable_arrays))
    
    def apply(self, sdfg, arr_name):
        if not arr_name in self.find_copyable_arrays(sdfg):
            return
        
        state = state_0(sdfg)
        data = sdfg.arrays[arr_name]
        input_nodes = state.source_nodes()
        output_nodes = state.sink_nodes()
        adj_state = state
        if arr_name in [node.data for node in input_nodes]:
            s = len(sdfg.nodes())
            new_state = sdfg.add_state(f'state_{s}')
            sdfg.add_edge(new_state, sdfg.nodes()[s-1], dace.InterstateEdge())
            adj_state = new_state
        
        new_name = ''
        i = 0
        while True:
            new_name = f'{arr_name}_copy_{i}'
            if not new_name in sdfg.arrays:
                break
            i += 1
        """
        transient = True
        if arr_name in [node.data for node in output_nodes]:
            transient = False
            o_data = sdfg.arrays[arr_name]
            o_data.transient = True
        """
        sdfg.add_array(name=new_name, shape=data.shape, dtype=data.dtype, storage=data.storage, transient=True, lifetime=data.lifetime)
        
        tasklet = adj_state.add_tasklet("copy_temp", {"in_1"}, {"out"}, "out = in_1")
        
       
        output_node = adj_state.add_access(new_name)
        if arr_name in [node.data for node in input_nodes]:
            # input nodes
            new_input_node = new_state.add_read(arr_name)
            loop_entry, loop_exit, idxs = self.add_maps(adj_state, data.shape)
            adj_state.add_memlet_path(new_input_node, *loop_entry, tasklet, dst_conn='in_1', memlet=dace.Memlet.simple(new_input_node, idxs))
            adj_state.add_memlet_path(tasklet, *loop_exit, output_node, src_conn='out', memlet=dace.Memlet.simple(output_node, idxs))
            state.replace(arr_name, new_name)
        else:
            input_node = None
            for edge in state.edges():
                if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == arr_name:
                    input_node = edge.dst
                    break 
            out_edges = state.out_edges(input_node)
            entry = state.entry_node(input_node)
            if not entry and out_edges:
                # map exit nodes
                loop_entry, loop_exit, idxs = self.add_maps(adj_state, data.shape)
                state.add_memlet_path(input_node, *loop_entry, tasklet, dst_conn='in_1', memlet=dace.Memlet.simple(input_node, idxs))
                state.add_memlet_path(tasklet, *loop_exit, output_node, src_conn='out', memlet=dace.Memlet.simple(output_node, idxs))
                for edge in out_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = output_node)
                self.traverse_n_replace(state, output_node, arr_name, new_name)
            elif not entry:
                #output node => to keep naming consistent, we switch temporary and original array
                in_edges = state.in_edges(input_node)
                loop_entry, loop_exit, idxs = self.add_maps(adj_state, data.shape)
                state.add_memlet_path(output_node, *loop_entry, tasklet, dst_conn='in_1', memlet=dace.Memlet.simple(output_node, idxs))
                state.add_memlet_path(tasklet, *loop_exit, input_node, src_conn='out', memlet=dace.Memlet.simple(input_node, idxs))
                for edge in in_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_dst = output_node)
                self.traverse_n_replace(state, output_node, arr_name, new_name, backwards=True)
            else:
                # nodes that are only used inside a scope
                entries = [entry]
                while True:
                    tmp = entries[-1]
                    e = state.entry_node(tmp)
                    if not e:
                        break
                    entries.append(e)
                dims = list(reversed([entry.map.range.size()[0] for entry in entries]))
                idxs = ''
                for shape in data.shape:
                    if not shape in dims:
                        #not implemented: if map operates not over all of the shape dimension
                        assert False
                    else:
                        idx = dims.index(shape)
                        idxs += f'i{idx},'
                idxs = idxs[:-1]
                state.add_edge(input_node, None, tasklet, 'in_1', dace.Memlet.simple(input_node, out_edges[0].data.subset))
                state.add_edge(tasklet, 'out', output_node, None, dace.Memlet.simple(output_node, idxs))
                for edge in out_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = output_node)
                self.traverse_n_replace(state, output_node, arr_name, new_name)
                
    def add_maps(self, state, shape):
        loop_entry, loop_exit = [], []
        idxs = ''
        for i,s in enumerate(shape):
            dim = str(s)
            index = f'i{i}'
            idxs += f'{index},'
            entry, exit = state.add_map(f'copy_map_{i}', {index: f'0:{dim}'})
            loop_entry.append(entry)
            loop_exit.append(exit)
        idxs = idxs[:-1]
        return loop_entry, loop_exit, idxs

    def traverse_n_replace(self, state, node, old_name, new_name, start=True, backwards=False):
        if not start:
            if not isinstance(node, dace.nodes.MapEntry) and not isinstance(node, dace.nodes.MapExit):
                return
      
        next_edges = None
        if backwards:
            next_edges = state.in_edges(node)
        else:
            next_edges = state.out_edges(node)

        for edge in next_edges:
            if edge.data.data == old_name:
                edge.data.data = new_name
            if backwards:
                self.traverse_n_replace(state, edge.src, old_name, new_name, False, True)
            else:
                self.traverse_n_replace(state, edge.dst, old_name, new_name, False)

class removeArrayCopies():

    def find_deleteable_arrays(self, sdfg):
        deleteable_arrays = set()
        for state in sdfg.nodes():
            out = [node.data for node in state.sink_nodes()]
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    if node.label.startswith('copy_temp') or node.label.startswith('move'):
                        out_node = state.out_edges(node)[0].data
                        if out_node.data in out:
                            continue
                        in_node = state.in_edges(node)[0].data
                        if in_node.subset == out_node.subset:
                            deleteable_arrays.add(out_node.data)
        
        return sorted(list(deleteable_arrays))
    
    def apply(self, sdfg, arr_name):
        if not arr_name in self.find_deleteable_arrays(sdfg):
            return
        data = sdfg.arrays[arr_name]
        arr_node = None
        state = None
        for tstate in sdfg.nodes():
            for edge in tstate.edges():
                if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == arr_name:
                    arr_node = edge.dst
                    state = tstate
                    break

        out_edges = state.out_edges(arr_node)
        entry = state.entry_node(arr_node)

        if entry:
            in_edge = state.in_edges(arr_node)[0]
            tasklet = in_edge.src
            orig_node = state.in_edges(tasklet)[0].src
            orig_name = orig_node.data
            for edge in out_edges:
                dace.transformation.helpers.redirect_edge(state, edge, new_src = orig_node)
            self.delete_copy(state, arr_node, orig_node)
            self.traverse_n_replace(state, orig_node, arr_name, orig_name)
        else:
            tasklet = None
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet) and(node.label.startswith('copy_temp') or node.label.startswith('move')):
                    if state.out_edges(node)[0].data.data == arr_name:
                        tasklet = node
                        break
            entry = state.entry_node(tasklet)
            while True:
                t = state.entry_node(entry)
                if t:
                    entry = t
                else:
                    break
            orig_node = state.in_edges(entry)[0].src
            orig_name = orig_node.data
            if orig_node in state.source_nodes() and not out_edges:
                # orig_node is input node
                sdfg.remove_node(state)
                state_0(sdfg).replace(arr_name, orig_name)
            else:
                #rmv_edge = state.out_edges(orig_node)[0]
                for edge in out_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = orig_node)
                self.traverse_n_replace(state, orig_node, arr_name, orig_name)
                self.delete_copy(state, arr_node, orig_node)
    
        sdfg.remove_data(arr_name)

    def delete_copy(self, state, node, stop_node):
        if isinstance(node, dace.nodes.AccessNode) and node == stop_node:
            return
        in_edge = state.in_edges(node)
        state.remove_node(node)
        for edge in in_edge:
            src = edge.src
            #state.remove_edge(edge)
            self.delete_copy(state, src, stop_node)

    def traverse_n_replace(self, state, node, old_name, new_name, start=True, backwards=False):
        if not start:
            if not isinstance(node, dace.nodes.MapEntry) and not isinstance(node, dace.nodes.MapExit):
                return
      
        next_edges = None
        if backwards:
            next_edges = state.in_edges(node)
        else:
            next_edges = state.out_edges(node)

        for edge in next_edges:
            if edge.data.data == old_name:
                edge.data.data = new_name
            if backwards:
                self.traverse_n_replace(state, edge.src, old_name, new_name, False, True)
            else:
                self.traverse_n_replace(state, edge.dst, old_name, new_name, False)
                
class addArrayDims():
    """
    complex, keep for later
    """
    def get_entry_maps(self, state, node):
        dims = []
        entry = state.entry_node(node)
        if not entry:
            return dims
        
        while True:
            dims.append(entry)
            entry = state.entry_node(entry)
            if not entry:
                break
        return dims

    def only_one_child(self, state, entry):
        out_edges = state.out_edges(entry)
        outs = set()
        for edge in out_edges:
            outs.add(edge.dst)
        if not len(outs) == 1:
            return False

        out = list(outs)[0]
        if isinstance(out, dace.nodes.MapEntry):
            return True
        elif isinstance(out, dace.nodes.Tasklet):
            out_edge = state.out_edges(out)[0]
            if isinstance(out_edge.dst, dace.nodes.MapExit):
                return True
        else:
            return False   

    def single_map_scope(self,state, node):
        if not state.entry_node(node):
            return True
        
        for edge in state.in_edges(node):
            if isinstance(edge.src, dace.nodes.AccessNode):
                return False
        for out_edge in state.out_edges(node):
            if isinstance(out_edge.dst, dace.nodes.AccessNode):
                return False
        
        return True
                      

    def find_createable_dim(self, sdfg):
        result = []

        # --- read to write access, write is target
        read_dims = {}
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        data_name = edge.data.data
                        if not data_name in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[data_name]
                        if not data_node.transient:
                            continue
                        if not self.single_map_scope(state, node):
                            continue
                        tmp = set(edge.data.subset)
                        subset = set()
                        for subs in tmp:
                            for idx in subs:
                                subset.add(str(idx))

                        entry_maps = self.get_entry_maps(state, node)
                        map_params = {m.map.params[0] for m in entry_maps}
                        unused_params = map_params - subset
                        unused_dims = set()

                        for entry in entry_maps:
                           
                            for param in unused_params:
                                if param in entry.map.params:
                                    unused_dims.add(entry.map.range.size()[0])
                    
                        if data_name in read_dims:
                            read_dims[data_name] &= unused_dims
                        else:
                            read_dims[data_name] = unused_dims

        #print(f'read_dims {read_dims}')
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        if not edge.data.data in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[edge.data.data]
                        if not data_node.transient:
                            continue
                        if not read_dims.get(edge.data.data, set()):
                            continue
                        result.append((node, 0))

        # --- write to read access, read is target
        write_dims = {}
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        if not edge.data.data in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[edge.data.data]
                        if not data_node.transient:
                            continue
                        if not edge.data.wcr:
                            continue
                        if not self.single_map_scope(state, node):
                            continue
                        tmp = set(edge.data.subset)
                        subset = set()
                        for subs in tmp:
                            for idx in subs:
                                subset.add(str(idx))
                        entry_maps = self.get_entry_maps(state, node)
                        map_params = {m.map.params[0] for m in entry_maps}
                        unused_params = map_params - subset
                        unused_dims = set()
                        for entry in entry_maps:
                            for param in unused_params:
                                if param in entry.map.params:
                                    unused_dims.add(entry.map.range.size()[0])

                        write_dims[edge.data.data] = (unused_dims, node)

        #print(f'write_dims {write_dims}')
        candidates = set()
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        if not edge.data.data in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[edge.data.data]
                        if not data_node.transient:
                            continue
                        if not edge.data.data in write_dims:
                            continue
                        unused_dims, write_node = write_dims[edge.data.data]
                        if not "reduce" in node.label or ("reduce" in node.label and not node.label == write_node.label) or not node.label in ["move", "copy_temp"]:
                            continue
                        candidates.add(edge.data.data)

        #print(f'candidates {candidates}')
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        if edge.data.data in candidates:
                            result.append((node, 1))
                            candidates.remove(edge.data.data)

        return result

    def apply(self, sdfg, op, mode):
        if not (op, mode) in self.find_createable_dim(sdfg):
            return
        

        # mode 0: read to write access
        if mode == 0:
            target = state_0(sdfg).out_edges(op)[0].data.data

            read_tasklet = self.find_read_tasklet(sdfg, target)
            write_tasklet = self.find_write_tasklet(sdfg, target)
           
            common_unused = None
            for i_state, node in read_tasklet:
                if not i_state.out_edges(node)[0].data.data == target:
                    continue
                in_edge = None
                for edge in i_state.in_edges(node):
                    if edge.data.data == target:
                        in_edge = edge
                        break
                map_scopes = self.get_entry_maps(i_state, node)
                edge_subset = [str(idx[0]) for idx in in_edge.data.subset]
                map_params = sorted({m.map.params[0] for m in map_scopes} - set(edge_subset))

                unused_dims = []
                for entry in map_scopes:
                    for param in map_params:
                        if param in entry.map.params:
                            unused_dims.append(entry.map.range.size()[0])

                if common_unused is None:
                    common_unused = unused_dims
                else:
                    tmp = common_unused & unused_dims
                    common_unused = [dim for dim in common_unused if dim in tmp]
            
            unused_dim = common_unused[0]

            self.adjust_data_array(sdfg, target, unused_dim)
      
            for i_state, op_node in write_tasklet:
                self.add_new_map(i_state, op_node, unused_dim)

            for i,(i_state, node) in enumerate(write_tasklet):
                self.replace_write_memlet(i_state, node, target, unused_dim)
                
            for i_state, node in read_tasklet:
                self.replace_read_memlet(i_state, node, target, unused_dim)
                
        # mode 1: write to read access
        elif mode == 1:
            target = state_0(sdfg).out_edges(op)[0].data.data
            wcr = state_0(sdfg).out_edges(op)[0].data.wcr
            unused_dims = set()
            write_tasklet = self.find_write_tasklet(sdfg, target)
            unused_dims = self.find_unused_dims(sdfg, write_tasklet, write=True)
            unused_dim = unused_dims[0]

            self.adjust_data_array(sdfg, target, unused_dim)

            for i_state, tasklet_node in write_tasklet:
                self.replace_write_memlet(i_state, tasklet_node, target, unused_dim)
            
            for i_state, tasklet_node in read_tasklet:
                if not i_state.out_edges(tasklet_node)[0].data.data == target:
                    self.add_new_map(i_state, tasklet_node, unused_dim)
            
            for i_state, tasklet_node in read_tasklet:
                self.replace_read_memlet(i_state, tasklet_node, target, unused_dim)
                self.add_reduction(i_state, tasklet_node, target, wcr)
            
            for i_state, tasklet_node in write_tasklet:
                self.delete_reduction(i_state, tasklet_node)

    def find_write_tasklet(self, sdfg, target):
        result = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        if edge.data.data == target:
                            result.append((state, node))
        return result
    
    def find_read_tasklet(self, sdfg, target):  
        result = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        if edge.data.data == target:
                            result.append((state, node))
        return result
    
    def find_unused_dims(self, sdfg, tasklets, write=True):
        result = []
        for state, node in tasklets:
            edges = []
            if write:
                edges = state.out_edges(node)
            else:
                edges = state.in_edges(node)
            for edge in edges:
                entry_maps = self.get_entry_maps(state, node)
                unacc_indices = {entry.map.params[0] for entry in entry_maps}
                unacc_indices -= set(edge.data.subset)
                for entry in entry_maps:
                    if entry.map.params[0] in unacc_indices:
                        result.append(entry.map.range.size()[0])
        return result

    def adjust_data_array(self, sdfg, target, unused_dim):
        # shape adjustment
        shape = list(sdfg.arrays[target].shape)
        shape.append(unused_dim)
        sdfg.arrays[target].shape = tuple(shape)
        # total size adjustment
        sdfg.arrays[target].total_size *= unused_dim
        offset = list(sdfg.arrays[target].offset)
        offset.append(0)
        # offset adjustment
        sdfg.arrays[target].offset = tuple(offset)
        # strides adjustment
        strides = sdfg.arrays[target].strides
        new_strides = []
        for stride in strides:
            new_strides.append(stride*unused_dim)
        new_strides.append(1)
        sdfg.arrays[target].strides = tuple(new_strides)

    def add_new_map(self, state, op_node, unused_dim):
        entries = self.get_entry_maps(state, op_node)
        new_idx = f'i{len(entries)}'
        #idxs.append(new_idx)
        new_entry, new_exit = state.add_map(f'map_{len(entries)}_{op_node.label}', {new_idx: f'0:{unused_dim}'})
        
        # incorporate new map
        for edge in state.in_edges(op_node):
            #memlet = copy.deepcopy(edge.data)
            memlet = None
            if edge.data.data:
                memlet = edge.data.__deepcopy__({})
                state.add_memlet_path(edge.src, new_entry, op_node, memlet=memlet, dst_conn=edge.dst_conn, src_conn=edge.src_conn)
            else:
                state.add_nedge(edge.src, new_entry, data=dace.Memlet())
                state.add_nedge(new_entry, op_node, data=dace.Memlet())
            state.remove_edge(edge)
        
        for edge in state.out_edges(op_node):
            memlet = edge.data.__deepcopy__({})
            state.add_memlet_path(op_node, new_exit, edge.dst, memlet=memlet, src_conn=edge.src_conn, dst_conn=edge.dst_conn)
            state.remove_edge(edge)

    def replace_write_memlet(self, state, node, target, unused_dim):
        entries = self.get_entry_maps(state, node)
        idx = None
        for entry in entries:
            if str(unused_dim) == str(entry.map.range.size()[0]):
                idx = entry.map.params[0]
                break

        write_edges = []
        self.get_write_memlet(state, node, target, write_edges)
        subset = list(write_edges[0].data.subset.ranges)
        subset.append((idx, idx, 1))
        new_subset = dace.subsets.Range(subset)

        path_nodes = [edge.src for edge in write_edges]
        path_nodes.append(write_edges[-1].dst)
        out_conn = write_edges[0].src_conn
        wcr = write_edges[0].data.wcr
        for edge in write_edges:
            src_conn = edge.src_conn
            if src_conn:
                edge.src.remove_out_connector(src_conn)
            dst_conn = edge.dst_conn
            if dst_conn:
                edge.dst.remove_in_connector(dst_conn)
            state.remove_edge(edge)
        path_nodes[0].add_out_connector(out_conn)
        state.add_memlet_path(*path_nodes, memlet=dace.Memlet(data=target, subset= new_subset, wcr=wcr), src_conn=out_conn)

    def get_write_memlet(self, state, node, dst, result, traverse=None):
        if traverse is None:
            traverse = set()
        for edge in state.out_edges(node):
            if edge.data.data == dst:
                #subset = list(edge.data.subset.ranges)
                #subset.append((new_idx, new_idx, 1))
                #edge.data.subset = dace.subsets.Range(subset)
                result.append(edge)
            if not edge.dst in traverse:
                if not (isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == dst):
                    traverse.add(edge.dst)
                    self.get_write_memlet(state, edge.dst, dst, result, traverse)
    
    def delete_reduction(self, state, node, visited=None):
        if visited is None:
            visited = set()
        out_edges = state.out_edges(node)
        for edge in out_edges:
            edge.data.wcr = None
            if not edge.dst in visited and not isinstance(edge.dst, dace.nodes.AccessNode):
                visited.add(edge.dst)
                self.delete_reduction(state, edge.dst, visited)
    
    def add_reduction(self, state, node, wcr, visited=None):
        if visited is None:
            visited = set()
        out_edges = state.out_edges(node)
        for edge in out_edges:
           edge.data.wcr = wcr
           if not edge.dst in visited and not isinstance(edge.dst, dace.nodes.AccessNode):
                visited.add(edge.dst)
                self.add_reduction(state, edge.dst, wcr, visited)

    def replace_read_memlet(self, state, node, target, unused_dim):
        entries = self.get_entry_maps(state, node)
        idx = None
        for entry in entries:
            if str(unused_dim) == str(entry.map.range.size()[0]):
                idx = entry.map.params[0]
                break
        
        read_edges = []
        self.get_read_memlet(state, node, target, read_edges)
        subset = list(read_edges[0].data.subset.ranges)
        subset.append((idx, idx, 1))
        #subset.append(idx)
        new_subset = dace.subsets.Range(subset)
        
        path_nodes = [edge.src for edge in reversed(read_edges)]
        path_nodes.append(read_edges[0].dst)
        out_conn = read_edges[0].dst_conn
        for edge in read_edges:
            src_conn = edge.src_conn
            if src_conn:
                edge.src.remove_out_connector(src_conn)
            dst_conn = edge.dst_conn
            if dst_conn:
                edge.dst.remove_in_connector(dst_conn)
            state.remove_edge(edge)

        path_nodes[-1].add_in_connector(out_conn)
        state.add_memlet_path(*path_nodes, memlet=dace.Memlet(data=target, subset= new_subset), dst_conn=out_conn)

    def get_read_memlet(self, state, node, target, result, traverse=None):
        if traverse is None:
            traverse = set()
        for edge in state.in_edges(node):
            if edge.data.data == target:
                result.append(edge)
            if not edge.src in traverse:
                traverse.add(edge.src)
                if not (isinstance(edge.src, dace.nodes.AccessNode) and edge.src.data == target):
                    self.get_read_memlet(state, edge.src, target, result, traverse)
    
class removeArrayDims():

    def find_deleteable_dims(self, sdfg):
        result = []
        for name, data in sdfg.arrays.items():
            if not data.transient:
                continue
            discovered_rtw, discovered_wtr = self.deleteable_dim_candidates(sdfg, name)
            if discovered_rtw is not None:
                map_dim = discovered_rtw.map.range.size()[0]
                result.append((name, map_dim, 1))
            if discovered_wtr is not None:
                map_dim = discovered_wtr.map.range.size()[0]
                result.append((name, map_dim, 0))
        return result

    def deleteable_dim_candidates(self, sdfg, array_name):
        write_tasklets = self.find_write_tasklet(sdfg, array_name)
        read_tasklets = self.find_read_tasklet(sdfg, array_name)

        discovered_rtw = None
        discovered_wtr = None
        
        for w_state, w_node in write_tasklets:
            if not w_state.label == 'state_0':
                continue
            map_scopes = self.get_entry_maps(w_state, w_node)
            subset = w_state.out_edges(w_node)[0].data.subset
            for write_loc, widx in reversed(list(enumerate(subset))):
                idx = str(widx[0])
                if len(idx) != 2:
                    continue
                candidate_write_scope = [entry for entry in map_scopes if entry.map.params[0] == idx][0]
                candidate_dim = candidate_write_scope.map.range.size()[0]
                candidate_rejected = False
                read_scope_indices = []
                for r_state, r_node in read_tasklets:
                    if array_name == r_state.out_edges(r_node)[0].data.data:
                        continue
                    candidate_read_scope = None
                    index_dim_map = {entry.map.params[0]: entry.map.range.size()[0] for entry in self.get_entry_maps(r_state, r_node)}
                    for edge in r_state.in_edges(r_node):
                        if not edge.data.data == array_name:
                            continue
                        r_subset = edge.data.subset
                        for read_loc, ridx in reversed(list(enumerate(r_subset))):
                            read_idx = str(ridx[0])
                            if not index_dim_map[read_idx] == candidate_dim:
                                continue
                            #if not idx in read_idx:
                            #    continue
                            if len(read_idx) != 2:
                                candidate_rejected = True
                                break
                            if read_loc != write_loc:
                                candidate_rejected = True
                                break
                            cur_read_scope = [entry for entry in self.get_entry_maps(r_state, r_node) if entry.map.params[0] == read_idx][0]
                            if candidate_read_scope is None:
                                candidate_read_scope = cur_read_scope
                                read_scope_indices.append(candidate_read_scope.map.params[0])
                            elif candidate_read_scope != cur_read_scope:
                                candidate_rejected = True
                                break
                            read_dim = cur_read_scope.map.range.size()[0]
                            if read_dim != candidate_dim:
                                candidate_rejected = True
                                break
                    if candidate_read_scope is None:
                        candidate_rejected = True
                if candidate_rejected:
                    continue
                
                if discovered_rtw is None:
                    rtw_rejected = False
                    for edge in w_state.in_edges(w_node):
                        if edge.data.data == array_name:
                            continue
                        edge_subset = [str(idx[0]) for idx in list(edge.data.subset)]
                        if candidate_write_scope.map.params[0] in edge_subset:
                            rtw_rejected = True
                    if not rtw_rejected:
                        discovered_rtw = candidate_write_scope
                
                if discovered_wtr is None:
                    wtr_rejected = False
                    for (rstate, rop), read_map_idx in zip(read_tasklets, read_scope_indices):
                        r_map_scopes = self.get_entry_maps(rstate, rop)
                        read_scope_idx = [entry.map.params[0] for entry in r_map_scopes if entry.map.params[0] == read_map_idx][0]
                        out_subset = [str(idx[0]) for idx in rstate.out_edges(rop)[0].data.subset]
                        if read_scope_idx in out_subset:
                            wtr_rejected = True
                    
                    common_read_reduce = None
                    for rstate, rop in read_tasklets:
                        out_edge = rstate.out_edges(rop)[0]
                        if out_edge.data.wcr:
                            if common_read_reduce is None:
                                common_read_reduce = rop.label
                            elif common_read_reduce != rop.label:
                                wtr_rejected = True

                    if not (w_node.label == 'move' or w_node.label == 'copy_temp'):
                        if common_read_reduce != w_node.label:
                            wtr_rejected = True
                    if not wtr_rejected:
                        discovered_wtr = candidate_write_scope

        return discovered_rtw, discovered_wtr

    def get_entry_maps(self, state, node):
        dims = []
        entry = state.entry_node(node)
        if not entry:
            return dims
        
        while True:
            dims.append(entry)
            entry = state.entry_node(entry)
            if not entry:
                break
        return dims

    def apply(self, sdfg, name, dim, mode):
        if not (name, dim, mode) in self.find_deleteable_dims(sdfg):
            return
        
        self.adjust_data_array(sdfg, name, dim)
        self.remove_read_dim(sdfg, name, dim)
        self.remove_write_dim(sdfg, name, dim)

        w_tasklet = self.find_write_tasklet(sdfg, name)
        if mode == 0:
            r_tasklet = self.find_read_tasklet(sdfg, name)
            reducation_type = r_tasklet[0][0].out_edges(r_tasklet[0][1])[0].data.wcr
            for state, node in w_tasklet:
                if state.label == 'state_0':
                    self.add_reduction(state, node, reducation_type)
            
            for state, node in r_tasklet:
                target_entry = None
                entries = self.get_entry_maps(state, node)
                for i, tmp_e in enumerate(entries):
                    if tmp_e.map.range.size()[0] == dim:
                        target_entry = tmp_e
                        break
                exit = state.exit_node(target_entry)
                self.fix_memlet_volume(state, node, name, target_entry, exit, dim)
                self.remove_map(state, node, target_entry, exit)
                self.delete_reduction(state, node, name)
        else:
            for state, node in w_tasklet:
                target_entry = None
                entries = self.get_entry_maps(state, node)
                for i, tmp_e in enumerate(entries):
                    if tmp_e.map.range.size()[0] == dim:
                        target_entry = tmp_e
                        break
                exit = state.exit_node(target_entry)
                self.fix_memlet_volume(state, node, name, target_entry, exit, dim)

                self.remove_map(state, node, target_entry, exit)
    
    def add_reduction(self, state, node, wcr, name):
        out_edges = state.out_edges(node)
        if isinstance(node, dace.nodes.AccessNode) and node.data == name:
            return
        for edge in out_edges:
            edge.data.wcr = wcr
            self.add_reduction(state, edge.dst, wcr)
    
    def delete_reduction(self, state, node, name):
        out_edges = state.out_edges(node)
        if isinstance(node, dace.nodes.AccessNode):
            return
        for edge in out_edges:
            edge.data.wcr = None
            self.delete_reduction(state, edge.dst, name)
    
    def remove_map(self, state, node, entry, exit):
        for out_edge in state.out_edges(entry):
                    for in_edge in state.in_edges(entry):
                        if in_edge.data.data == out_edge.data.data:
                            dace.transformation.helpers.redirect_edge(state, out_edge, new_src = in_edge.src)
                
        for in_edge in state.in_edges(exit):
            for out_edge in state.out_edges(exit):
                if in_edge.data.data == out_edge.data.data:
                    dace.transformation.helpers.redirect_edge(state, in_edge, new_dst = out_edge.dst)
        
        state.remove_node(exit)
        state.remove_node(entry)

    def fix_memlet_volume(self, state, node, name, entry, exit, dim):

        def traverse(node, del_map, backwards=False, traversed=None, passed=False):
            if traversed is None:
               traversed = set()
            next_edges = state.in_edges(node) if backwards else state.out_edges(node)
            for edge in next_edges:
                if edge.data.data == name and passed:
                    volume = edge.data.volume / dim
                    edge.data.volume = volume
                elif not edge.data.data == name:
                    continue
                next_node = edge.src if backwards else edge.dst
                if not next_node in traversed and not (isinstance(next_node, dace.nodes.AccessNode) and next_node.data == name):
                    traversed.add(next_node)
                    t = passed
                    if next_node == del_map:
                        t = True
                    traverse(next_node, del_map, backwards, traversed, t)
        traverse(node, entry, backwards=True, traversed=None, passed=False)
        traverse(node, exit, backwards=False, traversed=None, passed=False)
        

    def remove_read_dim(self, sdfg, target, dim):
        r_tasklet = self.find_read_tasklet(sdfg, target)
        for state, node in r_tasklet:
            entries = self.get_entry_maps(state, node)
            idx = [entry.map.params[0] for entry in entries if entry.map.range.size()[0] == dim][0]
            self.traverse_and_replace(state, node, target, dim, idx, backwards=True)
    
    def remove_write_dim(self, sdfg, target, dim):
        w_tasklet = self.find_write_tasklet(sdfg, target)
        for state, node in w_tasklet:
            entries = self.get_entry_maps(state, node)
            idx = [entry.map.params[0] for entry in entries if entry.map.range.size()[0] == dim][0]
            
            self.traverse_and_replace(state, node, target, dim, idx)
    
    def traverse_and_replace(self, state, node, name, dim, idx, backwards=False, traversed=None):
        if traversed is None:
            traversed = set()
        
        if isinstance(node, dace.nodes.AccessNode) and node.data == name:
            return

        next_edges = None
        if backwards:
            next_edges = state.in_edges(node)
        else:
            next_edges = state.out_edges(node)
        for edge in next_edges:
            if edge.data.data == name:
                subset = list(edge.data.subset.ranges)
                new_subset = []
                replaced = False
                for sub_idx in reversed(subset):
                    if (str(sub_idx[0]) == str(idx)) and not replaced:
                        replaced = True
                        continue
                    new_subset.append(sub_idx)
                if not replaced:
                    new_subset = []
                    for sub_idx in reversed(subset):
                        if sub_idx[1]+1 == dim and not replaced:
                            replaced = True
                            continue
                        new_subset.append(sub_idx)
                new_subset = dace.subsets.Range(reversed(new_subset))
                edge.data.subset = new_subset
            
            if not node in traversed:
                traversed.add(node)
                if backwards:
                    self.traverse_and_replace(state, edge.src, name, dim, idx, backwards, traversed)
                else:
                    self.traverse_and_replace(state, edge.dst, name, dim, idx, backwards, traversed)

    def adjust_data_array(self, sdfg, target, dim):
        data_array = sdfg.arrays[target]
        data_array.total_size //= dim
        shape = list(data_array.shape)
        new_shape = []
        removed = False
        pos = -1
        length = len(shape)
        for i,s_dim in reversed(list(enumerate(shape))):
            if s_dim == dim and not removed:
                removed = True
                pos = i
            else:
                new_shape.append(s_dim)   
        data_array.shape = tuple(new_shape)
        strides = list(data_array.strides)
        new_strides = []
        for j in range(len(strides)-1):
            if j == pos:
                continue
            new_strides.append(strides[j]//dim)
        data_array.strides = tuple(new_strides)
        offset = list(data_array.offset)
        new_offset = []
        for k in range(len(offset)-1):
            if k == pos:
                continue
            new_offset.append(offset[k])
        data_array.offset = tuple(new_offset)

    def find_write_tasklet(self, sdfg, target):
        result = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        if edge.data.data == target:
                            result.append((state, node))
        return result
    
    def find_read_tasklet(self, sdfg, target):  
        result = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        if edge.data.data == target:
                            result.append((state, node))
                            break
        return result

class reuseArray():
    """
    change semantics and fzse two arrays completley 
    """
    def find_reusable_arrays(self, sdfg):
        same_data = set()
        data = sdfg.arrays

        all_nodes = set()
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    all_nodes.add(node.data)
        
        for arr_name1, data1 in data.items():
            for arr_name2, data2 in data.items():
                if arr_name1 == arr_name2:
                    continue
                if not type(data1) == type(data2):
                    continue
                if isinstance(data1, dace.data.Array):
                    if data1.shape == data2.shape and data1.dtype == data2.dtype:
                        same_data.add((arr_name1, arr_name2)) if arr_name1 < arr_name2 else same_data.add((arr_name2, arr_name1))
                if isinstance(data1, dace.data.Scalar):
                    if not arr_name1 in all_nodes or not arr_name2 in all_nodes:
                        continue
                    if data1.dtype == data2.dtype:
                        same_data.add((arr_name1, arr_name2)) if arr_name1 < arr_name2 else same_data.add((arr_name2, arr_name1))
                        
    
      
        #outputs = sdfg.sink_nodes()
        state = None
        for tmpstate in sdfg.nodes():
            if tmpstate.label == 'state_0':
                state = tmpstate
        inputs = state.source_nodes()
        input_names = [node.data for node in inputs]
        reusable_arrays = []

        for arr1, arr2 in same_data:

            works = True
            if not arr2 in input_names:
                for node in inputs:
                    check = self.traverser_check(state, arr1, arr2, node, False)
                    if not check:
                        works = False
                        break
            else:
                works = False
            
            if not arr1 in input_names:
                if not works:
                    works = True
                    for node in inputs:
                        check = self.traverser_check(state, arr2, arr1, node, False)
                        if not check:
                            works = False
                            break
                    
            if works:
                reusable_arrays.append((arr1, arr2))
        
        #TODO: check if arr1 used as input and arr2 as output in same map scope, that their index are matching => no index mismatch

        return sorted((reusable_arrays))
    
    def traverser_check(self, state, arr1, arr2, node, write, visited = None):

        if visited is None:
            visited = set()
        visited.add(node)

        # last read arr1 < first write arr2
        if isinstance(node, dace.nodes.Tasklet):
            in_edges = state.in_edges(node)
            out_edges = state.out_edges(node)
            for edge in in_edges:
                if edge.data.data == arr1:
                    if write:
                        return False
            for edge in out_edges:
                if edge.data.data == arr2:
                    write = True
        
        out_edges = state.out_edges(node)
        for edge in out_edges:
            if not edge.dst in visited:
                ret = self.traverser_check(state, arr1, arr2, edge.dst, write, visited)
                if not ret:
                    return False
                
        return True
    
    def apply(self, sdfg, array1, array2):
        if not (array1, array2) in self.find_reusable_arrays(sdfg):
            return
        
     
        for state in sdfg.nodes():
            state.replace(array1, array2)

        data1 = sdfg.arrays[array1]
        if not data1.transient:
            data2 = sdfg.arrays[array2]
            data2.transient = False
        
        #sdfg.remove_data(array1)
        # keep array1 for now, information not lost

        """
        data1 = sdfg.arrays[array1]
        print(f'old {data1} and {sdfg.arrays[array2]}')
        sdfg.arrays[array2] = data1

        does not work: Invalid SDFG: Duplicate data descriptor object detected: "<array1>". Please copy objects rather than using multiple references to the same one
        """

class reuseArrayLocations():

    def get_read_ops(self, state, target):
        ops = set()
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                in_edges = state.in_edges(node)
                for edge in in_edges:
                    if edge.data.data == target.data:
                        ops.add(node)
                        continue
        return ops
    
    def get_write_ops(self, state, target):
        ops = set()
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                out_edges = state.out_edges(node)
                for edge in out_edges:
                    if edge.data.data == target.data:
                        ops.add(node)
        return ops
    
    def get_entry_maps(self, state, node):
        dims = []
        entry = state.entry_node(node)

        while True:
            dims.append(entry)
            entry = state.entry_node(entry)
            if not entry:
                break

        return dims
    
    def find_reusable_dimensions(self, sdfg):
        reusable_dims = set()
        for state in sdfg.nodes():
            #source, sink = state.source_nodes(), state.sink_nodes()
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    data_node = sdfg.arrays[node.data]
                    if not data_node.transient:
                        continue
                    if data_node.shape is [1]:
                        continue
                    if not state.entry_node(node):
                        continue
                    read_ops = self.get_read_ops(state, node)
                    write_ops = self.get_write_ops(state, node)
                    shape = data_node.shape
                    tmp = set()

                    for op in write_ops:
                        maps = self.get_entry_maps(state, op)
                        tmp = {map_entry for map_entry in maps if map_entry.range.size()[0] in shape}
             
                    for op in read_ops:
                        entry_dims = self.get_entry_maps(state, op)
                        if set(tmp).issubset(set(entry_dims)):
                            for maps in tmp:
                                reusable_dims.add((node, maps.range.size()[0]))
                        else:
                            diff = set(tmp) - set(entry_dims)
                            for maps in diff:
                                reusable_dims.add((node, maps.range.size()[0]))

        return sorted(list(reusable_dims), key=lambda x: x[0].data)
                                    
                    
    def apply(self, sdfg, node, dim):
        if not (node,dim) in self.find_reusable_dimensions(sdfg):
            return
        
        data_node = sdfg.arrays[node.data]
        shape = data_node.shape
        strides = list(data_node.strides)
        print(f'old strides: {data_node.strides}')
        new_dim = []
        for i,dims in enumerate(shape):
            if dims == dim:
                strides[i] = 0
        data_node.strides = tuple(strides)
        print(f'new strides: {data_node.strides}')
          
class StackToHeap():
    def apply(self, data):
        data.storage = dace.StorageType.CPU_Heap

class HeapToStack():
    def apply(self, data):
        data.storage = dace.StorageType.Register

class changeMemoryLayout():
    def find_changeable_arrays(self, sdfg):
        changeable_arrays = set()
        for state in sdfg.nodes():
            source, sink = state.source_nodes(), state.sink_nodes()
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    if not node in source and not node in sink:
                        changeable_arrays.add(node)
        
        return (list(changeable_arrays))
    
    def apply(self, sdfg, node, dim_order):
        if not node in self.find_changeable_arrays(sdfg):
            return
        data = sdfg.arrays[node.data]
        shape = data.shape

        # check if order of dimension is correct
        assert len(shape) == len(dim_order)
        assert max(dim_order) == len(shape) - 1
        assert min(dim_order) == 0

        print(node.data)
        print(shape)
        print(dim_order)

        m = dict(zip(shape, dim_order))
        new_strides = list()
        for dim in shape:
            stride = 1
            for dim2 in shape:
                if m[dim2] > m[dim]:
                    stride *= dim2
            new_strides.append(stride)

        # set new stride
        print(f'old strides: {data.strides}, new strides: {new_strides}')
        data.strides = tuple(new_strides)
        return
    
        """
        state = state_0(sdfg)
        #in_edges = state.in_edges(node)
        #out_edges = state.out_edges(node)
        orig_shape = data.shape
        assert len(orig_shape) == len(dim_order)
        new_shape = [orig_shape[dim_order.index(i)] for i in range(len(orig_shape))]
        print(f'orig shape: {orig_shape}, new shape: {new_shape}')
        #edge = in_edges[0]

        entry = state.entry_node(node)
        if entry:
            dim_to_idx = {}
            while True:
                idx = entry.map.params[0]
                dim = entry.map.range.size()[0]
                dim_to_idx[dim] = idx
                entry = state.entry_node(entry)
                if not entry:
                    break

        else:
            up_map = {}
            down_map = {}
            self.get_loop_indices(state, node.data, node, up_map, downwards=False)
            self.get_loop_indices(state, node.data, node, down_map, downwards=True)
            print(up_map)
            print(down_map)

            self.replace_index(state, node.data, node, new_shape, up_map, downwards=False)
            self.replace_index(state, node.data, node, new_shape, down_map, downwards=True)

    
    def get_loop_indices(self, state, data, node, dim_map, visited = None, downwards=False):
        if visited is None:
            visited = []
        
        if downwards and isinstance(node, dace.nodes.MapEntry) or not downwards and isinstance(node, dace.nodes.MapExit):
            idx = node.map.params[0]
            dim = node.map.range.size()[0]
            dim_map[dim] = idx

        visited.append(node)

        adj_edges = state.out_edges(node) if downwards else state.in_edges(node)
        for edge in adj_edges:
            if not edge.data.data == data:
                continue
            target = edge.dst if downwards else edge.src
            if not target in visited:
                self.get_loop_indices(state, data, target, dim_map, visited, downwards)

        return 
    
    def replace_index(self, state, data, node, dim_order, dim_map, visited = None, downwards=False):
        if visited is None:
            visited = []
        
        adj_edges = state.out_edges(node) if downwards else state.in_edges(node)
        for edge in adj_edges:
            if not edge.data.data == data:
                continue
            else:
                print(edge.data.data)
                print(edge.data.subset)
                new_subset = edge.data.subset
                for i, subs in enumerate(edge.data.subset):
                    if isinstance(subs[1], dace.symbolic.symbol):
                        new_dim = dim_order[i]
                        new_idx = dim_map[new_dim]
                        print(f'{str(subs[1])} => {new_dim},{new_idx}')
                        if subs[1] == new_idx:
                            continue
                        for j,tmp in enumerate(edge.data.subset):
                            if list(tmp[1].free_symbols)[0] == new_dim or str(tmp[1]) == new_idx:
                                if (new_subset[j] == subs):
                                    break
                                a = tmp
                                new_subset[j] = subs
                                new_subset[i] = a
                                break
                    else:
                        dim = list(subs[1].free_symbols)[0]
                        new_dim = dim_order[i]
                        if dim == new_dim:
                            continue
                        new_idx = dim_map[new_dim]
                        print(f'{str(dim)} => {new_dim},{new_idx}')
                        for j,tmp in enumerate(edge.data.subset):
                              if list(tmp[1].free_symbols)[0] == new_dim or str(tmp[1]) == new_idx:
                                if (new_subset[j] == subs):
                                    break
                                a = tmp
                                new_subset[j] = subs
                                new_subset[i] = a
                                break
            
                edge.data.subset = new_subset
                print(edge.data.subset)
                print()

            target = edge.dst if downwards else edge.src
            if not target in visited:
                visited.append(target)
                self.replace_index(state, data, target, dim_order, dim_map, visited, downwards)
            
            return
        """

def get_source_node(state, node, data, visited = []):
    if isinstance(node, dace.nodes.AccessNode):
        if node.data == data:
            return node
    visited.append(node)

    for edge in state.in_edges(node):
        if not edge.src in visited:
            get_source_node(state, edge.src, data)
    
    return None

def get_sink_node(state, node, data, visited = []):
    if isinstance(node, dace.nodes.AccessNode):
        if node.data == data:
            return node
    visited.append(node)

    for edge in state.out_edges(node):
        if not edge.dst in visited:
            get_sink_node(state, edge.dst, data)
    
    return None

def find_state(sdfg, node):
    for sdfg_state in sdfg.nodes():
        if node in sdfg_state.nodes():
            return sdfg_state
        
def find_state_by_name(sdfg, name):
    for state in sdfg.nodes():
        for node in state.nodes():
          if isinstance(node, dace.nodes.AccessNode):
              if node.data == name:
                return state
          
def start_nodes(state):
    incoming_edges = {node: 0 for node in state.nodes()}
    for edge in state.edges():
        incoming_edges[edge.dst] += 1
    starting_nodes = [node for node, count in incoming_edges.items() if count == 0]
    return starting_nodes

def traverse_sdfg(state):
    pass

def order_sdfg(state):
    incoming_edges = {node: 0 for node in state.nodes()}
    for edge in state.edges():
        incoming_edges[edge.dst] += 1
    starting_nodes = [node for node, count in incoming_edges.items() if count == 0]

    order = []
    visited = set()

    # Stack to manage nested structures
    stack = [order]

    def traverse(node):
        if node in visited:
            return
        visited.add(node)

        # Check if node is a MapEntry, if so, create a new sublist
        if isinstance(node, dace.nodes.MapEntry):
            current_list = []
            stack[-1].append(node)
            stack[-1].append(current_list)
            stack.append(current_list)
        else:
            stack[-1].append(node)

        for edge in state.edges():
            if edge.src == node:
                traverse(edge.dst)

        # If node is a MapExit, pop the last list from the stack
        if isinstance(node, dace.nodes.MapExit):
            stack.pop()

    for node in starting_nodes:
        traverse(node)

    return order

def state_0(sdfg):
    for state in sdfg.nodes():
        if state.label == 'state_0':
            return state
        
def optimize_sdfg(sdfg, optimization):
    # Apply transformations
    pass

if __name__ == "__main__":
    print("==== Program start ====")

    source = ''
    sdfg_name = ''
    if args.kernel == "gemm":
        postfix = ""
        if args.sym:
            source = "test_kernel/test_gemm.sdfg"
            #source = ".dacecache/affine_grid_2d/program.sdfg"
        else: 
            source  = "test_kernel/test_gemm_const.sdfg"
            postfix = "_const"
        sdfg_name = "gemm_" + args.opt + postfix
    elif args.kernel == "softmax":
        postfix = ""
        if args.sym:
            source = "test_kernel/test_softmax.sdfg"
        else:
            source = "test_kernel/test_softmax_const.sdfg"
            postfix = "_const"
        sdfg_name = "softmax_" + args.opt + postfix
    else:
        sys.exit("Invalid kernel")
    
    sdfg = dace.SDFG.from_file(source)
    sdfg.name = sdfg_name

    if args.opt == "tile":
        transformation = LoopTile()
        locations = transformation.find_tileable_maps(sdfg)
        if locations:
            loc = locations[0]
            print("tile location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied loop tile")

    if args.opt == "swap":
        transformation = SwapScope()
        locations = transformation.find_swappable_maps(sdfg)
        if locations:
            loc = locations[0]
            print("swap location", loc)
            transformation.apply(sdfg, loc)
            print("applied swap scope")

    if args.opt == "unroll":
        transformation = UnrollScope()
        locations = transformation.find_unrollable_maps(sdfg)
        if locations:
            loc = locations[0]
            print("unroll location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied unroll scope")

    if args.opt == "split":
        transformation = SplitMapScopes()
        locations = transformation.find_splitable_scopes(sdfg)
        if locations:
            loc = locations[0]
            print("split location", loc)
            transformation.apply4(sdfg, loc)
            print("applied split scope")
    
    if args.opt == "reuseArray":
        transformation = reuseArray()
        locations = transformation.find_reusable_arrays(sdfg)
        if locations:
            print("reuse array locations", locations)
            loc = locations[1]
            print("reuse array location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied reuse array")
    
    if args.opt == "addCopies":
        transformation = addArrayCopies()
        locations = transformation.find_copyable_arrays(sdfg)
        if locations:
            loc = "dst"
            print("add copies location", loc)
            transformation.apply(sdfg, loc)
            print("applied add copies")
    
    if args.opt == "removeCopies":
        transformation = addArrayCopies()
        loc = "dst"
        transformation.apply(sdfg, loc)
        transformation = removeArrayCopies()
        locations = transformation.find_deleteable_arrays(sdfg)
        print(f'applicable locations {locations}')
        if locations:
            loc = loc + "_copy_0"
            print("remove copies location", loc)
            transformation.apply(sdfg, loc)
            print("applied remove copies")

    if args.opt == "untile":
        transformation = LoopTile()
        locations = transformation.find_tiled_maps(sdfg)
        if locations:
            loc = locations[0]
            transformation.apply(sdfg, loc)
        transformation = LoopUntile()
        locations = transformation.find_untileable_maps(sdfg)
        if locations:
            loc = locations[0]
            print("untile location", loc)
            transformation.apply(sdfg, loc)
            print("applied untile scope")
        
    if args.opt == "reuseLoc":  
        transformation = reuseArrayLocations()
        locations = transformation.find_reusable_dimensions(sdfg)
        print(f'reusable dimensions {locations}')
        if locations:
            loc = locations[0]
            print("reuse location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied reuse location")

    if args.opt == "changeLayout":
        transformation = changeMemoryLayout()
        locations = transformation.find_changeable_arrays(sdfg)
        print(f'changeable arrays {locations}')
        if locations:
            loc = locations[0]
            print("change layout location", loc)
            transformation.apply(sdfg, loc, [1,0])
            print("applied change layout")
        
    if args.opt == "addDimension":
        transformation = addArrayDims()
        locations = transformation.find_createable_dim(sdfg)
        print(f'createable dimensions {locations}')
        if locations:
            loc = locations[0]
            print("add dimension location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied add dimension")
        
    if args.opt == "removeDimension":
        transformation = addArrayDims()
        locations = transformation.find_createable_dim(sdfg)
        if locations:
            loc = locations[0]
            transformation.apply(sdfg, loc[0], loc[1])
        transformation = removeArrayDims()
        locations = transformation.find_deleteable_dims(sdfg)
        print(f'deleteable dimensions {locations}')
        if locations:
            loc = locations[0]
            print("remove dimension location", loc)
            print(f'{loc[0]} {loc[1]} {loc[2]}')
            transformation.apply(sdfg, loc[0], loc[1], loc[2])
            print("applied remove dimension")
    
    if args.opt == "swapScopes":
        transformation = SwapMapScope()
        locations = transformation.find_swappable_maps(sdfg)
        if locations:
            loc = locations[0]
            print("swap operation location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied swap operation")
        
    if args.opt == "swapOp":
        transformation = SwapOpertions()
        locations = transformation.find_swappable_ops(sdfg)
        print(f'swappable operations {locations}')
        if locations:
            loc = locations[0]
            print("swap operation location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied swap operation")

    sdfg.view()
    csdfg = sdfg.compile()


    if args.validate:
        size = 32

        inputs = set()
        for state in sdfg.nodes():
            source = state.source_nodes()
            for src in source:
                if isinstance(src, dace.nodes.AccessNode):
                    if not 'copy' in src.data:
                        inputs.add(src)

        input_names = [node.data for node in inputs if not node.desc(sdfg).transient]
        outputs = state_0(sdfg).sink_nodes()
        output_names = [node.data for node in outputs]
        data = sdfg.arrays

        loop_index = {name: size for name, data in data.items() if isinstance(data, dace.data.Scalar) and not name in [n.data for n in state_0(sdfg).nodes() if isinstance(n, dace.nodes.AccessNode)]}
        #print(f'inputs {input_names}, outputs {output_names}, loop index {loop_index}')
        input_data = {}
        for output_name in output_names:
            if output_name in input_names:
                continue
            data_container = data[output_name]
            if isinstance(data_container, dace.data.Scalar):
                input_data[output_name] = np.zeros(1)
            else:
                shape = data_container.shape
                np_shape = []
                for dim in shape:
                    d = str(dim)
                    if d in loop_index:
                        np_shape.append(loop_index[d])
                input_data[output_name] = np.zeros(shape=np_shape)
          
        for input_name in input_names:
            data_container = data[input_name]
            if isinstance(data_container, dace.data.Scalar):
                input_data[input_name] = np.random.rand()
            else:
                shape = data_container.shape
                np_shape = []
                for dim in shape:
                    d = str(dim)
                    if d in loop_index:
                        np_shape.append(loop_index[d])
                input_data[input_name] = np.random.rand(*np_shape)


        expected = np.array([])
        output = np.array([])
        if args.kernel == "gemm":
            expected = input_data['alpha'] * input_data['A'] @ input_data['B'] + input_data['beta']*input_data['C']
            output = input_data['D']
        elif args.kernel == "softmax":
            argsrc = np.copy(input_data['src'])
            argsrc -= np.max(argsrc, axis=1, keepdims=True)
            np.exp(argsrc, out=argsrc)
            argsrc /= np.sum(argsrc, axis=1, keepdims=True)
            expected = argsrc
            output = input_data['dst']
        
        input_data.update(loop_index)
        try:
            csdfg(**input_data)
        except Exception as e:
            print(e)     

        diff = np.linalg.norm(output - expected)
        print("Difference:", diff)
        if diff >= 1e-4:
            print("Validation failed")
        else:
            print("Validation successful")
        print("==== Program end ====")
