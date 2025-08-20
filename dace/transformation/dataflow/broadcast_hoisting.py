# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from dace import properties
from dace.transformation import transformation as xf
from dace.sdfg import SDFGState,SDFG, nodes, utils as sdutil
from dace.memlet import Memlet
from dace.sdfg.state import ControlFlowRegion
from dace.subsets import Range, Subset
from dace.symbolic import evaluate


@properties.make_properties
class BroadcastHoisting(xf.SingleStateTransformation):
    """
    Interchanges two consecutive maps, where one map broadcasts data.

    Example:
    ```
    for n in range(...):
      x[n] = y[n] # one to one
    for n in range(...):
      for k in range(...):
        z[n, k] = x[n] # broadcast
    ```
    becomes
    ```
    for n in range(...):
      for k in range(...):
        x[n, k] = y[n] # broadcast
    for n in range(...):
      for k in range(...):
        z[n, k] = x[n, k] # one to one
    ```

    """

    map_exit_1 = xf.PatternNode(nodes.MapExit)
    access_node = xf.PatternNode(nodes.AccessNode)
    map_entry_2 = xf.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_exit_1, cls.access_node, cls.map_entry_2)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        # Access node must be a transient array
        if not sdfg.arrays[self.access_node.data].transient:
            return False

        # Access node must participate in a broadcast in the second map
        # This the case if the read volume is higher than the write volume and both are not zero
        write_vol = 0
        for edge in graph.in_edges(self.access_node):
                write_vol += edge.data.volume

        for edge in graph.out_edges(self.access_node):
                read_vol = edge.data.volume

                sym_assignments = {}
                for sym in read_vol.free_symbols | write_vol.free_symbols:
                    sym_assignments[sym] = 10
                read_vol_eval = evaluate(read_vol, sym_assignments)
                write_vol_eval = evaluate(write_vol, sym_assignments)

                if read_vol_eval == 0 or write_vol_eval == 0:
                    return False
                if read_vol_eval <= write_vol_eval:
                    return False

        return True


    def apply(self, graph: SDFGState, sdfg: SDFG):
        # Collect all successor maps
        succ_maps = set()
        for node in graph.successors(self.access_node):
            if isinstance(node, nodes.MapEntry):
                succ_maps.add(node)

        min_lower_bound = None
        max_upper_bound = None
        for succ_map in succ_maps:
          # Get symbols used in the access node
          used_indices = set()
          for node in graph.all_nodes_between(succ_map, graph.exit_node(succ_map)):
              for edge in graph.in_edges(node):
                  if isinstance(edge.data, Memlet) and edge.data.data == self.access_node.data:
                      used_indices.update(edge.data.subset.free_symbols)
              for edge in graph.out_edges(node):
                  if isinstance(edge.data, Memlet) and edge.data.data == self.access_node.data:
                      used_indices.update(edge.data.subset.free_symbols)

          # Get the range and index name of the innermost map of the second map, which has parameters not already used by the access node
          innermost_map_2  = succ_map
          next_succ = list(graph.successors(innermost_map_2))[0]
          while isinstance(next_succ, nodes.MapEntry):
              innermost_map_2 = next_succ
              next_succ = list(graph.successors(innermost_map_2))[0]
          
          while isinstance(innermost_map_2, nodes.MapEntry) and len(set(innermost_map_2.map.params) - used_indices) == 0:
              innermost_map_2 = list(graph.predecessors(innermost_map_2))[0]

          map_range_2: Range = innermost_map_2.map.range
          ranges_2 = None
          idx_2 = None
          for i, idx in enumerate(innermost_map_2.map.params):
              if idx not in used_indices:
                  ranges_2 = map_range_2.ranges[i]
                  idx_2 = idx
                  break   

          # Add index to the second map
          handled_edges = set()
          for node in graph.all_nodes_between(innermost_map_2, graph.exit_node(innermost_map_2)):
              for edge in graph.in_edges(node):
                  if isinstance(edge.data, Memlet) and edge.data.data == self.access_node.data and idx_2 not in edge.data.free_symbols:
                      edge.data.subset = edge.data.subset + Range([(idx_2, idx_2, 1)])
                      handled_edges.add(edge)
              for edge in graph.out_edges(node):
                  if isinstance(edge.data, Memlet) and edge.data.data == self.access_node.data and idx_2 not in edge.data.free_symbols:
                      edge.data.subset = edge.data.subset + Range([(idx_2, idx_2, 1)])
                      handled_edges.add(edge)

          # store the bounds of the second map
          if min_lower_bound is None or ranges_2[0] < min_lower_bound:
              min_lower_bound = ranges_2[0]
          if max_upper_bound is None or ranges_2[1] > max_upper_bound:
              max_upper_bound = ranges_2[1]
                    
        # Add a new range to the first map
        innermost_map_1 = self.map_exit_1
        while isinstance(list(graph.predecessors(innermost_map_1))[0], nodes.MapExit):
            innermost_map_1 = list(graph.predecessors(innermost_map_1))[0]
        idx_1 = sdfg.find_new_symbol(idx_2)
        innermost_map_1.map.params.append(idx_1)
        innermost_map_1.map.range += Range([(min_lower_bound, max_upper_bound, 1)])

        for node in graph.all_nodes_between(graph.entry_node(innermost_map_1), innermost_map_1):
            for edge in graph.in_edges(node):
                if isinstance(edge.data, Memlet) and edge.data.data == self.access_node.data:
                    edge.data.subset = edge.data.subset + Range([(idx_1, idx_1, 1)])
                    handled_edges.add(edge)
            for edge in graph.out_edges(node):
                if isinstance(edge.data, Memlet) and edge.data.data == self.access_node.data:
                    edge.data.subset = edge.data.subset + Range([(idx_1, idx_1, 1)])
                    handled_edges.add(edge)
        
                     # Add a new dimension to the access node
        old_shape = sdfg.arrays[self.access_node.data].shape
        new_shape = old_shape + (max_upper_bound - min_lower_bound + 1, )
        sdfg.arrays[self.access_node.data].set_shape(new_shape)

        # Propagate memlets
        sdutil.propagation.propagate_memlets_sdfg(sdfg)

