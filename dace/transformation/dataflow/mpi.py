""" Contains the MPITransformMap transformation. """

from dace import types, symbolic
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from dace.properties import make_properties


@make_properties
class MPITransformMap(pattern_matching.Transformation):
    """ Implements the MPI parallelization pattern.

        Takes a map and makes it an MPI-scheduled map, introduces transients
        that keep locally accessed data.
        
         Original SDFG
         =============
         ```
         Input1 -                                            Output1
                 \                                          /
         Input2 --- MapEntry -- Arbitrary R  -- MapExit -- Output2
                 /                                          \
         InputN -                                            OutputN
         ```

         Nothing in R may access other inputs/outputs that are not defined in R
         itself and do not go through MapEntry/MapExit
         Map must be a one-dimensional map for now.
         The range of the map must be a Range object.

         Output:
         =======
        
         * Add transients for the accessed parts
         * The schedule property of Map is set to MPI
         * The range of Map is changed to
           var = startexpr + p * chunksize ... startexpr + p + 1 * chunksize
           where p is the current rank and P is the total number of ranks,
           and chunksize is defined as (endexpr - startexpr) / P, adding the 
           remaining K iterations to the first K procs.
         * For each input InputI, create a new transient transInputI, which 
           has an attribute that specifies that it needs to be filled with 
           (possibly) remote data
         * Collect all accesses to InputI within R, assume their convex hull is
           InputI[rs ... re]
         * The transInputI transient will contain InputI[rs ... re]
         * Change all accesses to InputI within R to accesses to transInputI
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(MPITransformMap._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[MPITransformMap._map_entry]]

        # Check if the map is one-dimensional
        if map_entry.map.range.dims() != 1:
            return False

        # We cannot transform a map which is already of schedule type MPI
        if map_entry.map.schedule == types.ScheduleType.MPI:
            return False

        # We cannot transform a map which is already inside a MPI map, or in
        # another device
        schedule_whitelist = [
            types.ScheduleType.Default, types.ScheduleType.Sequential
        ]
        sdict = graph.scope_dict()
        parent = sdict[map_entry]
        while parent is not None:
            if parent.map.schedule not in schedule_whitelist:
                return False
            parent = sdict[parent]

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[MPITransformMap._map_entry]]

        return map_entry.map.label

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]

        map_entry = graph.nodes()[self.subgraph[MPITransformMap._map_entry]]

        # Avoiding import loops
        from dace.transformation.dataflow import StripMining
        from dace.transformation.dataflow.stream_transient import (
            InLocalStorage)
        from dace.transformation.dataflow.stream_transient import (
            OutLocalStorage)
        from dace.graph import labeling

        rangeexpr = str(map_entry.map.range.num_elements())

        stripmine_subgraph = {
            StripMining._map_entry: self.subgraph[MPITransformMap._map_entry]
        }
        sdfg_id = sdfg.sdfg_list.index(sdfg)
        stripmine = StripMining(sdfg_id, self.state_id, stripmine_subgraph,
                                self.expr_index)
        stripmine.dim_idx = -1
        stripmine.new_dim_prefix = "mpi"
        stripmine.tile_size = "(" + rangeexpr + "/__dace_comm_size)"
        stripmine.divides_evenly = True
        stripmine.apply(sdfg)

        # Find all in-edges that lead to candidate[MPITransformMap._map_entry]
        outer_map = None
        edges = [
            e for e in graph.in_edges(map_entry)
            if isinstance(e.src, nodes.EntryNode)
        ]

        outer_map = edges[0].src

        # We need a tasklet for InLocalStorage
        tasklet = None
        for e in graph.out_edges(map_entry):
            if isinstance(e.dst, nodes.CodeNode):
                tasklet = e.dst
                break

        if tasklet is None:
            raise ValueError("Tasklet not found")

        # Add MPI schedule attribute to outer map
        outer_map.map._schedule = types.ScheduleType.MPI

        # Now create a transient for each array
        for e in edges:
            in_local_storage_subgraph = {
                InLocalStorage._outer_map_entry:
                graph.node_id(outer_map),
                InLocalStorage._inner_map_entry:
                self.subgraph[MPITransformMap._map_entry]
            }
            sdfg_id = sdfg.sdfg_list.index(sdfg)
            in_local_storage = InLocalStorage(sdfg_id, self.state_id,
                                              in_local_storage_subgraph,
                                              self.expr_index)
            in_local_storage.array = e.data.data
            in_local_storage.apply(sdfg)

        # Transform OutLocalStorage for each output of the MPI map
        in_map_exits = graph.exit_nodes(map_entry)
        out_map_exits = graph.exit_nodes(outer_map)
        in_map_exit = in_map_exits[0]
        out_map_exit = out_map_exits[0]

        for e in graph.out_edges(out_map_exit):
            name = e.data.data
            outlocalstorage_subgraph = {
                OutLocalStorage._inner_map_exit: graph.node_id(in_map_exit),
                OutLocalStorage._outer_map_exit: graph.node_id(out_map_exit)
            }
            sdfg_id = sdfg.sdfg_list.index(sdfg)
            outlocalstorage = OutLocalStorage(sdfg_id, self.state_id,
                                              outlocalstorage_subgraph,
                                              self.expr_index)
            outlocalstorage.array = name
            outlocalstorage.apply(sdfg)

        return


pattern_matching.Transformation.register_pattern(MPITransformMap)
