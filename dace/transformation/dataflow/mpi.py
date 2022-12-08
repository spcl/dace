# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the MPITransformMap transformation. """

from dace import dtypes
from dace.sdfg import has_dynamic_map_inputs
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class MPITransformMap(transformation.SingleStateTransformation):
    """ Implements the MPI parallelization pattern.

        Takes a map and makes it an MPI-scheduled map, introduces transients
        that keep locally accessed data.
        
        Original SDFG:

        .. code-block:: text
        
            Input1 -                                            Output1
                    \                                          /
            Input2 --- MapEntry -- Arbitrary R  -- MapExit -- Output2
                    /                                          \ 
            InputN -                                            OutputN


        Nothing in R may access other inputs/outputs that are not defined in R
        itself and do not go through MapEntry/MapExit
        Map must be a one-dimensional map for now.
        The range of the map must be a Range object.
     
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

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        # Check if the map is one-dimensional
        if map_entry.map.range.dims() != 1:
            return False

        # We cannot transform a map which is already of schedule type MPI
        if map_entry.map.schedule == dtypes.ScheduleType.MPI:
            return False

        # We cannot transform a map which is already inside a MPI map, or in
        # another device
        schedule_whitelist = [dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential]
        sdict = graph.scope_dict()
        parent = sdict[map_entry]
        while parent is not None:
            if parent.map.schedule not in schedule_whitelist:
                return False
            parent = sdict[parent]

        # Dynamic map ranges not supported (will allocate dynamic memory)
        if has_dynamic_map_inputs(graph, map_entry):
            return False

        # MPI schedules currently do not support WCR
        map_exit = graph.exit_node(map_entry)
        if any(e.data.wcr for e in graph.out_edges(map_exit)):
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry

        # Avoiding import loops
        from dace.transformation.dataflow.strip_mining import StripMining
        from dace.transformation.dataflow.local_storage import InLocalStorage, OutLocalStorage, LocalStorage

        rangeexpr = str(map_entry.map.range.num_elements())

        stripmine_subgraph = {StripMining.map_entry: self.subgraph[MPITransformMap.map_entry]}
        sdfg_id = sdfg.sdfg_id
        stripmine = StripMining()
        stripmine.setup_match(sdfg, sdfg_id, self.state_id, stripmine_subgraph, self.expr_index)
        stripmine.dim_idx = -1
        stripmine.new_dim_prefix = "mpi"
        stripmine.tile_size = "(" + rangeexpr + "/__dace_comm_size)"
        stripmine.divides_evenly = True
        stripmine.apply(graph, sdfg)

        # Find all in-edges that lead to the map entry
        outer_map = None
        edges = [e for e in graph.in_edges(map_entry) if isinstance(e.src, nodes.EntryNode)]

        outer_map = edges[0].src

        # Add MPI schedule attribute to outer map
        outer_map.map._schedule = dtypes.ScheduleType.MPI

        # Now create a transient for each array
        for e in edges:
            if e.data.is_empty():
                continue
            in_local_storage_subgraph = {
                LocalStorage.node_a: graph.node_id(outer_map),
                LocalStorage.node_b: self.subgraph[MPITransformMap.map_entry]
            }
            sdfg_id = sdfg.sdfg_id
            in_local_storage = InLocalStorage()
            in_local_storage.setup_match(sdfg, sdfg_id, self.state_id, in_local_storage_subgraph, self.expr_index)
            in_local_storage.array = e.data.data
            in_local_storage.apply(graph, sdfg)

        # Transform OutLocalStorage for each output of the MPI map
        in_map_exit = graph.exit_node(map_entry)
        out_map_exit = graph.exit_node(outer_map)

        for e in graph.out_edges(out_map_exit):
            if e.data.is_empty():
                continue
            name = e.data.data
            outlocalstorage_subgraph = {
                LocalStorage.node_a: graph.node_id(in_map_exit),
                LocalStorage.node_b: graph.node_id(out_map_exit)
            }
            sdfg_id = sdfg.sdfg_id
            outlocalstorage = OutLocalStorage()
            outlocalstorage.setup_match(sdfg, sdfg_id, self.state_id, outlocalstorage_subgraph, self.expr_index)
            outlocalstorage.array = name
            outlocalstorage.apply(graph, sdfg)
