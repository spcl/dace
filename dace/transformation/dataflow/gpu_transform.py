# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the GPU Transform Map transformation. """

from dace import data, dtypes, sdfg as sd
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import transformation, helpers
from dace.properties import Property, make_properties


@make_properties
class GPUTransformMap(transformation.SingleStateTransformation):
    """ Implements the GPUTransformMap transformation.

        Converts a single map to a GPU-scheduled map and creates GPU arrays
        outside it, generating CPU<->GPU memory copies automatically.
    """

    fullcopy = Property(desc="Copy whole arrays rather than used subset", dtype=bool, default=False)

    toplevel_trans = Property(desc="Make all GPU transients top-level", dtype=bool, default=False)

    register_trans = Property(desc="Make all transients inside GPU maps registers", dtype=bool, default=False)

    sequential_innermaps = Property(desc="Make all internal maps Sequential", dtype=bool, default=False)

    map_entry = transformation.PatternNode(nodes.MapEntry)

    import dace.libraries.standard as stdlib  # Avoid import loop
    reduce = transformation.PatternNode(stdlib.Reduce)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry), sdutil.node_path_graph(cls.reduce)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if expr_index == 0:
            map_entry = self.map_entry
            candidate_map = map_entry.map

            # Map schedules that are disallowed to transform to GPUs
            if (candidate_map.schedule in [dtypes.ScheduleType.MPI] + dtypes.GPU_SCHEDULES):
                return False
            if sd.is_devicelevel_gpu(sdfg, graph, map_entry):
                return False

            # Dynamic map ranges cannot become kernels
            if sd.has_dynamic_map_inputs(graph, map_entry):
                return False

            # Ensure that map does not include internal arrays that are
            # allocated on non-default space
            subgraph = graph.scope_subgraph(map_entry)
            for node in subgraph.nodes():
                if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).storage != dtypes.StorageType.Default
                        and node.desc(sdfg).storage != dtypes.StorageType.Register):
                    return False

            # If one of the outputs is a stream, do not match
            map_exit = graph.exit_node(map_entry)
            for edge in graph.out_edges(map_exit):
                dst = graph.memlet_path(edge)[-1].dst
                if (isinstance(dst, nodes.AccessNode) and isinstance(sdfg.arrays[dst.data], data.Stream)):
                    return False

            return True
        elif expr_index == 1:
            reduce = self.reduce

            # Disallow GPU transformation if already in device-level code
            if sd.is_devicelevel_gpu(sdfg, graph, reduce):
                return False

            return True

    def match_to_str(self, graph):
        if self.expr_index == 1:
            return str(self.reduce)
        else:
            return str(self.map_entry)

    def apply(self, graph: SDFGState, sdfg: SDFG):
        if self.expr_index == 0:
            map_entry = self.map_entry
            nsdfg_node = helpers.nest_state_subgraph(sdfg,
                                                     graph,
                                                     graph.scope_subgraph(map_entry),
                                                     full_data=self.fullcopy)
        else:
            cnode = self.reduce
            nsdfg_node = helpers.nest_state_subgraph(sdfg, graph, SubgraphView(graph, [cnode]), full_data=self.fullcopy)

        # Avoiding import loops
        from dace.transformation.interstate import GPUTransformSDFG
        transformation = GPUTransformSDFG()
        transformation.setup_match(sdfg, 0, -1, {}, 0)
        transformation.register_trans = self.register_trans
        transformation.sequential_innermaps = self.sequential_innermaps
        transformation.toplevel_trans = self.toplevel_trans

        transformation.apply(nsdfg_node.sdfg, nsdfg_node.sdfg)

        # Inline back as necessary
        sdfg.simplify()
