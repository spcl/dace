import dace
from dace import data, dtypes, nodes, properties
from dace.libraries.standard import Gearbox
from dace.memlet import Memlet
from dace.sdfg.graph import SubgraphView
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.transformation.subgraph import helpers

@properties.make_properties
class TemporalVectorization(transformation.SubgraphTransformation):
    '''
    This transformation applies the multi-pumping optimization to a subgraph targeting FPGAs, in turn packing more computations temporally rather than spatially as is done in traditional vectorization.

    Currently, it can only be applied to applications targeting Xilinx FPGAs, and to subgraphs that purely communicate through streams. It can be applied in two ways:
    1. Where the widths of the internal paths remain unchanged while the external paths are widened by the multi-pumping factor. This gives the benefit of increased throughput at the same critical-resource footprint.
    2. Where the widths of the internal paths are divided by the multi-pumping factor, while the widths of the external paths remain unchanged. This gives the benefit of a reduced critical-resource footprint at the same throughput. 
    '''
    # TODO 3rd approach: where the subgraph is just clocked faster without introducing gearboxing. This could help designs where the subgraph reaches II=2, then by clocking it two times faster, it essentially behaves as II=1 from the slow clock domain. 
    factor = properties.Property(dtype=int, default=2, desc='The multi-pumping factor. E.g. double-pumping is a factor of 2.')
    approach = properties.Property(dtype=int, default=2, desc='Which approach to use. Can be 1 (increased throughput, same resources) or 2 (same throughput, reduced resourced).')

    def can_be_applied(self, sdfg: SDFG, subgraph: SubgraphView) -> bool:
        '''
        Temporal vectorization can be applied if:
        1. There is one outermost map in the subgraph. 
        2. All of the non- source and sink nodes are only accessed within this subgraph. 
        3. All of the source and sink nodes are either streams or scalars. 
        4. If the direction is 1, then either:
            - The elemental type of the streams must be a vector type.
            - The elemental type must be convertible to a vector type. 
            - Data packers/issuers are allowed to be inserted at the cost of performance through additional data plumbing overhead.
        5. If the direction is 2, all the elemental types of the streams must be a vector type that is integer divisable by the multi-pumping factor.
        '''
        # Extract all of the relevant components of the subgraph
        graph = subgraph.graph
        src_nodes = subgraph.source_nodes()
        dst_nodes = subgraph.sink_nodes()
        srcdst_nodes = src_nodes + dst_nodes
        srcdst_arrays = [sdfg.arrays[node.data] for node in srcdst_nodes]
        access_nodes = [node for node in subgraph.nodes() if isinstance(node, nodes.AccessNode) and not node in srcdst_nodes]
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]
        maps = [map_entry.map for map_entry in map_entries]
        
        # Perform checks
        # 1. There is at least one map.
        if len(maps) < 1: return False

        # TODO array of streams
        # TODO map of computation (matmul)
        # TODO scalars

        # 2. All of the non- source and sink nodes only resides within this subgraph.
        for sg in dace.sdfg.concurrent_subgraphs(graph):
            if sg == subgraph: continue
            for nd in sg.nodes():
                if isinstance(nd, nodes.AccessNode) and nd in access_nodes:
                    return False

        # 3. All of the source and sink nodes only are either streams or scalars.
        for arr in srcdst_arrays:
            if not (isinstance(arr, data.Stream) or isinstance(arr, data.Scalar)):
                return False

        # 4. If the approach is 1, then either the dataype must be a vector type or must be convertible to a vector type.
        if self.approach == 1:
            # TODO not implemented yet.
            return False

        # 5. If the approach is 2, then all the elemental datatype of the streams must be a vector type.
        elif self.approach == 2:
            for arr in srcdst_arrays:
                if (isinstance(arr, data.Stream) and not isinstance(arr.dtype, dtypes.vector)) or arr.veclen % self.factor != 0:
                    return False
        
        # If the approach is wrong, then it should not be applied.
        else:
            return False

        return True
    
    def issuer(self, sdfg: SDFG, state: SDFGState, subgraph: SubgraphView, src):
        arr = sdfg.arrays[src.data]
        veclen = arr.dtype.veclen // self.factor
        dtype = dace.vector(arr.dtype.base_type, veclen)
        name = f'{src.data}_pumped'
        new_src = sdfg.add_stream(name, dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        
        # Update the subgraph
        old_edge = subgraph.out_edges(src)[0]
        old_path = state.memlet_path(old_edge)
        for edge in old_path[1:]:
            edge.data = dace.Memlet(f'{name}[0]')
        old_path[-1].dst.in_connectors[old_path[-1].dst_conn] = dtype
        state.remove_edge(old_edge)
        new_src = state.add_read(name)
        state.add_edge(new_src, old_edge.src_conn, old_edge.dst, old_edge.dst_conn, memlet=dace.Memlet(f'{name}[0]'))
        innermost_map = [n.src.map for n in old_path if isinstance(n.src, nodes.MapEntry)][-1]

        # Insert gearboxing for converting stream widths
        gearbox = Gearbox(innermost_map.range.ranges[0][1]+1, schedule=dtypes.ScheduleType.FPGA_Multi_Pumped)
        gearbox_src = state.add_write(name)
        state.add_memlet_path(src, gearbox, dst_conn='from_memory', memlet=Memlet(f'{src.data}[0]'))
        state.add_memlet_path(gearbox, gearbox_src, src_conn='to_kernel', memlet=Memlet(f'{name}[0]'))
        return innermost_map

    def packer(self, sdfg: SDFG, state: SDFGState, subgraph: SubgraphView, dst):
        arr = sdfg.arrays[dst.data]
        veclen = arr.dtype.veclen // self.factor
        dtype = dace.vector(arr.dtype.base_type, veclen)
        name = f'{dst.data}_pumped'
        sdfg.add_stream(name, dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        
        # Update the subgraph
        old_edge = subgraph.in_edges(dst)[0]
        old_path = state.memlet_path(old_edge)
        for edge in old_path[:-1]:
            edge.data = dace.Memlet(f'{name}[0]')
        old_path[0].src.out_connectors[old_path[0].src_conn] = dtype
        state.remove_edge(old_edge)
        new_dst = state.add_write(name)
        state.add_edge(old_edge.src, old_edge.src_conn, new_dst, old_edge.dst_conn, memlet=dace.Memlet(f'{name}[0]'))
        innermost_map = [n.dst.map for n in old_path if isinstance(n.dst, nodes.MapExit)][0]
        
        # Insert gearbox for converting stream widths.
        gearbox = Gearbox(innermost_map.range.ranges[0][1]+1, schedule=dtypes.ScheduleType.FPGA_Multi_Pumped)
        gearbox_dst = state.add_read(name)
        state.add_memlet_path(gearbox_dst, gearbox, dst_conn='from_memory', memlet=Memlet(f'{name}[0]'))
        state.add_memlet_path(gearbox, dst, src_conn='to_kernel', memlet=Memlet(f'{dst.data}[0]'))
        return innermost_map
    
    def apply(self, sdfg: SDFG, **kwargs):
        # Get the graphs and the nodes
        subgraph = self.subgraph_view(sdfg)
        graph = subgraph.graph
        src_nodes = subgraph.source_nodes()
        dst_nodes = subgraph.sink_nodes()
        affected_maps = set()
        
        # Update all of the subgraph inputs
        for src in src_nodes:
            affected_maps.add(self.issuer(sdfg, graph, subgraph, src))
        
        # Update all of the subgraph outputs
        for dst in dst_nodes:
            affected_maps.add(self.packer(sdfg, graph, subgraph, dst))

        # Update the schedules of the innermost affected maps.
        for map in affected_maps:
            rng = list(map.range.ranges[0])
            rng[1] = ((rng[1] + 1) * 2) - 1
            map.range.ranges[0] = rng
            map.schedule = dtypes.ScheduleType.FPGA_Multi_Pumped