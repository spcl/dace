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
    1. Inwards - where the internal widths of the subgraph are narrowed. This gives the benefit of a reduced critical-resource footprint at the same throughput.
    2. Outwards - where the external widths of the streams are widened. This gives the benefit of increased throughput at the same critical-resource footprint. 
    '''
    # TODO currently, it is limited to streams. Should be possible to extend this to memory-mapped as well in the future, at least for Xilinx using their IP cores. 
    factor = properties.Property(dtype=int, default=2, desc='The multi-pumping factor. E.g. double-pumping is a factor of 2.')
    inwards = properties.Property(dtype=bool, default=True, desc='Flag for whether the optimization should be applied inwards (reduced resources, locked throughput).')


    def can_be_applied(self, sdfg: SDFG, subgraph: SubgraphView) -> bool:
        '''
        Temporal vectorization can be applied if:
        1. There is one outermost map in the subgraph. 
        2. All of the non- source and sink nodes are only accessed within this subgraph. 
        3. All of the source and sink nodes are either streams or scalars. 
        4. If the direction is inwards, all the elemental types of the streams must be a vector type that is integer divisable by the multi-pumping factor.
        5. If the direction is outwards, then either:
            - The elemental type of the streams must be a vector type.
            - The elemental type must be convertible to a vector type. 
            - Data packers/issuers are allowed to be inserted at the cost of performance through additional data plumbing overhead.
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

        # TODO array of streams!
        # TODO map of computation?? (matmul)

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

        # 4. If the direction is inwards, then all the elemental datatype of the streams must be a vector type.
        # TODO should the division by factor be handled here, or in code generation?
        # TODO check that there only exists one edge per stream. I think this is a general requirement, but it should be checked here as part of the feasibility check. 
        # Except for array of streams of course. Actually, couldn't that be its own transformation? Extract the streams from the array that are connected elsewhere into their own access nodes? It would make code generation much easier at least. 
        # Could be here. If symbolic, then pass on to codegen. If not, then check if it is divisible by the factor.  
        if self.inwards:
            for arr in srcdst_arrays:
                if (isinstance(arr, data.Stream) and not isinstance(arr.dtype, dtypes.vector)) or arr.veclen % self.factor != 0:
                    return False
                    

        # 5. If the direction is outwards, then either the dataype must be a vector type, must be convertible to a vector type, or data packers/issuers must be inserted. TODO maybe they should be optimized away? The idea is that if the analysis becomes too hard (i.e. they are not convertible), then we could insert the packers / issuers at the cost of degraded performance / plumbing overhead, since data would then be packed, unpacked, computed, packed, unpacked. Hmm, but that wouldn't work, would it, since the initial packing is now handled in the slow clock region, essentially halving the II. One would still get the benefit of half resources, so I guess that it could make sense for certain applications.
        else:
            # TODO not implemented yet.
            return False

        return True
    
    def create_intermediate_view(self, sdfg, state, data):
        arr = sdfg.arrays[data]
        veclen = arr.dtype.veclen // self.factor
        dtype = dace.vector(arr.dtype.base_type, veclen)
        name = f'{data}_pumped'
        sdfg.add_view(name, arr.shape, dtype)
        vw = state.add_access(name)
        return vw
    
    def issuer(self, sdfg: SDFG, state: SDFGState, subgraph: SubgraphView, map, src):
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
        state.remove_edge(old_edge)
        new_src = state.add_read(name)
        state.add_edge(new_src, old_edge.src_conn, old_edge.dst, old_edge.dst_conn, memlet=dace.Memlet(f'{name}[0]'))

        gearbox = Gearbox(map.range.ranges[0][1]+1, schedule=dace.ScheduleType.FPGA_Device)
        gearbox_src = state.add_write(name)
        state.add_memlet_path(src, gearbox, dst_conn='from_memory', memlet=Memlet(f'{src.data}[0]'))
        state.add_memlet_path(gearbox, gearbox_src, src_conn='to_kernel', memlet=Memlet(f'{name}[0]'))
        
        return

        # Build the new subgraph
        tasklet = state.add_tasklet(f'issue_{src.data}', {'inp'}, {'out'}, '''
tmp = inp
out.push(inp[0])
out.push(inp[1])
        ''')
        m_entry, m_exit = state.add_map(f'issue_{src.data}_map', {'i': map.range.ranges[0]}, dace.ScheduleType.FPGA_Device)
        new_src = state.add_write(name)
        state.add_memlet_path(src, m_entry, tasklet, memlet=old_edge.data, src_conn=old_edge.src_conn, dst_conn='inp')
        state.add_memlet_path(tasklet, m_exit, new_src, memlet=dace.Memlet(f'{name}[0]'), src_conn='out', dst_conn=old_edge.dst_conn)

    def packer(self, sdfg: SDFG, state: SDFGState, subgraph: SubgraphView, map, dst):
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
        state.remove_edge(old_edge)
        new_dst = state.add_write(name)
        state.add_edge(old_edge.src, old_edge.src_conn, new_dst, old_edge.dst_conn, memlet=dace.Memlet(f'{name}[0]'))
        
        gearbox = Gearbox(map.range.ranges[0][1]+1, schedule=dace.ScheduleType.FPGA_Device)
        gearbox_dst = state.add_read(name)
        state.add_memlet_path(gearbox_dst, gearbox, dst_conn='from_memory', memlet=Memlet(f'{name}[0]'))
        state.add_memlet_path(gearbox, dst, src_conn='to_kernel', memlet=Memlet(f'{dst.data}[0]'))
        
        return

        # Build the new subgraph
        # me0 > me1 > tasklet0 > mx1 > tmp > tasklet1 > mx0
        m_outer_entry, m_outer_exit = state.add_map(f'pack_{dst.data}_map_outer', {'i': map.range.ranges[0]}, dace.ScheduleType.FPGA_Device)
        m_inner_entry, m_inner_exit = state.add_map(f'pack_{dst.data}_map_inner', {'j': f'0:{self.factor}'}, dace.ScheduleType.FPGA_Device)
        
        tasklet_pack = state.add_tasklet(f'pack_{dst.data}_pack', {'inp'}, {'out'}, '''
out = inp
        ''')
        tmp_name = f'pack_{dst.data}_tmp'
        sdfg.add_array(tmp_name, [self.factor], dtype=arr.dtype.base_type, storage=dtypes.StorageType.FPGA_Registers)
        tmp = state.add_access(tmp_name)
        tasklet_send = state.add_tasklet(f'pack_{dst.data}_send', {'inp'}, {'out'}, '''
out = inp
        ''')
        new_dst = state.add_read(name)
        state.add_memlet_path(dst, m_outer_entry, m_inner_entry, tasklet_pack, memlet=old_edge.data, dst_conn='inp')
        state.add_memlet_path(tasklet_pack, m_inner_exit, tmp, memlet=dace.Memlet(f'{tmp_name}[j]'), src_conn='out')
        state.add_edge(tmp, None, tasklet_send, 'inp', memlet=dace.Memlet(f'{tmp_name}[0:{self.factor}]'))
        state.add_memlet_path(tasklet_send, m_outer_exit, new_dst, memlet=dace.Memlet(f'{name}[0]'), src_conn='out') 
    
    def apply(self, sdfg: SDFG, **kwargs):
        # TODO Only the inwards direction is described here. 
        # TODO Check for scalars
        subgraph = self.subgraph_view(sdfg)
        graph = subgraph.graph
        src_nodes = subgraph.source_nodes()
        dst_nodes = subgraph.sink_nodes()
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]
        maps = [map_entry.map for map_entry in map_entries]
        
        for src in src_nodes:
            self.issuer(sdfg, graph, subgraph, maps[0], src)
        
        for dst in dst_nodes:
            self.packer(sdfg, graph, subgraph, maps[0], dst)
        
        # Insert views after each source node.
        #for src in []:#src_nodes:
        #    vw = self.create_intermediate_view(sdfg, graph, src.data)
        #    # Reconnect the old memlet to the new view
        #    for edge in subgraph.out_edges(src):
        #        graph.remove_edge(edge)
        #        graph.add_edge(src, edge.src_conn, vw, edge.dst_conn, edge.data)
        #        graph.add_edge(vw, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
        
        # Insert views before each sink node
        #for dst in []:#dst_nodes:
        #    vw = self.create_intermediate_view(sdfg, graph, dst.data)
        #    for edge in subgraph.in_edges(dst):
        #        graph.remove_edge(edge)
        #        graph.add_edge(edge.src, edge.src_conn, vw, edge.dst_conn, edge.data)
        #        graph.add_edge(vw, edge.src_conn, dst, edge.dst_conn, edge.data)

        # Update the schedule of the map.
        # TODO gør noget smartere. Men hvertfald, burde virke for vadd
        # TODO maybe it should just multiply all the maps that touches either of the src/dst nodes? Both through direct and through memlets?
        rng = list(maps[0].range.ranges[0])
        rng[1] = ((rng[1] + 1) * 2) - 1
        maps[0].range.ranges[0] = rng
        aoeu = 42