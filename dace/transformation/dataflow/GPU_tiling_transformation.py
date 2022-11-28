from operator import is_
import dace
import cupy
from dace.dtypes import ScheduleType
from dace.symbolic import int_ceil
import numpy as np

from dace.properties import make_properties, Property
from dace.sdfg import nodes, SDFG, SDFGState
from dace.transformation import transformation as xf
from dace.sdfg import utils as sdutil
from dace import subsets


@make_properties
class GPU_Tiling(xf. SingleStateTransformation):
    
    map_entry = xf.PatternNode(dace.nodes.MapEntry)

    # reduction_index = Property(type=int, default=0, desc="Index of the reduced dimension")
    
    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]
    
    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        
        # for now we only support tiling for up to two parameters!
        if (len(self.map_entry.params) > 2):
            return False
        else:
            return True
    
    def apply(self, graph: subsets.Union[SDFG, SDFGState], sdfg: SDFG):
        # we expand it to i,j, 'old_para', where i is DeviceMapped, j is the block and old_para is anything that is left over
        
        # remember to connect all the edges... or can you just change the range of old_para?
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        # red_idx = self.reduction_index
        is_unidim = (len(map_entry.params) == 1)
        if is_unidim:
            old_para = map_entry.params[0]
            para_N = map_entry.range[0][1]
        else:
            old_para = map_entry.params[-1]
            para_N = map_entry.range[-1][1]
            other_N = map_entry.range[0][1]
        
        #NOTE: old_para should not be __i or __j
        
        # NOTE: I am going to assume BlockDim, GridDim and MaxTs are symbols in the SDFG!
        # NOTE: how do I make sure I don't end up with aliasing?
        
        
        default_symbols = {'WarpSize': 32, 'BlockDim': 256, 'GridDim': 2048}
        symbols = {}
        
        for symbol in default_symbols:
            if symbol not in sdfg.symbols:
                symbols[symbol] = default_symbols[symbol]
                sdfg.add_symbol(symbol, dace.int32)
        

        for s in ('BlockDim', 'GridDim'):
            try:
                sdfg.add_symbol(s, dace.int32)
            except FileExistsError:
                pass

        new_i = '__i' + old_para if old_para == '__i' else '__i'
        new_j = '__j' + old_para if old_para == '__j' else '__j'
        
        current_map = map_entry.map
        if is_unidim:
            current_map.range = subsets.Range([(new_i +'*BlockDim+' + new_j, str(para_N), 'BlockDim * GridDim')])
        else:
            current_map.range = subsets.Range([(new_j, str(para_N), 'BlockDim')])
            # This is for the non-reduced dimensions/modes
            new_k = current_map.params.pop(0)
        current_map.schedule = dace.ScheduleType.Sequential
        
        para_N += 1
        
        if is_unidim:
            i_map = nodes.Map(label ='GPU_map_i',
                            ndrange = subsets.Range([('0', 'Min(GridDim, int_ceil(' + str(para_N) +',BlockDim))-1', '1')]),
                            params=[new_i],
                            schedule= dace.dtypes.ScheduleType.GPU_Device)
            
            j_map = nodes.Map(label= 'Block_map_j',
                            ndrange= subsets.Range([('0', 'BlockDim-1' , '1')]),
                            params=[new_j],
                            schedule= dace.dtypes.ScheduleType.GPU_ThreadBlock)
        else:
            i_map = nodes.Map(label ='GPU_map_i',
                            ndrange = subsets.Range([('0', 'Min(GridDim, ' + str(other_N) + ' + 1)-1', '1')]),
                            params=[new_i],
                            schedule= dace.dtypes.ScheduleType.GPU_Device)
            
            j_map = nodes.Map(label= 'Block_map_j',
                            ndrange= subsets.Range([('0', 'BlockDim-1' , '1')]),
                            params=[new_j],
                            schedule= dace.dtypes.ScheduleType.GPU_ThreadBlock)

        i_entry = nodes.MapEntry(i_map)
        j_entry = nodes.MapEntry(j_map)
        
        i_exit = nodes.MapExit(i_map)
        j_exit = nodes.MapExit(j_map)
    
        if is_unidim:
        
            for edge in graph.out_edges(map_entry):
                src = graph.memlet_path(edge)[0].src
                src_conn = graph.memlet_path(edge)[0].src_conn
                graph.add_memlet_path(
                                src,
                                i_entry,
                                j_entry,
                                map_entry,
                                edge.dst,
                                memlet=edge.data,
                                src_conn=src_conn,
                                dst_conn=edge.dst_conn)
                graph.remove_memlet_path(edge)
            
            for edge in graph.in_edges(map_exit):
                dst = graph.memlet_path(edge)[-1].dst
                dst_conn = graph.memlet_path(edge)[-1].dst_conn
                graph.add_memlet_path(
                                    edge.src,
                                    map_exit,
                                    j_exit,
                                    i_exit,
                                    dst,                                      
                                    src_conn=edge.src_conn,
                                    memlet= edge.data,
                                    dst_conn=dst_conn)
                graph.remove_memlet_path(edge)

        else:

            k_map = nodes.Map(label= 'nonreduce_map_k',
                              ndrange= subsets.Range([(new_i, str(other_N) , 'GridDim')]),
                              params=[new_k],
                              schedule=dace.dtypes.ScheduleType.Sequential)
            k_entry = nodes.MapEntry(k_map)
            k_exit = nodes.MapExit(k_map)
            
            for edge in graph.out_edges(map_entry):
                src = graph.memlet_path(edge)[0].src
                src_conn = graph.memlet_path(edge)[0].src_conn
                graph.add_memlet_path(
                                src,
                                i_entry,
                                j_entry,
                                k_entry,
                                map_entry,
                                edge.dst,
                                memlet=edge.data,
                                src_conn=src_conn,
                                dst_conn=edge.dst_conn)
                graph.remove_memlet_path(edge)
            
            for edge in graph.in_edges(map_exit):
                dst = graph.memlet_path(edge)[-1].dst
                dst_conn = graph.memlet_path(edge)[-1].dst_conn
                graph.add_memlet_path(
                                    edge.src,
                                    map_exit,
                                    k_exit,
                                    j_exit,
                                    i_exit,
                                    dst,                                      
                                    src_conn=edge.src_conn,
                                    memlet= edge.data,
                                    dst_conn=dst_conn)
                graph.remove_memlet_path(edge)

            return [map_entry] + [i_entry, j_entry, k_entry]
            
        
        return [map_entry] + [i_entry, j_entry]