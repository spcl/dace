import argparse
from dace.sdfg import nodes as nd
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg import infer_types
from dace import SDFG, SDFGState
from typing import Dict
import os
import sympy as sp
from copy import deepcopy
from dace.symbolic import pystr_to_symbolic
from dace.dtypes import StorageType

from dace.transformation.passes.symbol_ssa import StrictSymbolSSA
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.analysis import loop_analysis

from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion, ConditionalBlock

def calculate_edge_volume(state: SDFGState, edge:MultiConnectorEdge):
    vol = sp.sympify(1)
    if not edge.data.volume == 0:
        # if propagate memlets was able to assign a volume to the edge, we use 
        vol = edge.data.volume*state.sdfg.arrays[edge.data.data].dtype.bytes
    else:
        # else we approximate the volume by the accessed range
        vol = sp.sympify(1)
        for (l, u, s) in edge.data.subset.ranges:
            vol *= (u-l)/s    
    return vol

def scope_volume(state: SDFGState, entry=None, access_node_volume_map:dict[nd.AccessNode, (sp.Expr, sp.Expr)]={}, region_volume_map:dict[AbstractControlFlowRegion,tuple:[sp.Expr, sp.Expr, sp.Expr, sp.Expr]]={}, range_var_stack:list[str, tuple]=[]):
    scope_nodes = state.scope_children()[entry]
    min_read = sp.sympify(0)
    min_write = sp.sympify(0)
    max_read = sp.sympify(0)
    max_write = sp.sympify(0)
    for node in scope_nodes:        
        if isinstance(node, nd.AccessNode):
            read_edge_volumes = []
        
            for edge in state.out_edges(node):
                if isinstance(edge.dst, nd.NestedSDFG):
                    continue
                if state.sdfg.arrays[node.data].storage is StorageType.CPU_Heap or state.sdfg.arrays[node.data].storage is StorageType.GPU_Global:
                    edge_vol = calculate_edge_volume(state, edge)
                    read_edge_volumes.append(edge_vol)
            

            access_node_read_volume = sp.sympify(sum(read_edge_volumes))

            write_edge_volumes = []
            
            for edge in state.in_edges(node):
                if isinstance(edge.src, nd.NestedSDFG):
                    continue
                if state.sdfg.arrays[node.data].storage is StorageType.CPU_Heap or state.sdfg.arrays[node.data].storage is StorageType.GPU_Global:
                    edge_vol = calculate_edge_volume(state, edge)
                    write_edge_volumes.append(edge_vol)                
            
            access_node_write_volume = sp.sympify(sum(write_edge_volumes))
            for (var, (lo, hi, step)) in reversed(range_var_stack):
                
                read_symbol_map = {}
                for sym in access_node_read_volume.free_symbols:
                    read_symbol_map[sym.name] = sym

                write_symbol_map = {}
                for sym in access_node_write_volume.free_symbols:
                    write_symbol_map[sym.name] = sym
                
                shifted_hi = (hi-lo)//step
                shifted_lo = sp.sympify(0)
                sp_var = sp.symbols(var)
                
                if var in read_symbol_map.keys():
                    access_node_read_volume = sp.summation(access_node_read_volume.subs(read_symbol_map[var], (sp.sympify(step)*sp_var+lo)), (var, shifted_lo, shifted_hi))
                else:
                    access_node_read_volume = sp.summation(access_node_read_volume, (var, shifted_lo, shifted_hi))
                
                if var in write_symbol_map.keys():
                    access_node_write_volume = sp.summation(access_node_write_volume.subs(write_symbol_map[var], (sp.sympify(step)*sp_var+lo)), (var, shifted_lo, shifted_hi))
                else:
                    access_node_write_volume = sp.summation(access_node_write_volume, (var, shifted_lo, shifted_hi))

            access_node_read_volume = sp.simplify(access_node_read_volume)
            access_node_write_volume = sp.simplify(access_node_write_volume)

            access_node_volume_map[node] = (sp.simplify(access_node_read_volume), sp.simplify(access_node_write_volume))
            min_read += access_node_read_volume
            min_write += access_node_write_volume
            max_read += access_node_read_volume
            max_write += access_node_write_volume

        elif isinstance(node, nd.NestedSDFG):
            # if we have a nested SDFG we calculate the volume of the nested function separately and then replace the symbols such that they match
            # the symbols of the top-level SDFG
            (max_write_nested, max_read_nested, min_write_nested, min_read_nested) = cfr_volume(node.sdfg, access_node_volume_map, region_volume_map , range_var_stack)
            mapping = {}
    
            # create mapping to replace symbols bound in higher level SDFG correctly
            for sym, parent_sym in node.symbol_mapping.items():
                mapping[sym] = parent_sym
            
            # replace local symbols to avoid name clashes
            for sym in node.sdfg.symbols.keys():
                if sym not in node.symbol_mapping:
                    new_name = f"{sym}_{node.sdfg.cfg_id}"
                    mapping[sym] = sp.Symbol(new_name)
            
            # replace the nested symbols with the corresponding symbols of the higher level SDFG and local symbols with a name that avoids name conflicts
            min_read_nested = min_read_nested.subs(mapping)
            min_write_nested = min_write_nested.subs(mapping)
            max_read_nested = max_read_nested.subs(mapping)
            max_write_nested = max_write_nested.subs(mapping)

            min_read += min_read_nested
            min_write += min_write_nested
            max_read += max_read_nested
            max_write += max_write_nested 

        elif isinstance(node, nd.MapEntry):
            # if we have a map node, we add the map variables to the range variable stack and analyze the scope recursively
            map_variables = list(zip(node.map.params, node.map.range))
            range_var_stack.extend(map_variables)
            min_read_map, min_write_map, max_read_map, max_write_map = scope_volume(state, node, access_node_volume_map, region_volume_map, range_var_stack)
            del range_var_stack[-len(map_variables):]
            min_read += min_read_map
            min_write += min_write_map
            max_read += max_read_map
            max_write += max_write_map
        
    return (min_read, min_write, max_read, max_write)

def cfr_volume(cfr:AbstractControlFlowRegion, access_node_volume_map:dict[nd.AccessNode, (sp.Expr, sp.Expr)], region_volume_map:dict[AbstractControlFlowRegion,tuple:[sp.Expr, sp.Expr, sp.Expr, sp.Expr]], range_var_stack:list[str, tuple])->tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    region_min_read = sp.sympify(0)
    region_min_write = sp.sympify(0)
    region_max_read = sp.sympify(0)
    region_max_write = sp.sympify(0)
    if isinstance(cfr,SDFGState):
            min_read, min_write, max_read, max_write = scope_volume(cfr, None, access_node_volume_map, region_volume_map, range_var_stack)
            region_min_read += min_read
            region_min_write += min_write
            region_max_read += max_read
            region_max_write += max_write

    elif isinstance(cfr, LoopRegion):
        try:
            loop_var = cfr.loop_variable
            lower_bound = loop_analysis.get_init_assignment(cfr)
            upper_bound = loop_analysis.get_loop_end(cfr)
            step = loop_analysis.get_loop_stride(cfr)
            range_var_stack.append((loop_var, (lower_bound, upper_bound, step)))

            min_read = sp.sympify(0)
            min_write = sp.sympify(0)
            max_read = sp.sympify(0)
            max_write = sp.sympify(0)

            for sub_region in cfr.nodes():
                min_read_volume, min_write_volume, max_read_volume, max_write_volume = cfr_volume(sub_region, access_node_volume_map, region_volume_map, range_var_stack)
                min_read += min_read_volume
                min_write += min_write_volume
                max_read += max_read_volume
                max_write += max_write_volume

            del range_var_stack[-1:]
            
            region_min_read += min_read
            region_min_write += min_write
            region_max_read += max_read
            region_max_write += max_write
        except Exception:
            error_msg = f"Loop could not be analyzed. Note that only statically bounded loops can be supported. The volume of this loop (cfg_id: {cfr.cfg_id}) will be counted as 0 for this analysis."
            print(error_msg)
        
    elif isinstance(cfr, ConditionalBlock):
        branch_conditions: Dict[AbstractControlFlowRegion, sp.Expr] = {}
        min_read_volumes = []
        max_read_volumes = []
        min_write_volumes = []
        max_write_volumes = []
        for (condition, branch) in cfr.branches:
            # TODO:Branch conditions are not used at the moment. Later, we might be able to infer more exact information by using them 
            branch_conditions[branch] = pystr_to_symbolic(
                condition.as_string) if condition is not None else sp.sympify(True)
        
            min_branch_read_volume, min_branch_write_volume, max_branch_read_volume, max_branch_write_volume = cfr_volume(branch, access_node_volume_map, region_volume_map, range_var_stack)
            
            min_read_volumes.append(min_branch_read_volume)
            max_read_volumes.append(max_branch_read_volume)
            min_write_volumes.append(min_branch_write_volume)
            max_write_volumes.append(max_branch_write_volume)

        (min_read, min_write, max_read, max_write) = (sp.Min(*min_read_volumes), sp.Min(*min_write_volumes), sp.Max(*max_read_volumes), sp.Max(*max_write_volumes))
        
        region_min_read += min_read
        region_min_write += min_write
        region_max_read += max_read
        region_max_write += max_write
    else:
        for sub_region in cfr.nodes():
            # Since the introduction of ControlFLow regions SDFGs only have only one path. Branching is handled by ControlFlowBlocks
            # Thus we can simply sum the volumes for each individual region to get a total
            min_read, min_write, max_read, max_write = cfr_volume(sub_region, access_node_volume_map, region_volume_map, range_var_stack)
            region_min_read += min_read
            region_min_write += min_write
            region_max_read += max_read
            region_max_write += max_write
    region_volume_map[cfr] = (sp.simplify(region_min_read), sp.simplify(region_min_write), sp.simplify(region_max_read), sp.simplify(region_max_write))
    return region_volume_map[cfr]

    
def calculate_sdfg_volume(sdfg:SDFG):
    # deepcopy such that original sdfg not changed
    sdfg = deepcopy(sdfg)

    # apply SSA pass
    pipeline = FixedPointPipeline([StrictSymbolSSA()])
    pipeline.apply_pass(sdfg, {})

    infer_types.set_default_schedule_and_storage_types(sdfg)
    tvm = {}
    rvm = {}

    (min_read, min_write, max_read, max_write) = cfr_volume(sdfg, tvm, rvm, [])
    return (min_read, min_write, max_read, max_write)
                

################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################

def main() -> None:

    parser = argparse.ArgumentParser('operational_intensity',
                                     usage='python operational_intensity.py [-h] filename',
                                     description='Analyze the operational_intensity of an SDFG.')

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('--assume', nargs='*', help='Collect assumptions about symbols, e.g. x>0 x>y y==5')
    args = parser.parse_args()

    args = parser.parse_args()
    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    sdfg = SDFG.from_file(args.filename)
    
    if args.assume is None:
        args.assume = []

    assumptions = {}
    for x in args.assume:
        a, b = x.split('==')
        if b.isdigit():
            assumptions[a] = int(b)
        else:
            assumptions[a] = b
    
    (min_read, min_write, max_read, max_write) = calculate_sdfg_volume(sdfg)

    print(80 * '-')
    print("Min Reads:", min_read, "bytes \nMax Reads:", max_read, "bytes \nMin writes:", min_write, "bytes \nMax Writes:", max_write, "bytes")
    print(80 * '-')


if __name__ == '__main__':
    main()
