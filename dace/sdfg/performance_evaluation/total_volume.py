import argparse
from dace.sdfg import nodes as nd
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg import infer_types
from dace import SDFG, SDFGState
from dace.data import View
from typing import Dict
import os
import sympy as sp
import dace.dtypes as dtypes
from copy import deepcopy
from dace.symbolic import pystr_to_symbolic
from dace.dtypes import StorageType
import re
from collections import deque
from dace.transformation.passes.analysis import loop_analysis

from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion, ConditionalBlock, ReturnBlock, ContinueBlock, BreakBlock

import dace.transformation.auto.auto_optimize as opt


def subs_till_fixed_point(expr:sp.Expr, symbol_map:Dict[sp.Expr, sp.Expr]):
    """
    Takes a sympy expression and a symbol mapping and applies the mapping to the expression until a fixed point is reached
    Needs the guarantee that the symbol mapping does not have cyclic dependencies.

    :param expr: Description
    :param symbol_map: Description
    :return: Description
    """
    prev = None
    curr = expr

    while prev != curr:
        prev = curr
        curr = curr.subs(symbol_map)

    return curr

def get_static_symbols(sdfg: SDFG):
    """
    Returns a mapping of symbols that are assigned exactly at one point in the sdfg.
    
    :param sdfg: The sdfg for which we want to find the static symbols and their corresponding assignment
    :return: The mapping of the symbols to higher levels (iterated to a fixed point)
    """

    
    patterns = [
        "dace.complex128",
        "dace.float64",
        "dace.float32",
        "dace.int64",
        "dace.int32",
        "dace.int16",
        "dace.uint32",
        "dace.uint16",
        "dace.uint8",
        "float",
        "int"
    ]


    type_regex = re.compile("|".join(map(re.escape, patterns)))

    static_symbol_mapping:Dict[sp.Symbol, sp.Expr] = {sp.Symbol(a): sp.Symbol(a) for a in sdfg.arg_names}
    non_static_symbols = set() 
    for node, containing_state in sdfg.all_nodes_recursive():
        if isinstance(node, nd.AccessNode):
            
            if containing_state.in_degree(node) == 1:
                edge = containing_state.in_edges(node)[0]
                source = edge.src
                
                if edge.data.volume == 1:
                    if isinstance(source, nd.Tasklet):
                        tasklet = source
                        in_map = {}
                        out_map = {}
                        # Incoming edges: symbols feeding the tasklet
                        for e in containing_state.in_edges(tasklet):
                            if not isinstance(e.src, nd.AccessNode):
                                continue
                            sym = str(e.src.data)
                            in_map[e.dst_conn] = sym
                        # Outgoing edges: symbols written by the tasklet
                        # Out edges should only be one, but for safety we iterate
                        for e in containing_state.out_edges(tasklet):
                            if not isinstance(e.dst, nd.AccessNode):
                                continue
                            sym = str(e.dst.data)
                            out_map[e.src_conn] = sym

                        in_map = {sp.Symbol(k): sp.Symbol(v) for k,v in in_map.items()}
                        out_map = {sp.Symbol(k): sp.Symbol(v) for k,v in out_map.items()}
                        code = tasklet.code.as_string.strip()
                        # Expect a single assignment
                        lines = [l.strip() for l in code.splitlines() if l.strip()]
                        lhs, rhs = lines[0].split('=',1)
                        lhs = lhs.strip()
                        rhs = rhs.strip()
                        rhs = type_regex.sub("", rhs)
                        # Parse RHS using SymPy, with tasklet inputs substituted
                        lhs_sympy = pystr_to_symbolic(lhs)
                        lhs_sympy = lhs_sympy.subs(out_map)

                        if not lhs_sympy in static_symbol_mapping.keys():
                            try:
                                rhs_sympy = pystr_to_symbolic(rhs)
                                rhs_sympy = rhs_sympy.subs(in_map)
                                static_symbol_mapping[lhs_sympy] = rhs_sympy
                            except:
                                non_static_symbols.add(lhs_sympy)
                        else:
                            non_static_symbols.add(lhs_sympy)

                    elif isinstance(source, nd.AccessNode):
                        data_sym = sp.Symbol(source.data)
                        nd_sym = sp.Symbol(node.data)
                        if not data_sym in static_symbol_mapping.keys():
                            static_symbol_mapping[data_sym] = nd_sym
                        else:
                            non_static_symbols.add(data_sym)

    static_symbol_mapping = {k: v for (k, v) in static_symbol_mapping.items() if k not in non_static_symbols}
    static_symbol_mapping = {k: subs_till_fixed_point(v, static_symbol_mapping) for k,v in static_symbol_mapping.items()}
    return static_symbol_mapping


def calculate_edge_volume(state: SDFGState, edge:MultiConnectorEdge):
    vol = sp.sympify(1)
    vol = sp.sympify(1)
    for (l, u, s) in edge.data.subset.ranges:
        vol *= (u-l)/s    
    return vol*state.sdfg.arrays[edge.data.data].dtype.bytes

def scope_volume(state: SDFGState, entry=None, region_volume_map:dict[AbstractControlFlowRegion,tuple:[sp.Expr, sp.Expr, sp.Expr, sp.Expr]]={}, range_var_stack:list[tuple[str, tuple]]=[]):
    scope_nodes = state.scope_children()[entry]
    read = sp.sympify(0)
    write = sp.sympify(0)
    for node in scope_nodes:        
        if isinstance(node, nd.AccessNode):
            if isinstance(state.sdfg.arrays[node.data], View):
                continue
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
                sp_var = sp.Symbol(var)
                
                if var in read_symbol_map.keys():
                    access_node_read_volume = sp.summation(access_node_read_volume.subs(read_symbol_map[var], (sp.sympify(step)*sp_var+lo)), (sp_var, shifted_lo, shifted_hi))
                else:
                    access_node_read_volume = sp.summation(access_node_read_volume, (sp_var, shifted_lo, shifted_hi))
                
                if var in write_symbol_map.keys():
                    access_node_write_volume = sp.summation(access_node_write_volume.subs(write_symbol_map[var], (sp.sympify(step)*sp_var+lo)), (sp_var, shifted_lo, shifted_hi))
                else:
                    access_node_write_volume = sp.summation(access_node_write_volume, (sp_var, shifted_lo, shifted_hi))

            access_node_read_volume = sp.simplify(access_node_read_volume)
            access_node_write_volume = sp.simplify(access_node_write_volume)

            read += access_node_read_volume
            write += access_node_write_volume

        elif isinstance(node, nd.NestedSDFG):
            # if we have a nested SDFG we calculate the volume of the nested function separately and then replace the symbols such that they match
            # the symbols of the top-level SDFG
            read_nested, write_nested = cfr_volume(node.sdfg, region_volume_map , range_var_stack)
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
            read_nested = read_nested.subs(mapping)
            write_nested = write_nested.subs(mapping)

            read += read_nested
            write += write_nested

        elif isinstance(node, nd.MapEntry):
            # if we have a map node, we add the map variables to the range variable stack and analyze the scope recursively
            map_variables = list(zip(node.map.params, node.map.range))
            range_var_stack.extend(map_variables)
            read_map, write_map = scope_volume(state, node, region_volume_map, range_var_stack)
            del range_var_stack[-len(map_variables):]
            read += read_map
            write += write_map
           
    return read, write

def cfr_volume(control_flow_region:AbstractControlFlowRegion, region_volume_map:dict[AbstractControlFlowRegion,tuple:[sp.Expr, sp.Expr]], range_var_stack:list[str, tuple], detailed_analysis=False)->tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    
    for cfr in control_flow_region.nodes():
        if isinstance(cfr,SDFGState):
                scope_read, scope_write = scope_volume(cfr, None, region_volume_map, range_var_stack)
                region_volume_map[cfr] = (scope_read, scope_write)
        elif isinstance(cfr, LoopRegion):
            try:
                loop_var = cfr.loop_variable
                lower_bound = loop_analysis.get_init_assignment(cfr)
                upper_bound = loop_analysis.get_loop_end(cfr)
                step = loop_analysis.get_loop_stride(cfr)
                if not loop_var:
                    raise
                range_var_stack.append((loop_var, (lower_bound, upper_bound, step)))
                loop_read, loop_write = cfr_volume(cfr, region_volume_map, range_var_stack, detailed_analysis)

                del range_var_stack[-1:]
                
                region_volume_map[cfr] = (loop_read, loop_write)

            except Exception:
                loop_executions = cfr.start_block.executions
                range_var_stack.append((f"byte_access_loop_range_var_{len(range_var_stack)}", (sp.sympify(0), loop_executions, sp.sympify(1))))
                inner_read, inner_write = cfr_volume(cfr, region_volume_map, range_var_stack, detailed_analysis)
                del range_var_stack[-1:]

                region_volume_map[cfr] = (inner_read, inner_write)
                
        elif isinstance(cfr, ConditionalBlock):
            branch_conditions: Dict[AbstractControlFlowRegion, sp.Expr] = {}
            branch_reads = []
            branch_writes = []
            for (condition, branch) in cfr.branches:
                branch_conditions[branch] = pystr_to_symbolic(
                    condition.as_string) if condition is not None else sp.sympify(True)
            
                branch_read, branch_write = cfr_volume(branch, region_volume_map, range_var_stack)
                
                branch_reads.append(branch_read)
                branch_writes.append(branch_write)

            if detailed_analysis: 
                cond_reads = zip(branch_reads, branch_conditions)  
                cond_writes = zip(branch_writes, branch_conditions)  
                cond_read, cond_write = (sp.Piecewise(*cond_reads), sp.Piecewise(*cond_writes))
            else:
                cond_read, cond_write = (sp.Max(*branch_reads), sp.Max(*branch_writes))
            
            region_volume_map[cfr] = (cond_read, cond_write)
        elif isinstance(cfr, (ReturnBlock, ContinueBlock, BreakBlock)):
            region_volume_map[cfr] = (sp.sympify(0), sp.sympify(0))
        else:
            # Since the introduction of ControlFLow regions SDFGs only have only one path. Branching is handled by ControlFlowBlocks
            # Thus we can simply sum the volumes for each individual region to get a total
            reg_read, reg_write = cfr_volume(cfr, region_volume_map, range_var_stack)
            region_volume_map[cfr] = (reg_read, reg_write)

    traversal_q = deque()
    traversal_q.append((control_flow_region.start_block, {}))

    region_reads = sp.sympify(0)
    region_writes = sp.sympify(0)

    while traversal_q:
        current_region, current_mapping = traversal_q.popleft()

        for oedge in control_flow_region.out_edges(current_region):
            new_mapping = deepcopy(current_mapping)
            oedge_mapping = {pystr_to_symbolic(k):pystr_to_symbolic(v) for k, v in oedge.data.assignments.items()}
            for k, v in oedge_mapping.items():
                new_mapping[k] = oedge_mapping[k]
            
            for k, v in new_mapping.items():
                new_mapping[k] = v.subs(oedge_mapping)
            traversal_q.append((oedge.dst, new_mapping))
            
        region_reads += region_volume_map[current_region][0].subs(current_mapping)
        region_writes += region_volume_map[current_region][1].subs(current_mapping)
    
    region_volume_map[control_flow_region] = (region_reads, region_writes)

    return region_volume_map[control_flow_region]

    
def analyze_sdfg(sdfg:SDFG):
    # deepcopy such that original sdfg not changed
    sdfg = deepcopy(sdfg)

    # Try to use an optimized version of the SDFG to account for compiler optimizations
    try:
        opt.auto_optimize(sdfg, dtypes.DeviceType.CPU)
    except:
        pass

    infer_types.set_default_schedule_and_storage_types(sdfg)
    tvm = {}
    rvm = {}
    static_symbol_mapping = get_static_symbols(sdfg)

    read, write = cfr_volume(sdfg, tvm, [], False)

    read = read.subs(static_symbol_mapping)
    write = write.subs(static_symbol_mapping)
    return read, write
                

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
    
    (min_read, min_write, max_read, max_write) = analyze_sdfg(sdfg)

    print(80 * '-')
    print("Min Reads:", min_read, "bytes \nMax Reads:", max_read, "bytes \nMin writes:", min_write, "bytes \nMax Writes:", max_write, "bytes")
    print(80 * '-')


if __name__ == '__main__':
    main()
