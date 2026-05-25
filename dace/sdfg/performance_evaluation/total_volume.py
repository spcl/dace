# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Symbolic memory-volume analysis for any input SDFG. Estimates the number of bytes read from
and written to global memory by a DaCe program, as a closed-form symbolic expression in the
program's free symbols. Can be used from the command line as a Python script. """

import argparse
import os
import re
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import sympy as sp

from dace import SDFG, SDFGState, dtypes
from dace.data import View
from dace.dtypes import StorageType
from dace.sdfg import infer_types, nodes as nd
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import (AbstractControlFlowRegion, BreakBlock, ConditionalBlock, ContinueBlock, LoopRegion,
                             ReturnBlock)
from dace.symbolic import pystr_to_symbolic
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.analysis import loop_analysis

RegionVolumeMap = Dict[AbstractControlFlowRegion, Tuple[sp.Expr, sp.Expr]]
RangeVarStack = List[Tuple[str, Tuple[sp.Expr, sp.Expr, sp.Expr]]]


def subs_till_fixed_point(expr: sp.Expr, symbol_map: Dict[sp.Expr, sp.Expr]) -> sp.Expr:
    """
    Apply a symbol mapping to a symbolic expression repeatedly until a fixed point is reached.

    Requires that the symbol mapping has no cyclic dependencies, otherwise it would not converge.

    :param expr: The expression to substitute into.
    :param symbol_map: Mapping from symbols to their replacement expressions.
    :return: The expression after substituting to a fixed point.
    """
    prev = None
    curr = expr
    while prev != curr:
        prev = curr
        curr = curr.subs(symbol_map)
    return curr


def get_static_symbols(sdfg: SDFG) -> Dict[sp.Symbol, sp.Expr]:
    """
    Find the symbols that are assigned at exactly one point in the SDFG (i.e., statically known).

    A symbol is static if it is written by a single length-1 access (from a tasklet performing one
    assignment, or by a single copy from another access node). Symbols written in more than one
    place are excluded.

    :param sdfg: The SDFG for which to find static symbols and their assignments.
    :return: Mapping from each static symbol to its defining expression (resolved to a fixed point).
    """
    patterns = [
        "dace.complex128", "dace.float64", "dace.float32", "dace.int64", "dace.int32", "dace.int16", "dace.uint32",
        "dace.uint16", "dace.uint8", "float", "int"
    ]
    type_regex = re.compile("|".join(map(re.escape, patterns)))

    static_symbol_mapping: Dict[sp.Symbol, sp.Expr] = {sp.Symbol(a): sp.Symbol(a) for a in sdfg.arg_names}
    non_static_symbols = set()
    for node, containing_state in sdfg.all_nodes_recursive():
        if not isinstance(node, nd.AccessNode):
            continue
        if containing_state.in_degree(node) != 1:
            continue
        edge = containing_state.in_edges(node)[0]
        source = edge.src
        if edge.data.volume != 1:
            continue

        if isinstance(source, nd.Tasklet):
            tasklet = source
            in_map = {}
            out_map = {}
            # Incoming edges: symbols feeding the tasklet.
            for e in containing_state.in_edges(tasklet):
                if not isinstance(e.src, nd.AccessNode):
                    continue
                in_map[e.dst_conn] = str(e.src.data)
            # Outgoing edges: symbols written by the tasklet (expected to be a single edge).
            for e in containing_state.out_edges(tasklet):
                if not isinstance(e.dst, nd.AccessNode):
                    continue
                out_map[e.src_conn] = str(e.dst.data)

            in_map = {sp.Symbol(k): sp.Symbol(v) for k, v in in_map.items()}
            out_map = {sp.Symbol(k): sp.Symbol(v) for k, v in out_map.items()}
            code = tasklet.code.as_string.strip()
            # Expect a single assignment.
            lines = [l.strip() for l in code.splitlines() if l.strip()]
            if len(lines) > 1:
                non_static_symbols.add(node.data)
                continue
            lhs, rhs = lines[0].split('=', 1)
            lhs = lhs.strip()
            rhs = type_regex.sub("", rhs.strip())
            lhs_sympy = pystr_to_symbolic(lhs).subs(out_map)

            if lhs_sympy not in static_symbol_mapping.keys():
                try:
                    static_symbol_mapping[lhs_sympy] = pystr_to_symbolic(rhs).subs(in_map)
                except Exception:
                    non_static_symbols.add(lhs_sympy)
            else:
                non_static_symbols.add(lhs_sympy)

        elif isinstance(source, nd.AccessNode):
            data_sym = sp.Symbol(source.data)
            if data_sym not in static_symbol_mapping.keys():
                static_symbol_mapping[data_sym] = sp.Symbol(node.data)
            else:
                non_static_symbols.add(data_sym)

    static_symbol_mapping = {k: v for (k, v) in static_symbol_mapping.items() if k not in non_static_symbols}
    static_symbol_mapping = {k: subs_till_fixed_point(v, static_symbol_mapping) for k, v in static_symbol_mapping.items()}
    return static_symbol_mapping


def calculate_edge_volume(state: SDFGState, edge: MultiConnectorEdge) -> sp.Expr:
    """
    Compute the number of bytes moved by a single memlet edge.

    :param state: The state containing the edge.
    :param edge: The memlet edge whose byte volume is computed.
    :return: The number of elements in the edge's subset multiplied by the array's element size.
    """
    vol = edge.data.subset.num_elements()
    return vol * state.sdfg.arrays[edge.data.data].dtype.bytes


def scope_volume(state: SDFGState,
                 entry: Optional[nd.MapEntry] = None,
                 region_volume_map: Optional[RegionVolumeMap] = None,
                 range_var_stack: Optional[RangeVarStack] = None) -> Tuple[sp.Expr, sp.Expr]:
    """
    Compute the read and write byte volume of a single scope within a state.

    Access-node edges to/from global memory (CPU heap, GPU global) contribute their byte volume.
    Map scopes are analyzed recursively, with the map's parameters pushed onto the range-variable
    stack so that the enclosed accesses are summed over the map's iteration domain.

    :param state: The state being analyzed.
    :param entry: The map entry whose scope is analyzed, or ``None`` for the top-level scope.
    :param region_volume_map: Accumulator mapping control-flow regions to their volumes.
    :param range_var_stack: Stack of enclosing iteration variables and their ``(lo, hi, step)`` ranges.
    :return: A tuple of ``(read_volume, write_volume)`` in bytes.
    """
    if region_volume_map is None:
        region_volume_map = {}
    if range_var_stack is None:
        range_var_stack = []

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
                if state.sdfg.arrays[node.data].storage in (StorageType.CPU_Heap, StorageType.GPU_Global):
                    read_edge_volumes.append(calculate_edge_volume(state, edge))

            access_node_read_volume = sp.sympify(sum(read_edge_volumes))
            write_edge_volumes = []
            for edge in state.in_edges(node):
                if isinstance(edge.src, nd.NestedSDFG):
                    continue
                if state.sdfg.arrays[node.data].storage in (StorageType.CPU_Heap, StorageType.GPU_Global):
                    write_edge_volumes.append(calculate_edge_volume(state, edge))

            access_node_write_volume = sp.sympify(sum(write_edge_volumes))
            for (var, (lo, hi, step)) in reversed(range_var_stack):
                read_symbol_map = {sym.name: sym for sym in access_node_read_volume.free_symbols}
                write_symbol_map = {sym.name: sym for sym in access_node_write_volume.free_symbols}

                shifted_hi = (hi - lo) // step
                shifted_lo = sp.sympify(0)
                sp_var = sp.Symbol(var)

                if var in read_symbol_map.keys():
                    access_node_read_volume = sp.summation(
                        access_node_read_volume.subs(read_symbol_map[var], (sp.sympify(step) * sp_var + lo)),
                        (sp_var, shifted_lo, shifted_hi))
                else:
                    access_node_read_volume = sp.summation(access_node_read_volume, (sp_var, shifted_lo, shifted_hi))

                if var in write_symbol_map.keys():
                    access_node_write_volume = sp.summation(
                        access_node_write_volume.subs(write_symbol_map[var], (sp.sympify(step) * sp_var + lo)),
                        (sp_var, shifted_lo, shifted_hi))
                else:
                    access_node_write_volume = sp.summation(access_node_write_volume, (sp_var, shifted_lo, shifted_hi))

            read += sp.simplify(access_node_read_volume)
            write += sp.simplify(access_node_write_volume)

        elif isinstance(node, nd.NestedSDFG):
            # Analyze the nested SDFG separately, then rename its symbols to match the parent SDFG.
            read_nested, write_nested = cfr_volume(node.sdfg, region_volume_map, range_var_stack)
            mapping = {}
            # Map symbols bound in the higher-level SDFG to their parent names.
            for sym, parent_sym in node.symbol_mapping.items():
                mapping[sym] = parent_sym
            # Rename purely local symbols to avoid clashes with parent-level names.
            for sym in node.sdfg.symbols.keys():
                if sym not in node.symbol_mapping:
                    mapping[sym] = sp.Symbol(f"{sym}_{node.sdfg.cfg_id}")
            read += read_nested.subs(mapping)
            write += write_nested.subs(mapping)

        elif isinstance(node, nd.MapEntry):
            # Push the map's parameters onto the range stack and analyze the scope recursively.
            map_variables = list(zip(node.map.params, node.map.range))
            range_var_stack.extend(map_variables)
            read_map, write_map = scope_volume(state, node, region_volume_map, range_var_stack)
            del range_var_stack[-len(map_variables):]
            read += read_map
            write += write_map

    return read, write


def cfr_volume(control_flow_region: AbstractControlFlowRegion,
               region_volume_map: Optional[RegionVolumeMap] = None,
               range_var_stack: Optional[RangeVarStack] = None,
               detailed_analysis: bool = False) -> Tuple[sp.Expr, sp.Expr]:
    """
    Compute the read and write byte volume of a control-flow region.

    States are analyzed scope-by-scope; loop regions multiply (symbolically sum) their body over
    the loop range; conditional blocks combine their branches (``Max`` over branches by default, or
    a ``Piecewise`` over branch conditions when ``detailed_analysis`` is set). Interstate-edge
    assignments are propagated into the per-region volumes during the final traversal.

    :param control_flow_region: The region (SDFG, loop, conditional, ...) to analyze.
    :param region_volume_map: Accumulator mapping control-flow regions to their volumes.
    :param range_var_stack: Stack of enclosing iteration variables and their ``(lo, hi, step)`` ranges.
    :param detailed_analysis: If True, combine conditional branches with ``Piecewise`` instead of ``Max``.
    :return: A tuple of ``(read_volume, write_volume)`` in bytes.
    """
    if region_volume_map is None:
        region_volume_map = {}
    if range_var_stack is None:
        range_var_stack = []

    for cfr in control_flow_region.nodes():
        if isinstance(cfr, SDFGState):
            scope_read, scope_write = scope_volume(cfr, None, region_volume_map, range_var_stack)
            region_volume_map[cfr] = (scope_read, scope_write)
        elif isinstance(cfr, LoopRegion):
            try:
                loop_var = cfr.loop_variable
                lower_bound = loop_analysis.get_init_assignment(cfr)
                upper_bound = loop_analysis.get_loop_end(cfr)
                step = loop_analysis.get_loop_stride(cfr)
                if not loop_var:
                    raise ValueError('Loop region has no loop variable')
                range_var_stack.append((loop_var, (lower_bound, upper_bound, step)))
                loop_read, loop_write = cfr_volume(cfr, region_volume_map, range_var_stack, detailed_analysis)
                del range_var_stack[-1:]
                region_volume_map[cfr] = (loop_read, loop_write)
            except Exception:
                # Fall back to the (statically estimated) number of executions of the loop body.
                loop_executions = cfr.start_block.executions
                range_var_stack.append(
                    (f"byte_access_loop_range_var_{len(range_var_stack)}", (sp.sympify(0), loop_executions,
                                                                            sp.sympify(1))))
                inner_read, inner_write = cfr_volume(cfr, region_volume_map, range_var_stack, detailed_analysis)
                del range_var_stack[-1:]
                region_volume_map[cfr] = (inner_read, inner_write)

        elif isinstance(cfr, ConditionalBlock):
            branch_conditions: Dict[AbstractControlFlowRegion, sp.Expr] = {}
            branch_reads = []
            branch_writes = []
            for (condition, branch) in cfr.branches:
                branch_conditions[branch] = (pystr_to_symbolic(condition.as_string)
                                             if condition is not None else sp.sympify(True))
                branch_read, branch_write = cfr_volume(branch, region_volume_map, range_var_stack)
                branch_reads.append(branch_read)
                branch_writes.append(branch_write)

            if detailed_analysis:
                conditions = list(branch_conditions.values())
                cond_read = sp.Piecewise(*zip(branch_reads, conditions))
                cond_write = sp.Piecewise(*zip(branch_writes, conditions))
            else:
                cond_read, cond_write = (sp.Max(*branch_reads), sp.Max(*branch_writes))
            region_volume_map[cfr] = (cond_read, cond_write)
        elif isinstance(cfr, (ReturnBlock, ContinueBlock, BreakBlock)):
            region_volume_map[cfr] = (sp.sympify(0), sp.sympify(0))
        else:
            # With control-flow regions, branching is handled by control-flow blocks, so the
            # remaining regions form a single path and their volumes can simply be summed.
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
            oedge_mapping = {pystr_to_symbolic(k): pystr_to_symbolic(v) for k, v in oedge.data.assignments.items()}
            for k, v in oedge_mapping.items():
                new_mapping[k] = v
            for k, v in new_mapping.items():
                new_mapping[k] = v.subs(oedge_mapping)
            traversal_q.append((oedge.dst, new_mapping))

        region_reads += region_volume_map[current_region][0].subs(current_mapping)
        region_writes += region_volume_map[current_region][1].subs(current_mapping)

    region_volume_map[control_flow_region] = (region_reads, region_writes)
    return region_volume_map[control_flow_region]


def analyze_sdfg(sdfg: SDFG) -> Tuple[sp.Expr, sp.Expr]:
    """
    Estimate the global-memory read and write byte volume of an SDFG.

    The SDFG is deep-copied and auto-optimized (to approximate the effect of compiler
    optimizations) before the symbolic volume is computed and resolved against statically known
    symbols.

    :param sdfg: The SDFG to analyze (left unmodified).
    :return: A tuple of ``(read_volume, write_volume)`` in bytes, as symbolic expressions.
    """
    # Deep-copy so the original SDFG is not modified.
    sdfg = deepcopy(sdfg)
    # Try to use an optimized version of the SDFG to account for compiler optimizations.
    try:
        auto_optimize(sdfg, dtypes.DeviceType.CPU)
    except Exception:
        pass

    infer_types.set_default_schedule_and_storage_types(sdfg)
    region_volume_map: RegionVolumeMap = {}
    static_symbol_mapping = get_static_symbols(sdfg)

    read, write = cfr_volume(sdfg, region_volume_map, [], False)
    read = read.subs(static_symbol_mapping)
    write = write.subs(static_symbol_mapping)
    return read, write


def main() -> None:
    """ Command-line entry point: analyze the memory volume of an SDFG file. """
    parser = argparse.ArgumentParser('total_volume',
                                     usage='python total_volume.py [-h] filename',
                                     description='Analyze the memory volume of an SDFG.')
    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('--assume', nargs='*', help='Collect assumptions about symbols, e.g. x>0 x>y y==5')
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    sdfg = SDFG.from_file(args.filename)
    read, write = analyze_sdfg(sdfg)

    print(80 * '-')
    print("Reads:", read, "bytes \nWrites:", write, "bytes")
    print(80 * '-')


if __name__ == '__main__':
    main()
