# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Symbolic memory-volume analysis for any input SDFG. Estimates the number of bytes read from
and written to global memory by a DaCe program, as a closed-form symbolic expression in the
program's free symbols. Can be used from the command line as a Python script.

Cost model (where the "bytes moved" come from): every access node touching global memory contributes
the size of its accessed region -- the propagated boundary memlet -- times the element size. That
region is counted **once per enclosing parallel map nest** (data reused across a map is assumed to
stay on chip: infinite cache *within* a nest), but is **multiplied by the trip count of every
enclosing sequential loop** (the cache is assumed flushed on each loop iteration, and there is no
reuse between non-nested scopes). So a stencil whose spatial sweep is a map and whose time axis is a
loop costs ``tsteps * working_set``, while a triangular solver written as sequential loops pays for
its re-reads across those loops.

This is therefore the *compulsory traffic assuming reuse only within a parallel nest* -- a
cache-infinite-within-a-nest estimate. It is deliberately NOT the cache-size-parametric I/O-optimal
lower bound (cf. IOLB, Olivry et al., PLDI 2020, https://inria.hal.science/hal-02910961/document;
a possible future extension corresponds to taking the fast-memory size ``S -> infinity``), and not a
fully naive per-scalar-access count. """

import argparse
import os
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import sympy as sp

from dace import SDFG, SDFGState, dtypes
from dace.data import View
from dace.dtypes import StorageType
from dace.sdfg import infer_types, nodes as nd
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.performance_evaluation.helpers import get_static_symbols
from dace.sdfg.state import (AbstractControlFlowRegion, BreakBlock, ConditionalBlock, ContinueBlock, LoopRegion,
                             ReturnBlock)
from dace.symbolic import pystr_to_symbolic, symbol
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.analysis import loop_analysis

RegionVolumeMap = Dict[AbstractControlFlowRegion, Tuple[sp.Expr, sp.Expr]]
RangeVarStack = List[Tuple[str, Tuple[sp.Expr, sp.Expr, sp.Expr]]]


def safe_summation(summand: sp.Expr, var: sp.Symbol, lower: sp.Expr, upper: sp.Expr) -> sp.Expr:
    """
    Symbolically sum ``summand`` over ``var`` from ``lower`` to ``upper`` (both inclusive).

    Works around a SymPy limitation: ``summation`` may internally call ``posify``, which builds
    ``Dummy(s.name, positive=True, **s.assumptions0)`` for each free symbol. DaCe size symbols
    already carry ``positive=True`` in their assumptions, so this raises ``TypeError`` (the keyword
    is passed twice). The symbols are therefore stripped of their assumptions for the duration of
    the summation and restored afterwards; this does not affect the (polynomial) result.

    :param summand: The expression to sum.
    :param var: The summation variable.
    :param lower: Inclusive lower bound of the summation.
    :param upper: Inclusive upper bound of the summation.
    :return: The closed-form (or, if SymPy cannot evaluate it, unevaluated) sum.
    """
    lower = sp.sympify(lower)
    upper = sp.sympify(upper)
    free_syms = summand.free_symbols | lower.free_symbols | upper.free_symbols
    plain = {s: symbol(s.name) for s in free_syms if s != var and s.assumptions0}
    restore = {v: k for k, v in plain.items()}
    result = sp.summation(summand.subs(plain), (var, lower.subs(plain), upper.subs(plain)))
    return result.subs(restore)


def resolve_minmax_over_range(expr: sp.Expr, var: sp.Symbol, lower: sp.Expr, upper: sp.Expr) -> sp.Expr:
    """
    Resolve ``Max``/``Min`` nodes involving ``var`` when the iteration range ``[lower, upper]`` fixes
    the sign of the comparison.

    For a two-argument ``Max(a, b)``/``Min(a, b)`` whose difference ``a - b`` is affine in ``var``,
    the difference is monotonic over the range, so its sign there is determined by the relevant
    endpoint; when that sign is constant the node is replaced by the larger/smaller operand. This is
    needed because a triangular access (the inner index bounded by the outer one, e.g. cholesky, lu,
    ludcmp) yields ``Max``/``Min`` in the summand that SymPy's ``refine`` cannot simplify, leaving the
    volume as an unevaluated sum.

    :param expr: The summand to simplify.
    :param var: The summation variable.
    :param lower: Inclusive lower bound of the summation range.
    :param upper: Inclusive upper bound of the summation range.
    :return: ``expr`` with the resolvable Max/Min nodes replaced.
    """
    lower = sp.sympify(lower)
    upper = sp.sympify(upper)
    var_name = var.name

    # Symbols of the same name may be distinct instances (different assumptions), so e.g. the ``i``
    # inside ``Max(i, j)`` would not cancel against the ``i`` in a bound ``i - 1``. Compare in a
    # canonical positive-integer namespace so such names cancel and bounds become concrete.
    def canonical(e: sp.Expr) -> sp.Expr:
        return e.subs({s: symbol(s.name, positive=True) for s in e.free_symbols})

    canon_var = symbol(var_name, positive=True)
    canon_lower, canon_upper = canonical(lower), canonical(upper)

    replacements = {}
    for node in expr.atoms(sp.Max, sp.Min):
        if len(node.args) != 2:
            continue
        canon_node = canonical(node)
        if canon_var not in canon_node.free_symbols:
            continue
        a, b = canon_node.args
        diff = sp.expand(a - b)
        coeff = diff.coeff(canon_var)
        if sp.expand(diff - coeff * canon_var).has(canon_var):  # not affine in var
            continue
        # The affine difference is monotonic in var: take its extremes at the range endpoints.
        if coeff.is_nonnegative:
            minimum, maximum = diff.subs(canon_var, canon_lower), diff.subs(canon_var, canon_upper)
        elif coeff.is_nonpositive:
            minimum, maximum = diff.subs(canon_var, canon_upper), diff.subs(canon_var, canon_lower)
        else:
            continue
        if (sp.simplify(minimum) >= 0) == True:  # a >= b over the whole range
            winner = a if isinstance(node, sp.Max) else b
        elif (sp.simplify(maximum) <= 0) == True:  # a <= b over the whole range
            winner = b if isinstance(node, sp.Max) else a
        else:
            continue
        # Map the canonical winner back to the original operand (operand order may differ).
        for original_operand in node.args:
            if canonical(original_operand) == winner:
                replacements[node] = original_operand
                break
    return expr.subs(replacements) if replacements else expr


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
                sp_var = symbol(var)

                if var in read_symbol_map.keys():
                    rvar = read_symbol_map[var]
                    # Resolve Max/Min fixed by this iteration range (e.g. triangular access) before
                    # summing, so the sum has a closed form.
                    access_node_read_volume = resolve_minmax_over_range(access_node_read_volume, rvar, lo, hi)
                    access_node_read_volume = safe_summation(
                        access_node_read_volume.subs(rvar, (sp.sympify(step) * sp_var + lo)), sp_var, shifted_lo,
                        shifted_hi)
                else:
                    access_node_read_volume = safe_summation(access_node_read_volume, sp_var, shifted_lo, shifted_hi)

                if var in write_symbol_map.keys():
                    wvar = write_symbol_map[var]
                    access_node_write_volume = resolve_minmax_over_range(access_node_write_volume, wvar, lo, hi)
                    access_node_write_volume = safe_summation(
                        access_node_write_volume.subs(wvar, (sp.sympify(step) * sp_var + lo)), sp_var, shifted_lo,
                        shifted_hi)
                else:
                    access_node_write_volume = safe_summation(access_node_write_volume, sp_var, shifted_lo, shifted_hi)

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
                    mapping[sym] = symbol(f"{sym}_{node.sdfg.cfg_id}")
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
                range_var_stack.append((f"byte_access_loop_range_var_{len(range_var_stack)}",
                                        (sp.sympify(0), loop_executions, sp.sympify(1))))
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


def analyze_sdfg(sdfg: SDFG, optimize: bool = True) -> Tuple[sp.Expr, sp.Expr]:
    """
    Estimate the global-memory read and write byte volume of an SDFG.

    The SDFG is deep-copied before the symbolic volume is computed and resolved against statically
    known symbols.

    :param sdfg: The SDFG to analyze (left unmodified).
    :param optimize: If True, auto-optimize the SDFG copy first to approximate the effect of
                     compiler optimizations; if False, analyze the raw data volume.
    :return: A tuple of ``(read_volume, write_volume)`` in bytes, as symbolic expressions.
    """
    # Deep-copy so the original SDFG is not modified.
    sdfg = deepcopy(sdfg)
    if optimize:
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
