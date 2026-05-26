# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Symbolic memory-volume analysis for any input SDFG. Estimates the number of bytes read from
and written to global memory by a DaCe program, as a closed-form symbolic expression in the
program's free symbols. Can be used from the command line as a Python script.

Cost model (where the "bytes moved" come from): every access node touching global memory contributes
the size of its accessed region times the element size. That region is counted **once per enclosing
parallel map nest** (data reused across a map is assumed to stay on chip: infinite cache *within* a
nest), but is **multiplied by the trip count of every enclosing sequential loop** (the cache is
assumed flushed on each loop iteration, and there is no reuse between non-nested scopes). So a stencil
whose spatial sweep is a map and whose time axis is a loop costs ``tsteps * working_set``, while a
triangular solver written as sequential loops pays for its re-reads across those loops.

The accessed region of a map is estimated as the tighter (``Min``) of two upper bounds on the working
set: the propagated boundary memlet (a bounding box, which over-counts disjoint slices such as a row
and a column of a triangular access) and the sum of the individual per-connector footprints (which
over-counts overlapping slices such as a stencil's neighbourhood). So a triangular access reading a
row and a column costs the two slices, not the dense box spanning them.

This is therefore the *compulsory traffic assuming reuse only within a parallel nest* -- a
cache-infinite-within-a-nest estimate. It is deliberately NOT the cache-size-parametric I/O-optimal
lower bound (cf. IOLB, Olivry et al., PLDI 2020, https://inria.hal.science/hal-02910961/document;
a possible future extension corresponds to taking the fast-memory size ``S -> infinity``), and not a
fully naive per-scalar-access count. """

import argparse
import os
import warnings
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import sympy as sp

from dace import SDFG, SDFGState, dtypes
from dace.data import View
from dace.dtypes import StorageType
from dace.sdfg import infer_types, nodes as nd
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.performance_evaluation.helpers import get_static_symbols, has_unstructured_control_flow
from dace.sdfg.propagation import propagate_memlet
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, LoopRegion
from dace.symbolic import pystr_to_symbolic, symbol, int_floor, simplify
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.analysis import loop_analysis

RegionVolumeMap = Dict[AbstractControlFlowRegion, Tuple[sp.Expr, sp.Expr]]
RangeVarStack = List[Tuple[str, Tuple[sp.Expr, sp.Expr, sp.Expr]]]

# Problem sizes at which a residual Max/Min (one no loop range could resolve) is compared to pick the
# dominant -- tighter -- operand; the choice must agree across all of them or the node is left as-is.
_DOMINANCE_PROBE_SIZES = (64, 257, 1024)


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
    lower = pystr_to_symbolic(lower)
    upper = pystr_to_symbolic(upper)
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
    lower = pystr_to_symbolic(lower)
    upper = pystr_to_symbolic(upper)
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
        # Canonicalization can resolve the node outright (e.g. ``Max(j, j - 1)`` -> ``j``), leaving
        # fewer than two arguments; the surviving expression already names the winning operand.
        if not isinstance(canon_node, (sp.Max, sp.Min)) or len(canon_node.args) != 2:
            for original_operand in node.args:
                if canonical(original_operand) == canon_node:
                    replacements[node] = original_operand
                    break
            continue
        if canon_var not in canon_node.free_symbols:
            continue
        a, b = canon_node.args
        # A positive common factor makes ``Min(c*X, d*X) = X*Min(c, d)``: divide it out so the
        # comparison becomes affine in ``var`` (e.g. ``Min(2*j, j*(i - j + 1))`` -> compare ``2`` vs
        # ``i - j + 1``). Comparing the reduced operands is valid because the factor is nonnegative
        # in the canonical positive namespace, so it preserves the ordering of ``a`` and ``b``.
        a_cmp, b_cmp = a, b
        try:
            common = sp.gcd(a, b)
        except Exception:
            common = sp.Integer(1)
        if common != 1 and common.is_nonnegative:
            a_cmp, b_cmp = sp.expand(a / common), sp.expand(b / common)
        diff = sp.expand(a_cmp - b_cmp)
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
        if (simplify(minimum) >= 0) == True:  # a >= b over the whole range
            winner = a if isinstance(node, sp.Max) else b
        elif (simplify(maximum) <= 0) == True:  # a <= b over the whole range
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


def resolve_size_dominated_minmax(expr: sp.Expr) -> sp.Expr:
    """
    Resolve any residual two-argument ``Max``/``Min`` -- one that no loop range could fix, e.g. a
    stencil's bounding box versus its per-connector sum -- by dominance at representative large
    problem sizes (:data:`_DOMINANCE_PROBE_SIZES`).

    Both operands are valid upper bounds on the accessed working set (see :func:`_edge_access_volume`),
    so resolving the node to either one is sound; this picks the dominant -- tighter -- operand for a
    clean closed form, and only when the choice agrees across all sample sizes (otherwise a crossover
    is possible and the node is left symbolic).

    :param expr: The volume expression possibly containing residual ``Max``/``Min`` nodes.
    :return: The expression with size-dominated ``Max``/``Min`` nodes resolved to a single operand.
    """
    expr = pystr_to_symbolic(expr)
    replacements = {}
    for node in expr.atoms(sp.Max, sp.Min):
        if len(node.args) != 2:
            continue
        free_symbols = list(node.free_symbols)
        if not free_symbols:
            continue
        winners = set()
        for sample in _DOMINANCE_PROBE_SIZES:
            try:
                values = [float(arg.subs({s: sample for s in free_symbols})) for arg in node.args]
            except (TypeError, ValueError):
                winners.clear()
                break
            dominant = min(values) if isinstance(node, sp.Min) else max(values)
            winners.add(node.args[values.index(dominant)])
        if len(winners) == 1:
            replacements[node] = winners.pop()
    return expr.subs(replacements) if replacements else expr


def _edge_access_volume(state: SDFGState, edge: MultiConnectorEdge, neighbor) -> sp.Expr:
    """
    Compute the bytes moved by one memlet ``edge`` crossing into or out of a scope.

    When the edge crosses a map scope, the accessed region is taken as the tighter (``Min``) of two
    valid upper bounds on the working set: the propagated boundary memlet (a bounding box, which
    over-counts disjoint slices such as a row and a column of a triangular access) and the sum of the
    individual per-connector footprints (which over-counts overlapping slices such as a stencil's
    neighbourhood). Otherwise the boundary memlet is used directly.

    :param edge: The memlet edge whose byte volume is computed.
    :param neighbor: Callable returning the node on the other end of the edge (the scope node for a
                     map-crossing edge).
    :return: The byte volume of the edge.
    """
    bounding_box = calculate_edge_volume(state, edge)
    scope_node = neighbor(edge)
    if not isinstance(scope_node, (nd.MapEntry, nd.MapExit)):
        return bounding_box
    # The per-connector footprints are the inner edges on the same array, each propagated over the
    # map on its own (rather than unioned into the single boundary memlet).
    inner_edges = state.out_edges(scope_node) if isinstance(scope_node, nd.MapEntry) else state.in_edges(scope_node)
    inner_edges = [e for e in inner_edges if e.data.data == edge.data.data and e.data.subset is not None]
    if not inner_edges:
        return bounding_box
    element_bytes = state.sdfg.arrays[edge.data.data].dtype.bytes
    try:
        per_connector = sum(
            propagate_memlet(state, e.data, scope_node, False).subset.num_elements() for e in inner_edges)
    except Exception:
        return bounding_box
    return sp.Min(bounding_box, per_connector * element_bytes)


def _access_volume(state: SDFGState, node: nd.AccessNode, edges, neighbor) -> sp.Expr:
    """
    Sum the byte volume of an access node's ``edges`` that touch global memory.

    :param edges: The incoming or outgoing edges of ``node`` to account.
    :param neighbor: Callable returning the node on the other end of an edge; edges to/from a
                     ``NestedSDFG`` are skipped, since the nested SDFG is analyzed on its own.
    :return: The total byte volume, or zero if the array is not in global memory.
    """
    if state.sdfg.arrays[node.data].storage not in (StorageType.CPU_Heap, StorageType.GPU_Global):
        return pystr_to_symbolic(0)
    return pystr_to_symbolic(
        sum(_edge_access_volume(state, e, neighbor) for e in edges if not isinstance(neighbor(e), nd.NestedSDFG)))


def _accumulate_volume_over_var(volume: sp.Expr, var: str, lo: sp.Expr, hi: sp.Expr, step: sp.Expr) -> sp.Expr:
    """
    Sum a per-iteration access ``volume`` over one enclosing loop/map variable ``var`` ranging over
    ``[lo, hi]`` with stride ``step`` (the volume is summed across a sequential range).

    :param volume: The byte volume accessed in one iteration.
    :param var: The iteration variable name.
    :return: The volume summed over the range.
    """
    sp_var = symbol(var)
    shifted_hi = int_floor(hi - lo, step)
    present = {s.name: s for s in volume.free_symbols}
    if var in present:
        # Resolve Max/Min fixed by this range (e.g. a triangular access) before summing for a closed
        # form. Iterate to a fixed point so nested nodes collapse inside-out (an inner ``Max(i, j)``
        # must resolve before the ``Min`` containing it becomes comparable).
        previous = None
        while volume != previous:
            previous = volume
            volume = resolve_minmax_over_range(volume, present[var], lo, hi)
        volume = volume.subs(present[var], pystr_to_symbolic(step) * sp_var + lo)
    return safe_summation(volume, sp_var, pystr_to_symbolic(0), shifted_hi)


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
    read = pystr_to_symbolic(0)
    write = pystr_to_symbolic(0)
    for node in scope_nodes:
        if isinstance(node, nd.AccessNode):
            if isinstance(state.sdfg.arrays[node.data], View):
                continue
            # Read edges leave the access node (its data feeds consumers); write edges enter it.
            access_node_read_volume = _access_volume(state, node, state.out_edges(node), lambda e: e.dst)
            access_node_write_volume = _access_volume(state, node, state.in_edges(node), lambda e: e.src)
            for (var, (lo, hi, step)) in reversed(range_var_stack):
                access_node_read_volume = _accumulate_volume_over_var(access_node_read_volume, var, lo, hi, step)
                access_node_write_volume = _accumulate_volume_over_var(access_node_write_volume, var, lo, hi, step)

            read += simplify(access_node_read_volume)
            write += simplify(access_node_write_volume)

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
                # Pop in ``finally`` so a failure inside the recursion cannot leak the frame and
                # inflate sibling/enclosing scopes (the leaked range would multiply their volume).
                try:
                    loop_read, loop_write = cfr_volume(cfr, region_volume_map, range_var_stack, detailed_analysis)
                finally:
                    del range_var_stack[-1:]
                region_volume_map[cfr] = (loop_read, loop_write)
            except Exception:
                # Fall back to the (statically estimated) number of executions of the loop body.
                loop_executions = cfr.start_block.executions
                range_var_stack.append(
                    (f"byte_access_loop_range_var_{len(range_var_stack)}", (pystr_to_symbolic(0), loop_executions,
                                                                            pystr_to_symbolic(1))))
                try:
                    inner_read, inner_write = cfr_volume(cfr, region_volume_map, range_var_stack, detailed_analysis)
                finally:
                    del range_var_stack[-1:]
                region_volume_map[cfr] = (inner_read, inner_write)

        elif isinstance(cfr, ConditionalBlock):
            branch_conditions: Dict[AbstractControlFlowRegion, sp.Expr] = {}
            branch_reads = []
            branch_writes = []
            for (condition, branch) in cfr.branches:
                branch_conditions[branch] = (pystr_to_symbolic(condition.as_string)
                                             if condition is not None else pystr_to_symbolic(True))
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
        else:
            # With control-flow regions, branching is handled by control-flow blocks, so the
            # remaining regions form a single path and their volumes can simply be summed.
            reg_read, reg_write = cfr_volume(cfr, region_volume_map, range_var_stack)
            region_volume_map[cfr] = (reg_read, reg_write)

    traversal_q = deque()
    traversal_q.append((control_flow_region.start_block, {}))

    region_reads = pystr_to_symbolic(0)
    region_writes = pystr_to_symbolic(0)
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

    :note: Only structured control flow is supported (loops as ``LoopRegion``, branches as
        ``ConditionalBlock``, no ``break`` / ``continue`` / ``return``). An SDFG with unstructured
        control flow is not analyzed: the analysis warns and returns a zero result.
    :param sdfg: The SDFG to analyze (left unmodified).
    :param optimize: If True, auto-optimize the SDFG copy first to approximate the effect of
                     compiler optimizations; if False, analyze the raw data volume.
    :return: A tuple of ``(read_volume, write_volume)`` in bytes, as symbolic expressions.
    """
    # Deep-copy so the original SDFG is not modified.
    sdfg = deepcopy(sdfg)

    # The analysis only models structured control flow. If the SDFG has a legacy loop or
    # unstructured branching, bail out with a zero result rather than producing a wrong one.
    if has_unstructured_control_flow(sdfg):
        warnings.warn('Memory-volume analysis supports only structured control flow (LoopRegion / '
                      'ConditionalBlock); the SDFG contains a legacy loop or unstructured branch, '
                      'so no result is produced.')
        return pystr_to_symbolic(0), pystr_to_symbolic(0)

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
    # Resolve residual Max/Min (per-connector vs bounding box left where no loop range fixed it).
    read = resolve_size_dominated_minmax(read.subs(static_symbol_mapping))
    write = resolve_size_dominated_minmax(write.subs(static_symbol_mapping))
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
