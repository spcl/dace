# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Simplify derived induction variables in LoopRegions.

For each ``LoopRegion``, classify its induction variables (basic + affine-derived)
using :func:`dace.transformation.passes.analysis.loop_analysis.detect_induction_variables`
and fold derived-IV symbols into their closed-form affine expression at every
use site inside the loop. When the derived-IV's defining interstate-edge
assignment has no remaining consumers after folding, remove it.

This is the DaCe-level analogue of LLVM's ``IndVarSimplify``: fewer intermediate
scalars, cleaner subscripts, and — as a side effect — downstream pattern matchers
(``LoopToMap`` and the parallelization-prep passes) start matching shapes that a
derived scalar previously obscured.

Scope is limited to:
  * LoopRegion induction variables (not Map parameters).
  * Interstate-edge-defined derived IVs (not tasklet-written scalars — folding
    those requires rewriting dataflow, which is out of scope here).
  * The defining assignment must be loop-invariant in its scale and offset;
    guaranteed by the detection pass.
"""

from typing import Any

from dace import SDFG, data, nodes, properties, symbolic
from dace.ordered import OrderedSet
from dace.properties import CodeBlock
from dace.sdfg.narrowing import as_basic, as_expr
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis


@properties.make_properties
@xf.explicit_cf_compatible
class SimplifyInductionVariables(ppl.Pass):
    """Fold affine-derived induction variables into their closed form."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.InterstateEdges | ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.InterstateEdges | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        loops = [n for n, _p in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
        loops.sort(key=loop_nesting_depth, reverse=True)
        total = 0
        # Carried symbols that nested loops increment. Maps symbol name -> per-enclosing-loop
        # increment expression. Populated when a self-referential iedge IV is folded in an inner
        # loop; consumed by the enclosing loop so the outer carry can close too.
        nested_carries: dict[str, tuple[LoopRegion, symbolic.SymbolicType]] = {}
        for loop in loops:
            total += simplify_loop(loop, nested_carries)
        return total or None


def loop_nesting_depth(loop: LoopRegion) -> int:
    depth = 0
    parent = loop.parent_graph
    while parent is not None:
        if isinstance(parent, LoopRegion):
            depth += 1
        parent = parent.parent_graph
    return depth


def loop_trip_count(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """Return the number of iterations of ``loop`` if its bounds are simple enough.

    Mirrors the trip-count computation used by ``InductionVariableSubstitution``.
    """
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    if symbolic.simplify(stride) == 0:
        return None
    return symbolic.simplify(symbolic.int_floor(end - start, stride) + 1)


def self_referential_step(name: str, rhs: str) -> symbolic.SymbolicType | None:
    """If ``rhs`` equals ``name + step`` (or ``name - step``), return the signed
    loop-invariant step. Otherwise return ``None``.
    """
    try:
        lhs_sym = symbolic.pystr_to_symbolic(name)
        rhs_sym = symbolic.pystr_to_symbolic(rhs)
        unsimplified = as_expr(rhs_sym) - as_expr(lhs_sym)
        # A rhs that never names the counter leaves it in the difference, and simplify keeps value.
        if name not in {str(s) for s in rhs_sym.free_symbols} and name in {str(s) for s in unsimplified.free_symbols}:
            return None
        diff = symbolic.simplify(unsimplified)
    except (SyntaxError, TypeError, ValueError):
        return None
    if name in {str(s) for s in as_basic(diff).free_symbols}:
        return None
    return diff


def fold_self_referential_iedge_ivs(
    loop: LoopRegion,
    iv_edge_sites: dict[str, list],
    nested_carries: dict[str, tuple[LoopRegion, symbolic.SymbolicType]],
) -> int:
    """Fold iedge counters of the form ``sym := sym + const`` inside ``loop``.

    Returns the number of symbols folded. For each folded symbol the per-iteration
    net increment is recorded in ``nested_carries`` so any immediately enclosing
    loop can close the outer carry.
    """
    if not iv_edge_sites:
        return 0
    applied = 0
    start = loop_analysis.get_init_assignment(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or stride is None:
        return 0

    for name, edges in list(iv_edge_sites.items()):
        if len(edges) != 1:
            continue
        edge = edges[0]
        if edge.data.condition.as_string not in ("1", "True", "(1)"):
            continue
        if not edge.data.assignments or len(edge.data.assignments) != 1:
            continue
        (lhs, rhs) = next(iter(edge.data.assignments.items()))
        if lhs != name:
            continue
        step = self_referential_step(name, rhs)
        if step is None:
            continue
        # The closed form assumes the increment is loop-invariant. A step that
        # depends on the loop variable (e.g. TSVC s141 ``k = k + j + 1``) is
        # not a constant per-iteration increment and must not be folded here.
        loop_var_name = loop.loop_variable
        if loop_var_name is not None and loop_var_name in {str(s) for s in step.free_symbols}:
            continue
        # Only fold when the counter is dead after the loop. If its post-loop
        # value is consumed, InductionVariableSubstitution owns the rewrite (it
        # materialises the exit value); folding here would drop the update.
        if not symbol_is_dead_outside_loop(loop, name):
            continue
        # Determine whether the increment is before or after the body reads.
        # Post-increment (edge from empty start block): body sees ``sym + (t+1)*step``.
        # Pre-increment (edge to unique empty sink): body sees ``sym + t*step``.
        try:
            start_block = loop.start_block
        except ValueError:
            continue
        norm_iter = symbolic.simplify(
            symbolic.int_floor(symbolic.pystr_to_symbolic(loop.loop_variable) - start, stride)
        )
        src_is_empty_start = edge.src is start_block and isinstance(edge.src, SDFGState) and not edge.src.nodes()
        sinks = [b for b in loop.nodes() if loop.out_degree(b) == 0]
        dst_is_unique_empty_sink = (
            isinstance(edge.dst, SDFGState)
            and not edge.dst.nodes()
            and len(sinks) == 1
            and sinks[0] is edge.dst
            and loop.in_degree(edge.dst) == 1
        )
        if src_is_empty_start:
            body_offset = norm_iter + 1
        elif dst_is_unique_empty_sink:
            body_offset = norm_iter
        else:
            # Mid-body increment: conservatively refuse until a use-side analysis lands.
            continue

        # The symbol inside the loop now represents the value on entry, so the
        # closed form is ``entry_sym + body_offset * step``.
        closed = symbolic.symstr(symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(name)) + body_offset * step))
        edge.data.assignments.pop(name, None)
        loop.replace_dict({name: closed}, replace_keys=False)

        trip = loop_trip_count(loop)
        if trip is not None:
            nested_carries[name] = (loop, symbolic.simplify(trip * step))
        applied += 1

    return applied


def fold_nested_carried_symbols(
    loop: LoopRegion, nested_carries: dict[str, tuple[LoopRegion, symbolic.SymbolicType]]
) -> int:
    """Close the outer carry for symbols incremented by an immediately nested loop.

    After a nested inner loop folded a self-referential counter, this loop's
    iteration variable can be used to express the counter's value as a derived IV.
    """
    # Only consume carries that belong to loops immediately nested inside this one.
    if not has_nested_carry(loop, nested_carries):
        return 0
    immediate_nested = immediately_nested_loops(loop)
    applied = 0
    start = loop_analysis.get_init_assignment(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or stride is None:
        return 0
    loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
    norm_iter = symbolic.simplify(symbolic.int_floor(loop_var - start, stride))

    for name, (src_loop, per_iter) in list(nested_carries.items()):
        if src_loop not in immediate_nested:
            continue
        # Refuse if the symbol is still assigned on an interstate edge of this loop;
        # that would be a direct outer-loop increment (s126) and double-counting.
        if any(name in e.data.assignments for e in loop.all_interstate_edges()):
            continue
        # The symbol's value at the start of this loop is a derived IV: substitute
        # its closed form inside the loop body. The symbol still names the pre-loop
        # value; constant propagation downstream folds the initializer (e.g. ``k = -1``).
        closed = symbolic.symstr(symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(name)) + norm_iter * per_iter))
        loop.replace_dict({name: closed}, replace_keys=False)
        nested_carries.pop(name, None)
        applied += 1

    return applied


def immediately_nested_loops(loop: LoopRegion) -> OrderedSet[LoopRegion]:
    return OrderedSet(n for n in loop.nodes() if isinstance(n, LoopRegion))


def has_nested_carry(loop: LoopRegion, nested_carries: dict[str, tuple[LoopRegion, symbolic.SymbolicType]]) -> bool:
    """Whether a loop immediately nested in ``loop`` left a carried counter for it to close."""
    nested = immediately_nested_loops(loop)
    return any(src_loop in nested for src_loop, _ in nested_carries.values())


def simplify_loop(loop: LoopRegion, nested_carries: dict[str, tuple[LoopRegion, symbolic.SymbolicType]]) -> int:
    # Only fold derived IVs that came from interstate-edge assignments; skip
    # tasklet-derived entries (they refer to data descriptors, not symbols,
    # and folding them requires dataflow rewrites out of scope for v1).
    iv_edge_sites = collect_interstate_iv_sites(loop)
    # Every fold below needs an interstate assignment in the loop or a carry out of a nested loop.
    if not iv_edge_sites:
        if not has_nested_carry(loop, nested_carries):
            return 0

    ivs = loop_analysis.detect_induction_variables(loop)

    applied = 0

    # Fold self-referential iedge IVs (``k := k + step``) that the basic detector
    # rejects because the RHS mentions the LHS. These are common in TSVC kernels
    # (s125/s126) where a scalar counter is incremented each inner iteration and
    # also carried by the outer loop. We must fold them before normal derived-IV
    # folding so the per-iteration increment is visible to the outer-loop pass.
    applied += fold_self_referential_iedge_ivs(loop, iv_edge_sites, nested_carries)

    if not ivs:
        return applied

    # Order derived IVs so that deeper basis chains are substituted first.
    # That way, when we later substitute a shallower IV, the expression we
    # just introduced (which still references the shallower IV) gets rewritten
    # too — leaving every subset in terms of the root loop variable.
    def basis_depth(iv) -> int:
        d = 0
        cur = iv
        while cur.basis is not None:
            d += 1
            cur = cur.basis
        return d

    derived = [iv for iv in ivs.values() if iv.kind == "derived" and iv.name in iv_edge_sites]
    # Safety: only fold when the defining edge is known to execute before every
    # read of the symbol inside the loop. The cheapest conservative check: the
    # edge's source must be the loop's start_block, and the start_block must
    # not itself read the symbol. This catches loop-carried scalars whose
    # assignment sits on a mid-body edge (e.g. TSVC s292 ``im1 = i`` right
    # before the latch, with reads upstream in the body's first state).
    derived = [iv for iv in derived if assignment_dominates_uses(loop, iv.name, iv_edge_sites[iv.name])]
    # A symbol assigned only under a ``ConditionalBlock`` and live past the
    # loop (e.g. an argmax ``index``) is not a per-iteration IV: folding it
    # is unsound and its defining assignment cannot be removed, so the fold
    # never converges against ``ScalarToSymbolPromotion``.
    derived = [
        iv
        for iv in derived
        if not (
            assignment_is_conditional_in_loop(loop, iv_edge_sites[iv.name])
            and not symbol_is_dead_outside_loop(loop, iv.name)
        )
    ]
    derived.sort(key=basis_depth, reverse=True)

    for iv in derived:
        name = iv.name
        basis = iv.basis
        if basis is None:
            continue
        # A container sized by ``name`` that the fold below does not rewrite keeps reading it.
        dead = symbol_is_dead_outside_loop(loop, name) and not loop_containers_use_symbol(loop, name)
        if not dead and not loop_reads_symbol(loop, name):
            continue  # already folded: only the kept assignment is left, so a rerun must not report a change
        # Build the replacement: scale * basis + offset, parenthesized so later
        # string-based substitutions don't capture adjacent operators.
        replacement = f"({iv.scale} * ({basis.name}) + ({iv.offset}))"

        # Substitute every use of ``name`` inside the loop. ``replace_keys=False``
        # preserves the defining assignment on the interstate edge so we can
        # decide afterwards whether it's dead.
        loop.replace_dict({name: replacement}, replace_keys=False)
        fold_into_loop_local_descriptors(loop, name, replacement)

        # Decide whether the defining assignment is dead.
        if dead:
            for edge in iv_edge_sites[name]:
                edge.data.assignments.pop(name, None)
            # If a data descriptor with the same name exists and is now
            # unreferenced anywhere in the top-level SDFG, remove it. This
            # handles the case where the frontend reserved a scalar descriptor
            # to mirror the symbol.
            remove_dead_scalar(loop, name)

        applied += 1

    # A symbol carried by an immediately-nested inner loop can become a derived
    # IV of this loop once the inner loop's net per-iteration increment is known.
    applied += fold_nested_carried_symbols(loop, nested_carries)

    return applied


def fold_into_loop_local_descriptors(loop: LoopRegion, name: str, replacement: str) -> None:
    """Substitute ``name`` in the descriptors of the transients only ``loop`` accesses, which it allocates.

    ``LoopRegion.replace_dict`` leaves the SDFG's descriptors alone, so a view whose extent names the folded
    variable kept it read inside the loop and the fold reported a change on every rerun (mixed_precision_ir).
    """
    sdfg = loop.sdfg
    inside = OrderedSet(loop.states())
    accessed = OrderedSet(node.data for state in inside for node in state.data_nodes())
    for state in sdfg.states():
        if state not in inside:
            accessed -= OrderedSet(node.data for node in state.data_nodes())
    for container in accessed:
        desc = sdfg.arrays[container]
        if desc.transient and name in {str(sym) for sym in desc.free_symbols}:
            replace_properties_dict(desc, {name: replacement}, sdfg=sdfg)


def collect_interstate_iv_sites(loop: LoopRegion) -> dict[str, list]:
    """Map each IV-candidate symbol name to the list of interstate edges that
    carry an assignment for it inside ``loop``.
    """
    sites: dict[str, list] = {}
    for edge in loop.all_interstate_edges():
        for name in edge.data.assignments:
            sites.setdefault(name, []).append(edge)
    return sites


def assignment_is_conditional_in_loop(loop: LoopRegion, edges: list) -> bool:
    """Whether the symbol's assignment is branch-guarded inside the loop.

    :param loop: The enclosing ``LoopRegion``.
    :param edges: The interstate edges that assign the symbol.
    :returns: ``True`` iff a defining edge has a ``ConditionalBlock``
        ancestor below ``loop`` (it does not run on every iteration).
    """
    for edge in edges:
        for endpoint in (edge.src, edge.dst):
            graph = endpoint.parent_graph
            while graph is not None and graph is not loop:
                if isinstance(graph, ConditionalBlock):
                    return True
                graph = graph.parent_graph
    return False


def assignment_dominates_uses(loop: LoopRegion, name: str, edges: list) -> bool:
    """Return True iff the assignment(s) to ``name`` are guaranteed to execute
    before every read of ``name`` within one iteration of the loop.

    Implementation: iteration-graph reachability. Remove the defining edge(s)
    from the loop's CFG and BFS from ``start_block``. If any state reachable
    without the edge reads the symbol, then some read sees the previous
    iteration's (or initial) value — folding would be semantically wrong.

    This catches loop-carried scalars whose assignment sits on a mid-body edge
    (e.g. TSVC s292's ``im1 = i`` right before the latch, with reads upstream
    in the body's first state) while still accepting chained derived-IV
    assignments in straight-line bodies (``j = 2*i + 1`` on entry, then
    ``k = 3*j + 2`` on the next edge).
    """
    if len(edges) != 1:
        return False
    edge = edges[0]
    try:
        start = loop.start_block
    except ValueError:
        return False

    # Collect read sites of ``name`` inside the loop.
    readers = OrderedSet(state for state in loop.states() if state_reads_symbol(state, name))
    if not readers:
        return True  # Nothing to read — substitution is vacuously correct.

    # BFS from start_block over the loop's CFG, excluding the defining edge.
    # If any reader is reachable, the assignment doesn't dominate that read.
    visited = OrderedSet([start])
    stack = [start]
    excluded = (edge.src, edge.dst)
    while stack:
        node = stack.pop()
        if node in readers:
            return False
        parent = node.parent_graph
        if parent is None or parent is not loop:
            # Only traverse within the loop's own CFG (not nested regions).
            continue
        for out_edge in parent.out_edges(node):
            if (out_edge.src, out_edge.dst) == excluded:
                continue
            dst = out_edge.dst
            if dst in visited:
                continue
            visited.add(dst)
            stack.append(dst)
    # If start_block itself reads the symbol, it was already flagged above
    # (start is popped first; if it's a reader we return False).
    return True


def state_reads_symbol(state: SDFGState, name: str) -> bool:
    """Whether a memlet subset or a tasklet's code in ``state`` names ``name``."""
    for edge in state.edges():
        for subset in (edge.data.subset, edge.data.other_subset):
            if subset is not None and name in {str(sym) for sym in subset.free_symbols}:
                return True
    return any(
        isinstance(node, nodes.Tasklet) and name_in_expr_string(name, node.code.as_string) for node in state.nodes()
    )


def header_code(block: Any) -> list[CodeBlock]:
    """The code a loop or a conditional block evaluates outside its body."""
    if isinstance(block, LoopRegion):
        codes = [block.init_statement, block.update_statement, block.loop_condition]
    else:
        codes = [condition for condition, _ in block.branches]
    return [code for code in codes if code is not None]


def loop_containers_use_symbol(loop: LoopRegion, name: str) -> bool:
    """Whether a container accessed inside ``loop`` names ``name`` in its descriptor (a shape, a stride)."""
    sdfg = loop.sdfg
    return any(
        name in {str(s) for s in node.desc(sdfg).used_symbols(all_symbols=True)}
        for state in loop.states()
        for node in state.data_nodes()
    )


def loop_reads_symbol(loop: LoopRegion, name: str) -> bool:
    """Whether ``name`` is read inside ``loop`` where a fold rewrites it, nested SDFGs aside: by a memlet or a
    node, by a loop or branch header, or by an interstate edge's condition or assigned value. An assignment
    target is a write, not a read, and a container descriptor is not rewritten (see
    :func:`loop_containers_use_symbol`)."""
    if any(name in edge.data.read_symbols() for edge in loop.all_interstate_edges()):
        return True
    headers = [loop]
    for block in loop.all_control_flow_blocks():
        if isinstance(block, SDFGState):
            if any(name in {str(s) for s in e.data.free_symbols} for e in block.edges()) or any(
                name in {str(s) for s in n.free_symbols} for n in block.nodes() if not isinstance(n, nodes.AccessNode)
            ):
                return True
        elif isinstance(block, (LoopRegion, ConditionalBlock)):
            headers.append(block)
    return any(name in code.get_free_symbols() for block in headers for code in header_code(block))


def symbol_is_dead_outside_loop(loop: LoopRegion, name: str) -> bool:
    """Return True iff ``name`` is not referenced by any state/edge/memlet in
    the top-level SDFG outside this loop."""
    sdfg = loop.sdfg
    if sdfg is None:
        return False

    # Nodes inside the loop itself are already substituted; we only need to
    # check the rest of the SDFG.
    loop_states = OrderedSet(loop.states())
    loop_edges = OrderedSet(id(edge) for edge in loop.all_interstate_edges())

    # Check interstate-edge assignments and conditions everywhere in the SDFG.
    for edge in sdfg.all_interstate_edges():
        if id(edge) in loop_edges:
            continue
        # Assignments that reference ``name`` in their RHS are reads of the value.
        for rhs in edge.data.assignments.values():
            if name_in_expr_string(name, rhs):
                return False
        if name_in_expr_string(name, edge.data.condition.as_string):
            return False
        # Note: an LHS assignment to ``name`` outside the loop is a write, not a
        # read of the loop-carried value (e.g. the pre-loop initializer), so it
        # does not make the symbol live here.

    return not any(state_reads_symbol(state, name) for state in sdfg.states() if state not in loop_states)


def name_in_expr_string(name: str, expr: str) -> bool:
    """Approximate free-symbol check by parsing ``expr`` and inspecting free
    symbols. Falls back to a conservative ``True`` on parse failure."""
    if not expr:
        return False
    try:
        return name in {str(sym) for sym in symbolic.pystr_to_symbolic(expr).free_symbols}
    except (SyntaxError, TypeError, ValueError):
        return True


def remove_dead_scalar(loop: LoopRegion, name: str) -> None:
    sdfg = loop.sdfg
    if sdfg is None or name not in sdfg.arrays:
        return
    # Only remove transient scalars with no remaining AccessNodes.
    desc = sdfg.arrays[name]
    if not isinstance(desc, data.Scalar) or not desc.transient:
        return
    if any(node.data == name for state in sdfg.states() for node in state.data_nodes()):
        return
    sdfg.remove_data(name, validate=False)
