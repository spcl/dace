# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Various analyses concerning LopoRegions, and utility functions to get information about LoopRegions for other passes.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import sympy

from dace import symbolic
from dace.frontend.python import astutils
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
)

if TYPE_CHECKING:
    # Import-time only: ``dace.sdfg.state`` reaches back into this module (LoopRegion.new_symbols does
    # the import inside the function body, "avoid cyclic import"), so importing the SDFG class here at
    # module scope would close that cycle from the other side.
    from dace.properties import CodeBlock
    from dace.sdfg.sdfg import SDFG


def loop_header_signature(loop: LoopRegion) -> str:
    """The iterator and range of ``loop``, as text."""
    codes = (loop.init_statement, loop.loop_condition, loop.update_statement)
    return "|".join([loop.loop_variable or "", *(code.as_string if code is not None else "" for code in codes)])


def body_local_symbols(loop: LoopRegion) -> set[str]:
    """The symbols ``loop``'s body defines: interstate assignments and the iterators of the loops inside it."""
    local = {name for edge in loop.all_interstate_edges() for name in edge.data.assignments}
    local |= {block.loop_variable for block in loop.all_control_flow_regions() if isinstance(block, LoopRegion)}
    local.discard(loop.loop_variable)
    return local


def loop_signature(loop: LoopRegion) -> str:
    """A digest of ``loop``'s header and of the elements one iteration reads and writes: per non-transient container,
    the subsets of its top-level accesses (map scopes and nested SDFGs through their propagated memlets)
    and its WCR ops. A dimension that names a symbol the body defines is its full extent, so inlining a nested body,
    renaming its blocks or moving a read through a body symbol keeps the digest; a rewrite that makes an iteration
    touch another element does not."""
    from dace import subsets  # Avoid import loop

    sdfg = loop.sdfg
    local = body_local_symbols(loop)
    states = list(loop.states())
    shared = {name for name, desc in sdfg.arrays.items() if not desc.transient}

    def per_iteration(name: str, subset) -> str:
        if not isinstance(subset, subsets.Range):
            return str(subset)
        shape = sdfg.arrays[name].shape
        ranges = [
            (0, extent - 1, 1) if {str(free) for free in subsets.Range([rng]).free_symbols} & local else rng
            for extent, rng in zip(shape, subset.ranges)
        ]
        return str(subsets.Range(ranges))

    lines = []
    for state in states:
        scopes = state.scope_dict()
        for node in state.data_nodes():
            if node.data not in shared or scopes[node] is not None:
                continue
            lines.extend(
                f"W {node.data} {per_iteration(node.data, edge.data.get_dst_subset(edge, state))} {edge.data.wcr}"
                for edge in state.in_edges(node)
                if not edge.data.is_empty()
            )
            lines.extend(
                f"R {node.data} {per_iteration(node.data, edge.data.get_src_subset(edge, state))}"
                for edge in state.out_edges(node)
                if not edge.data.is_empty()
            )
    lines.extend(
        f"R {memlet.data} {per_iteration(memlet.data, memlet.subset)}"
        for edge in loop.all_interstate_edges()
        for memlet in edge.data.get_read_memlets(sdfg.arrays, include_scalars=True)
        if memlet.data in shared
    )
    return hashlib.sha1("\n".join([loop_header_signature(loop), *sorted(set(lines))]).encode()).hexdigest()


def mark_known_parallel(loop: LoopRegion, conflict_free: Iterable[str] = ()) -> None:
    """Records that ``loop``'s iterations are independent as it stands now, and that its writes to the
    ``conflict_free`` containers never collide under its current header."""
    loop.known_parallel = True
    loop.known_parallel_signature = loop_signature(loop)
    names = sorted(set(conflict_free))
    if names:
        loop.conflict_free_writes = names
        loop.conflict_free_header = loop_header_signature(loop)


def is_known_parallel(loop: LoopRegion) -> bool:
    """Whether ``loop`` is known-parallel and unchanged since it was marked: a pass that rewrites its body or header
    (an interchange swapping headers, a WCR turned into an explicit update) leaves it to the dependence test."""
    return loop.known_parallel and loop.known_parallel_signature == loop_signature(loop)


def conflict_free_writes(loop: LoopRegion) -> frozenset[str]:
    """The containers whose writes no two iterations of ``loop`` collide on, while its header is the one recorded."""
    if not loop.conflict_free_writes or loop.conflict_free_header != loop_header_signature(loop):
        return frozenset()
    return frozenset(loop.conflict_free_writes)


def symbol_use_sites(sdfg: "SDFG") -> tuple[dict[str, set[int]], set[str]]:
    """One walk of ``sdfg`` indexing where each symbol is used: name -> ``id()`` of every block and
    inter-state edge that references it, plus the set of names any data descriptor's shape or strides
    mention (anywhere in the SDFG tree, since a nested descriptor is materialised just as eagerly).

    Answering "is this symbol used outside that loop?" costs a full traversal, so a caller asking it per
    loop pays it per loop. Build the index once and answer every loop from it.

    Each block is enumerated individually, hence a REGION is asked only for the symbols it uses on
    itself (``with_contents=False`` -- its condition / init / update); its contents arrive as their own
    blocks. A state must be asked WITH contents: ``SDFGState.used_symbols(with_contents=False)`` returns
    the empty set, which would silently hide every real use.
    """
    uses: dict[str, set[int]] = {}
    for block in sdfg.all_control_flow_blocks():
        with_contents = not isinstance(block, AbstractControlFlowRegion)
        for name in block.used_symbols(all_symbols=True, with_contents=with_contents):
            uses.setdefault(name, set()).add(id(block))
    for edge in sdfg.all_interstate_edges():
        for name in set(edge.data.free_symbols) | set(edge.data.assignments):
            uses.setdefault(name, set()).add(id(edge))
    descriptor_symbols: set[str] = set()
    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            descriptor_symbols.update(str(s) for s in desc.free_symbols)
    return uses, descriptor_symbols


def symbol_used_outside(
    name: str, inside_sites: set[int], use_sites: dict[str, set[int]], descriptor_symbols: set[str]
) -> bool:
    """Whether ``name`` is read or written at any site not in ``inside_sites``.

    ``inside_sites`` holds the ``id()`` of every block and inter-state edge that counts as "inside";
    ``use_sites`` / ``descriptor_symbols`` come from :func:`symbol_use_sites`. A name a data descriptor
    mentions is used outside by definition -- the descriptor is materialised wherever the array is.
    """
    if name in descriptor_symbols:
        return True
    return any(site not in inside_sites for site in use_sites.get(name, ()))


def counter_used_outside_loop(
    name: str,
    loop: LoopRegion,
    sdfg: "SDFG",
    use_sites: dict[str, set[int]] | None = None,
    descriptor_symbols: set[str] | None = None,
    rebinding: dict[str, list[LoopRegion]] | None = None,
) -> bool:
    """Whether ``name`` is read or written anywhere outside ``loop``.

    A LoopRegion counter is NOT scoped to its loop the way a map parameter is scoped to its map: DaCe
    leaks its final value to subsequent blocks, and the default ``eager`` declaration placement hoists
    its ``int64_t i;`` to the top of the generated function. So any question of the form "may I treat
    this counter as loop-local?" -- scoping its declaration into the ``for``-init clause, dropping its
    value at a nesting boundary -- has to ask this, not assume it.

    Pass ``use_sites`` / ``descriptor_symbols`` from :func:`symbol_use_sites` when asking about several
    loops of the same SDFG; omitted, the index is built here for this one query. ``rebinding`` is
    :func:`rebinding_loops` of ``sdfg``, likewise.
    """
    if use_sites is None or descriptor_symbols is None:
        use_sites, descriptor_symbols = symbol_use_sites(sdfg)
    inside = {id(loop)}
    # Another loop that binds the same counter outright reads its own value, not this loop's
    others = (
        [b for b in sdfg.all_control_flow_blocks() if rebinds_counter(b, name)]
        if rebinding is None
        else rebinding.get(name, [])
    )
    for other in [loop] + [b for b in others if b is not loop]:
        inside.add(id(other))
        inside.update(id(block) for block in other.all_control_flow_blocks())
        inside.update(id(edge) for edge in other.all_interstate_edges())
    return symbol_used_outside(name, inside, use_sites, descriptor_symbols)


def rebinding_loops(sdfg: "SDFG") -> dict[str, list[LoopRegion]]:
    """Per counter name, the loops of ``sdfg`` that bind it outright (:func:`rebinds_counter`), in block order."""
    found: dict[str, list[LoopRegion]] = {}
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, LoopRegion) and rebinds_counter(block, block.loop_variable):
            found.setdefault(block.loop_variable, []).append(block)
    return found


def rebinds_counter(block: ControlFlowBlock, name: str) -> bool:
    """Whether ``block`` is a loop over ``name`` whose init sets it from scratch (``i = 0``, not ``i = i + 1``)."""
    if not isinstance(block, LoopRegion) or block.loop_variable != name or not block.init_statement:
        return False
    init = get_init_assignment(block)
    return init is not None and name not in symbolic.free_symbols_and_functions(init)


def loop_jumps(loop: LoopRegion) -> list[ControlFlowBlock]:
    """The ``break`` and ``continue`` blocks that target ``loop``.

    A jump targets its innermost enclosing loop, so the walk descends through conditional branches
    and plain regions but not into nested loops.
    """
    jumps: list[ControlFlowBlock] = []
    stack = list(loop.nodes())
    while stack:
        block = stack.pop()
        if isinstance(block, (BreakBlock, ContinueBlock)):
            jumps.append(block)
        elif isinstance(block, ConditionalBlock):
            for condition, branch in block.branches:
                stack.extend(branch.nodes())
        elif isinstance(block, ControlFlowRegion) and not isinstance(block, LoopRegion):
            stack.extend(block.nodes())
    return jumps


def get_loop_end(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """
    Parse a loop region to identify the end value of the iteration variable under normal loop termination (no break).
    """
    if loop.loop_variable is None or loop.loop_variable == "":
        return None
    end: symbolic.SymbolicType | None = None
    a = sympy.Wild("a")
    condition = symbolic.pystr_to_symbolic(loop.loop_condition.as_string)
    itersym = symbolic.pystr_to_symbolic(loop.loop_variable)
    match = condition.match(itersym < a)
    if match:
        end = match[a] - 1
    if end is None:
        match = condition.match(itersym <= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(itersym > a)
        if match:
            end = match[a] + 1
    if end is None:
        match = condition.match(itersym >= a)
        if match:
            end = match[a]
    return end


def get_assignments(statement) -> dict[str, str]:
    """
    All ``name = expression`` assignments of a loop init or update statement (a code block may hold several
    statements, e.g. the loop variable plus loop-carried cursor symbols), as a name-to-expression dictionary.
    Statements that are not simple assignments are ignored.
    """
    if statement is None:
        return {}
    codes = statement.code if isinstance(statement.code, list) else [statement.code]
    assignments: dict[str, str] = {}
    for code in codes:
        visitor = astutils.FindAssignment()
        visitor.visit(code)
        assignments.update(visitor.assignments)
    return assignments


def assignment_text(stmt: Optional["CodeBlock"], variable: str | None) -> str | None:
    """
    The right-hand side text a loop's init or update statement assigns to ``variable``.

    :param stmt: The statement, e.g. ``loop.init_statement``.
    :param variable: The assigned name, e.g. ``loop.loop_variable``.
    :return: ``None`` without a statement, when a name is assigned more than once, or when ``variable`` is not.
    """
    if stmt is None:
        return None
    assignments: dict[str, str] = {}
    for code in stmt.code if isinstance(stmt.code, list) else [stmt.code]:
        visitor = astutils.FindAssignment()
        visitor.visit(code)
        if visitor.multiple:
            return None
        for assign in visitor.assignments:
            if assign in assignments:
                return None
            assignments[assign] = visitor.assignments[assign]
    return assignments.get(variable)


def get_init_assignment(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """
    Parse a loop region's init statement to identify the exact init assignment expression.
    """
    text = assignment_text(loop.init_statement, loop.loop_variable)
    return None if text is None else symbolic.pystr_to_symbolic(text)


def get_update_assignment(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """
    Parse a loop region's update statement to identify the exact update assignment expression.
    """
    text = assignment_text(loop.update_statement, loop.loop_variable)
    return None if text is None else symbolic.pystr_to_symbolic(text)


def get_loop_stride(loop: LoopRegion) -> symbolic.SymbolicType | None:
    update_assignment = get_update_assignment(loop)
    if update_assignment:
        return update_assignment - symbolic.pystr_to_symbolic(loop.loop_variable)
    return None


def _provably_le(a: symbolic.SymbolicType, b: symbolic.SymbolicType, facts: symbolic.Facts) -> bool:
    """Prove ``a <= b`` under ``facts`` soundly, returning ``False`` when it cannot be decided (never a guess)."""
    return symbolic.provably_le(sympy.sympify(a), sympy.sympify(b), facts)


def loop_provably_at_most_one_iteration(loop: LoopRegion, facts: symbolic.Facts | None = None) -> bool:
    """Whether ``loop`` provably runs at most once (zero or one iterations).

    Such a loop carries no cross-iteration dependence by construction, so it is trivially DOALL --
    a ``LoopToMap`` can map it without any dependence analysis. Restricted to the unit ascending
    stride so the inclusive trip count is exactly ``end - start + 1``; the loop runs at most once iff
    ``end <= start``. Conservative: returns ``False`` whenever the bound cannot be proven.

    :param facts: ``loop.sdfg.facts()``, when the caller holds it.
    """
    start = get_init_assignment(loop)
    end = get_loop_end(loop)  # inclusive
    step = get_loop_stride(loop)
    if start is None or end is None or step is None:
        return False
    if facts is None:
        # A loop not yet placed in an SDFG has no facts beyond its own bounds
        facts = loop.sdfg.facts() if loop.sdfg is not None else symbolic.Facts.none()
    if symbolic.simplify(step, facts) != 1:
        return False
    return _provably_le(symbolic.simplify(end, facts), symbolic.simplify(start, facts), facts)


def loop_provably_at_least_one_iteration(loop: LoopRegion) -> bool:
    """Whether ``loop`` provably runs at least one iteration.

    Needed by any must-def reasoning that wants to carry a write out of a loop body: a write
    inside a loop that may run zero times defines nothing after the loop. The mirror image of
    :func:`loop_provably_at_most_one_iteration`, and conservative in the same direction --
    ``False`` whenever the entry test cannot be decided.

    Decided under the facts of the loop's SDFG: an extent known nonnegative gives ``N >= 0``, not ``N >= 1``,
    so the ubiquitous ``for i in range(N)`` is correctly REFUSED; only a concrete (or otherwise
    provably nonempty) bound is accepted.

    :param loop: The loop region to test.
    :returns: ``True`` only if at least one iteration is proven to execute.
    """
    # An inverted (do-while) loop runs its body before the condition is ever observed, so the
    # first iteration is unconditional whatever the bound says.
    if loop.inverted:
        return True
    start = get_init_assignment(loop)
    end = get_loop_end(loop)  # inclusive bound of the entry condition
    step = get_loop_stride(loop)
    if start is None or end is None or step is None:
        return False
    facts = loop.sdfg.facts()
    step = symbolic.simplify(step, facts)
    # The direction of the entry test follows the sign of the stride, so an unknown or zero stride
    # decides nothing.
    if not step.is_number or step == 0:
        return False
    start = symbolic.simplify(start, facts)
    end = symbolic.simplify(end, facts)
    return _provably_le(start, end, facts) if step > 0 else _provably_le(end, start, facts)


@dataclass(frozen=True)
class InductionVariable:
    """
    Record describing how a symbol evolves across iterations of a LoopRegion.

    Basic and affine-derived IVs share the same shape: ``start`` is the value
    at iteration 0 and ``step`` is the per-iteration increment, pre-flattened
    so consumers do not need to walk a basis chain. For a derived IV
    ``d = scale * basis + offset`` we store:

        start = scale * basis.start + offset
        step  = scale * basis.step

    The ``basis``, ``scale`` and ``offset`` fields are retained for diagnostic
    inspection but callers should generally rely on ``start`` / ``step``.
    """

    name: str
    start: symbolic.SymbolicType
    step: symbolic.SymbolicType
    loop: LoopRegion
    kind: str  # 'basic' | 'derived'
    basis: Optional["InductionVariable"] = None
    scale: symbolic.SymbolicType | None = None
    offset: symbolic.SymbolicType | None = None


def affine_in_iv(
    expr: symbolic.SymbolicType,
    ivs: dict[str, "InductionVariable"],
    invariant_syms: set[str] | None = None,
) -> tuple[str | None, symbolic.SymbolicType, symbolic.SymbolicType] | None:
    """
    If ``expr`` equals ``scale * iv + offset`` for some iv in ``ivs`` with
    loop-invariant ``scale`` and ``offset``, return a triple
    ``(iv_name, scale, offset)``. A pure-invariant expression (no IV
    referenced) returns ``(None, 0, expr)`` so callers can treat constants
    uniformly — this matches the zero-scale convention used by LLMR's
    subscript matcher at ``loop_local_memory_reduction.py:172-187``.

    Returns ``None`` if the expression is not affine in any single IV, or if
    ``scale`` / ``offset`` reference another IV.

    When ``invariant_syms`` is provided, ``scale`` and ``offset`` must have
    all free symbols inside that set; otherwise the match is rejected. This
    lets a caller restrict which outer-scope symbols are acceptable.
    """
    if expr is None:
        return None
    e = symbolic.pystr_to_symbolic(expr)
    iv_names = set(ivs)
    # DaCe uses its own ``symbolic.symbol`` subclass; sympy treats it as a
    # distinct Symbol from ``sympy.Symbol(name)``, so match by name string.
    sym_by_name = {str(s): s for s in e.free_symbols}
    free = set(sym_by_name)

    referenced = free & iv_names
    if not referenced:
        if invariant_syms is not None and not (free <= invariant_syms):
            return None
        return (None, symbolic.pystr_to_symbolic(0), e)

    for iv_name in referenced:
        iv_sym = sym_by_name[iv_name]
        # ``sympy.simplify`` is a sledgehammer for deciding this: reaching it through a comparison
        # sends it into ``equals``/``is_constant``, which samples numerically via mpmath PSLQ and
        # dominates the whole IV pass. Distribution alone settles almost every affine split, so
        # ``expand`` screens first and ``simplify`` is only consulted when the cheap form still
        # leaves the IV in the scale or the remainder.
        #
        # The screen decides only WHETHER to fold. What actually gets folded is always recomputed
        # with ``simplify`` below, because ``scale``/``offset`` are substituted into the SDFG
        # verbatim -- an expanded-but-unsimplified form would be equal in value yet different in
        # text, changing emitted subsets for every fold this pass performs.
        scale = offset = None
        screened_by_expand = False
        # Differentiate once; both normalize forms and the screened_by_expand recompute reuse it.
        try:
            raw_diff = sympy.diff(e, iv_sym)
        except Exception:
            continue
        for normalize in (sympy.expand, symbolic.simplify):
            try:
                cand_scale = normalize(raw_diff)
                cand_offset = normalize(e - cand_scale * iv_sym)
            except Exception:
                break
            cand_scale_free = {str(s) for s in cand_scale.free_symbols}
            cand_offset_free = {str(s) for s in cand_offset.free_symbols}
            if cand_scale_free & iv_names or iv_name in cand_offset_free or cand_offset_free & iv_names:
                # An IV the expression mentions inside a function application -- an array subscript
                # ``ztp1[i - 1, j - 1]``, an ``int_floor`` -- differentiates to an UNEVALUATED
                # ``Subs``/``Derivative`` node, which is sympy's way of saying it has no derivative
                # to give. ``simplify`` cannot turn that into an IV-free scale (its ``doit`` only
                # rewrites ``Subs(Derivative(f(x), x), x, a)`` to ``Derivative(f(a), a)``, still
                # naming the IV), so the refusal above is already final and the second round is dead
                # work on the most expensive input there is. CLOUDSC: 615 of these reach the round
                # and not one ever produced a split ``expand`` had not already refused.
                if normalize is sympy.expand and (
                    cand_scale.has(sympy.Subs, sympy.Derivative) or cand_offset.has(sympy.Subs, sympy.Derivative)
                ):
                    break
                continue
            scale, offset = cand_scale, cand_offset
            screened_by_expand = normalize is sympy.expand
            break
        if scale is None:
            continue
        if screened_by_expand:
            try:
                scale = symbolic.simplify(raw_diff)
                offset = symbolic.simplify(e - scale * iv_sym)
            except Exception:
                continue
        scale_free = {str(s) for s in scale.free_symbols}
        offset_free = {str(s) for s in offset.free_symbols}
        if scale_free & iv_names or iv_name in offset_free or offset_free & iv_names:
            continue
        if invariant_syms is not None:
            if not ((scale_free | offset_free) <= invariant_syms):
                continue
        return (iv_name, scale, offset)

    return None


def _collect_tasklet_derived_ivs(loop: LoopRegion, pending: dict[str, str]) -> None:
    """
    Populate ``pending`` with single-assignment Python tasklets in the loop's
    start state that write to a scalar data descriptor. The key is the data
    name (not the tasklet connector). Consumers that want to substitute must
    understand this refers to a data descriptor, not a DaCe symbol.
    """
    # Local imports to avoid import cycles at module load.
    from dace import dtypes
    from dace.sdfg import nodes
    from dace.sdfg.state import SDFGState

    try:
        start = loop.start_block
    except ValueError:
        return
    if not isinstance(start, SDFGState):
        return
    for n in start.nodes():
        if not isinstance(n, nodes.Tasklet):
            continue
        if n.language != dtypes.Language.Python:
            continue
        try:
            if n.side_effects:
                continue
        except AttributeError:
            pass
        out_edges = list(start.out_edges(n))
        if len(out_edges) != 1:
            continue
        oe = out_edges[0]
        if not isinstance(oe.dst, nodes.AccessNode):
            continue
        if oe.data is not None and oe.data.wcr is not None:
            continue
        out_data = oe.dst.data
        # Must be the only writer to that data anywhere in the loop.
        writers = 0
        for state in loop.states():
            for nn in state.data_nodes():
                if nn.data == out_data and state.in_degree(nn) > 0:
                    writers += 1
        if writers != 1:
            continue
        code = n.code.code
        if not isinstance(code, list) or len(code) != 1:
            continue
        stmt = code[0]
        if not isinstance(stmt, astutils.ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], astutils.ast.Name):
            continue
        rhs_str = astutils.unparse(stmt.value)
        pending.setdefault(out_data, rhs_str)


def detect_induction_variables(loop: LoopRegion) -> dict[str, InductionVariable]:
    """
    Classify induction variables of ``loop``: the loop variable itself as a
    basic IV, plus any symbol assigned on an interstate edge within the loop,
    or scalar data written by a single-assign Python tasklet in the loop's
    start state, whose RHS is an affine function of an already-classified IV
    with loop-invariant scale and offset.

    Returns an empty dict if the loop cannot be classified — missing
    ``loop_variable``, missing ``init_statement`` / ``update_statement``, or a
    self-referential step such as ``i = i * 2``.

    Detection is syntactic: it does not reason about trip counts or early
    exits. Callers that need trip-count accuracy should combine this with
    ``get_loop_end``.
    """
    ivs: dict[str, InductionVariable] = {}
    if not loop.loop_variable:
        return ivs
    start = get_init_assignment(loop)
    step = get_loop_stride(loop)
    if start is None or step is None:
        return ivs
    if loop.loop_variable in {str(s) for s in symbolic.pystr_to_symbolic(step).free_symbols}:
        return ivs

    ivs[loop.loop_variable] = InductionVariable(
        name=loop.loop_variable,
        start=start,
        step=step,
        loop=loop,
        kind="basic",
    )

    # Collect candidate derived IV assignments.
    pending: dict[str, str] = {}
    rejected: set[str] = set()
    for e in loop.all_interstate_edges():
        for name, rhs in e.data.assignments.items():
            if name == loop.loop_variable:
                continue
            if name in pending and pending[name] != rhs:
                # Multiple conflicting assignments to the same name — cannot
                # classify conservatively.
                rejected.add(name)
                continue
            pending[name] = rhs
    for name in rejected:
        pending.pop(name, None)

    _collect_tasklet_derived_ivs(loop, pending)

    changed = True
    while changed and pending:
        changed = False
        for name in list(pending):
            rhs = pending[name]
            try:
                rhs_sym = symbolic.pystr_to_symbolic(rhs)
            except Exception:
                pending.pop(name)
                continue
            # Reject self-referential assignments like ``j = j + i``: the RHS
            # must not mention the name being classified.
            if name in {str(s) for s in symbolic.pystr_to_symbolic(rhs_sym).free_symbols}:
                pending.pop(name)
                continue
            result = affine_in_iv(rhs_sym, ivs)
            if result is None:
                pending.pop(name)
                continue
            basis_name, scale, offset = result
            if basis_name is None:
                # Pure invariant — not an IV.
                pending.pop(name)
                continue
            basis_iv = ivs[basis_name]
            ivs[name] = InductionVariable(
                name=name,
                start=scale * basis_iv.start + offset,
                step=scale * basis_iv.step,
                loop=loop,
                kind="derived",
                basis=basis_iv,
                scale=scale,
                offset=offset,
            )
            pending.pop(name)
            changed = True

    return ivs
