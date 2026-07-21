# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Various analyses concerning LopoRegions, and utility functions to get information about LoopRegions for other passes.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
from dace.frontend.python import astutils

import sympy

from dace import symbolic
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion


def counter_used_outside_loop(name: str, loop: LoopRegion, sdfg: SDFG) -> bool:
    """Whether ``name`` is read or written anywhere outside ``loop``.

    A LoopRegion counter is NOT scoped to its loop the way a map parameter is scoped to its map: DaCe
    leaks its final value to subsequent blocks, and the default ``eager`` declaration placement hoists
    its ``int64_t i;`` to the top of the generated function. So any question of the form "may I treat
    this counter as loop-local?" -- scoping its declaration into the ``for``-init clause, dropping its
    value at a nesting boundary -- has to ask this, not assume it.

    Every block of the SDFG is enumerated individually, hence a REGION is asked only for the symbols it
    uses on itself (``with_contents=False`` -- its condition / init / update); its contents arrive as
    their own blocks. A state must be asked WITH contents: ``SDFGState.used_symbols(with_contents=False)``
    returns the empty set, which would silently hide every real use.
    """
    inside = {id(loop)} | {id(block) for block in loop.all_control_flow_blocks()}
    for block in sdfg.all_control_flow_blocks():
        if id(block) in inside:
            continue
        with_contents = not isinstance(block, AbstractControlFlowRegion)
        if name in block.used_symbols(all_symbols=True, with_contents=with_contents):
            return True
    inside_edges = {id(edge) for edge in loop.all_interstate_edges()}
    for edge in sdfg.all_interstate_edges():
        if id(edge) in inside_edges:
            continue
        if name in edge.data.free_symbols or name in edge.data.assignments:
            return True
    # A descriptor whose shape/strides mention the counter is materialised outside the loop.
    for desc in sdfg.arrays.values():
        if name in {str(s) for s in desc.free_symbols}:
            return True
    return False


def get_loop_end(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    """
    Parse a loop region to identify the end value of the iteration variable under normal loop termination (no break).
    """
    if loop.loop_variable is None or loop.loop_variable == '':
        return None
    end: Optional[symbolic.SymbolicType] = None
    a = sympy.Wild('a')
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


def get_init_assignment(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    """
    Parse a loop region's init statement to identify the exact init assignment expression.
    """
    init_stmt = loop.init_statement
    if init_stmt is None:
        return None

    init_codes_list = init_stmt.code if isinstance(init_stmt.code, list) else [init_stmt.code]
    assignments: Dict[str, str] = {}
    for code in init_codes_list:
        visitor = astutils.FindAssignment()
        visitor.visit(code)
        if visitor.multiple:
            return None
        for assign in visitor.assignments:
            if assign in assignments:
                return None
            assignments[assign] = visitor.assignments[assign]

    if loop.loop_variable in assignments:
        return symbolic.pystr_to_symbolic(assignments[loop.loop_variable])

    return None


def get_update_assignment(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    """
    Parse a loop region's update statement to identify the exact update assignment expression.
    """
    update_stmt = loop.update_statement
    if update_stmt is None:
        return None

    update_codes_list = update_stmt.code if isinstance(update_stmt.code, list) else [update_stmt.code]
    assignments: Dict[str, str] = {}
    for code in update_codes_list:
        visitor = astutils.FindAssignment()
        visitor.visit(code)
        if visitor.multiple:
            return None
        for assign in visitor.assignments:
            if assign in assignments:
                return None
            assignments[assign] = visitor.assignments[assign]

    if loop.loop_variable in assignments:
        return symbolic.pystr_to_symbolic(assignments[loop.loop_variable])

    return None


def get_loop_stride(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
    update_assignment = get_update_assignment(loop)
    if update_assignment:
        return update_assignment - symbolic.pystr_to_symbolic(loop.loop_variable)
    return None


def _provably_le(a: symbolic.SymbolicType, b: symbolic.SymbolicType) -> bool:
    """Prove ``a <= b`` SOUNDLY, returning ``False`` when it cannot be decided (never a guess).

    Beyond a concrete numeric verdict and sympy's own non-positivity engine (which respects the
    codebase's non-negative symbols), this proves the ``Min`` / ``Max`` clamp shape a range split
    leaves behind: ``Min(..., t) <= t`` for any of its own args ``t``, ``t <= Max(..., t)``, and the
    combined ``Min(..., t) <= t <= Max(..., t)`` sharing a term ``t``.
    """
    diff = symbolic.simplify(a - b)
    if diff.is_number:
        return bool(diff <= 0)
    if diff.is_nonpositive:  # sympy assumption engine; None (undecided) is falsy -> not proven
        return True
    a_min = a.args if isinstance(a, sympy.Min) else ()
    b_max = b.args if isinstance(b, sympy.Max) else ()
    if b in a_min:  # a = Min(..., b, ...) <= b
        return True
    if a in b_max:  # a <= Max(..., a, ...) = b
        return True
    return bool(a_min and b_max and (set(a_min) & set(b_max)))  # a <= shared t <= b


def loop_provably_at_most_one_iteration(loop: LoopRegion) -> bool:
    """Whether ``loop`` provably runs at most once (zero or one iterations).

    Such a loop carries no cross-iteration dependence by construction, so it is trivially DOALL --
    a ``LoopToMap`` can map it without any dependence analysis. Restricted to the unit ascending
    stride so the inclusive trip count is exactly ``end - start + 1``; the loop runs at most once iff
    ``end <= start``. Conservative: returns ``False`` whenever the bound cannot be proven.
    """
    start = get_init_assignment(loop)
    end = get_loop_end(loop)  # inclusive
    step = get_loop_stride(loop)
    if start is None or end is None or step is None or symbolic.simplify(step) != 1:
        return False
    return _provably_le(symbolic.simplify(end), symbolic.simplify(start))


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
    basis: Optional['InductionVariable'] = None
    scale: Optional[symbolic.SymbolicType] = None
    offset: Optional[symbolic.SymbolicType] = None


def affine_in_iv(
    expr: symbolic.SymbolicType,
    ivs: Dict[str, 'InductionVariable'],
    invariant_syms: Optional[Set[str]] = None,
) -> Optional[Tuple[Optional[str], symbolic.SymbolicType, symbolic.SymbolicType]]:
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
        try:
            scale = symbolic.simplify(sympy.diff(e, iv_sym))
        except Exception:
            continue
        scale_free = {str(s) for s in scale.free_symbols}
        if scale_free & iv_names:
            continue
        try:
            offset = symbolic.simplify(e - scale * iv_sym)
        except Exception:
            continue
        offset_free = {str(s) for s in offset.free_symbols}
        if iv_name in offset_free:
            continue
        if offset_free & iv_names:
            continue
        if invariant_syms is not None:
            if not ((scale_free | offset_free) <= invariant_syms):
                continue
        return (iv_name, scale, offset)

    return None


def _collect_tasklet_derived_ivs(loop: LoopRegion, pending: Dict[str, str]) -> None:
    """
    Populate ``pending`` with single-assignment Python tasklets in the loop's
    start state that write to a scalar data descriptor. The key is the data
    name (not the tasklet connector). Consumers that want to substitute must
    understand this refers to a data descriptor, not a DaCe symbol.
    """
    # Local imports to avoid import cycles at module load.
    from dace.sdfg import nodes
    from dace.sdfg.state import SDFGState
    from dace import dtypes

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
        for state in loop.all_states():
            for nn in state.data_nodes():
                if nn.data == out_data and state.in_degree(nn) > 0:
                    writers += 1
        if writers != 1:
            continue
        code = n.code.code if hasattr(n.code, 'code') else n.code
        if not isinstance(code, list) or len(code) != 1:
            continue
        stmt = code[0]
        if not isinstance(stmt, astutils.ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], astutils.ast.Name):
            continue
        rhs_str = astutils.unparse(stmt.value)
        pending.setdefault(out_data, rhs_str)


def detect_induction_variables(loop: LoopRegion) -> Dict[str, InductionVariable]:
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
    ivs: Dict[str, InductionVariable] = {}
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
        kind='basic',
    )

    # Collect candidate derived IV assignments.
    pending: Dict[str, str] = {}
    rejected: Set[str] = set()
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
                kind='derived',
                basis=basis_iv,
                scale=scale,
                offset=offset,
            )
            pending.pop(name)
            changed = True

    return ivs
