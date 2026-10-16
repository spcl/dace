# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give a while loop that counts its iterations a loop counter.

``while i < N: ...; i += 1`` reaches the SDFG as a ``LoopRegion`` without a ``loop_variable``, and every
dependence test needs one, so the loop is never examined and stays sequential. When the condition names a
symbol ``i`` that the body bumps by a constant exactly once, on an interstate edge after which nothing reads
``i``, the while loop and ``for (k = i; cond[i := k]; k = k + c)`` run the same iterations in the same order: the
condition is checked before each iteration and the bump is the last thing an iteration does to ``i``.

A condition held in a symbol (``while __tmp0`` with ``__tmp0 = i + 17 <= N`` recomputed at the end of the body)
is replaced by its expression when nothing outside the loop reads that symbol. ``i`` itself stays the program's
symbol and, when read after the loop, ends at its exit value, the first that fails the condition.
"""

from typing import Any

import sympy

from dace import SDFG, symbolic
from dace.properties import CodeBlock
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis


def body_edges(loop: LoopRegion, name: str) -> list:
    """The interstate edges anywhere under ``loop`` that assign ``name``."""
    return [e for e in loop.all_interstate_edges() if name in e.data.assignments]


def reads_symbol(block: ControlFlowBlock, name: str) -> bool:
    """Whether ``block`` or anything in it reads ``name``."""
    return name in block.used_symbols(all_symbols=True)


def inside(block: ControlFlowBlock, loop: LoopRegion) -> bool:
    """Whether ``block`` sits anywhere under ``loop``."""
    graph = block.parent_graph
    while graph is not None and graph is not loop:
        graph = graph.parent_graph
    return graph is loop


def read_outside(loop: LoopRegion, name: str, skip_body: bool = False) -> bool:
    """Whether anything other than ``loop``'s condition and the assignments to ``name`` reads ``name``; with
    ``skip_body``, anything outside ``loop``."""
    root = loop.sdfg
    # A loop that binds ``name`` itself reads its own counter there, not this one's value.
    binders = [
        r
        for r in root.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion) and r.loop_variable == name
    ]
    hidden = binders + ([loop] if skip_body else [])
    own = {e for region in hidden for e in region.all_interstate_edges()}
    for edge in root.all_interstate_edges(recursive=True):
        if edge in own:
            continue
        rhs = {k: v for k, v in edge.data.assignments.items() if k != name}
        if name in edge.data.condition.get_free_symbols() or any(
            name in {str(s) for s in symbolic.pystr_to_symbolic(v).free_symbols} for v in rhs.values()
        ):
            return True
    for block in root.all_control_flow_blocks(recursive=True):
        if any(inside(block, region) for region in hidden) or block in binders:
            continue
        if isinstance(block, SDFGState) and reads_symbol(block, name):
            return True
        if isinstance(block, LoopRegion) and block is not loop and name in block.loop_condition.get_free_symbols():
            return True
    return False


def inlined_condition(loop: LoopRegion) -> tuple[sympy.Basic, str | None] | None:
    """``(condition, symbol)``: the loop condition with a condition symbol replaced by its body assignment."""
    cond = symbolic.pystr_to_symbolic(loop.loop_condition.as_string)
    if not isinstance(cond, sympy.Symbol):
        return cond, None
    name = str(cond)
    edges = body_edges(loop, name)
    if len(edges) != 1 or read_outside(loop, name):
        return None
    return symbolic.pystr_to_symbolic(edges[0].data.assignments[name]), name


def later_blocks(region: ControlFlowRegion, start: ControlFlowBlock) -> list[ControlFlowBlock]:
    """``start`` and every block of ``region`` reachable from it."""
    seen, stack = [], [start]
    while stack:
        block = stack.pop()
        if block not in seen:
            seen.append(block)
            stack.extend(e.dst for e in region.out_edges(block))
    return seen


def bump(loop: LoopRegion, name: str, cond_symbol: str | None):
    """``(edge, step)`` when the body adds a constant to ``name`` once, at the top level of its body, after which
    only the condition symbol's assignment reads ``name``; else ``None``."""
    edges = body_edges(loop, name)
    if len(edges) != 1 or edges[0] not in loop.edges():
        return None
    edge = edges[0]
    step = symbolic.pystr_to_symbolic(edge.data.assignments[name]) - symbolic.pystr_to_symbolic(name)
    if not (step.is_Integer and step != 0):
        return None
    for block in later_blocks(loop, edge.dst):
        if reads_symbol(block, name):
            return None
        for out in loop.out_edges(block):
            reads = {
                k
                for k, v in out.data.assignments.items()
                if k != cond_symbol and name in {str(f) for f in symbolic.pystr_to_symbolic(v).free_symbols}
            }
            if reads or name in out.data.condition.get_free_symbols():
                return None
    return edge, int(step)


def convert(loop: LoopRegion) -> bool:
    """Turn ``loop`` into a counted loop when its condition rests on one bumped counter."""
    if loop.loop_variable or loop.inverted or loop_analysis.loop_jumps(loop):
        return False
    inlined = inlined_condition(loop)
    if inlined is None:
        return False
    cond, cond_symbol = inlined
    exits = loop.parent_graph.out_edges(loop)
    for name in sorted(str(s) for s in cond.free_symbols):
        found = bump(loop, name, cond_symbol)
        # The counter starts from a value an edge before the loop gives it: a loop counter cannot read itself.
        defining = start_edge(loop, name) if found is not None else None
        if defining is None:
            continue
        start = symbolic.pystr_to_symbolic(defining.data.assignments[name])
        bounded = solved_for(cond, name)
        last = None
        if read_outside(loop, name, skip_body=True):
            last = exit_value(bounded, name, start, found[1])
            if last is None or len(exits) != 1:
                continue
        rewrite(loop, bounded, cond_symbol, name, *found, defining)
        if last is not None:
            exits[0].data.assignments[name] = symbolic.symstr(last)
        return True
    return False


def start_edge(loop: LoopRegion, name: str):
    """The nearest edge before ``loop`` that assigns ``name``, up a chain of single unconditional edges through
    blocks that assign nothing to it; ``None`` when the chain branches first or the value reads ``name``."""
    region, block = loop.parent_graph, loop
    while True:
        ins = region.in_edges(block)
        if len(ins) != 1 or not ins[0].data.is_unconditional():
            return None
        edge = ins[0]
        if name in edge.data.assignments:
            value = symbolic.pystr_to_symbolic(edge.data.assignments[name])
            return None if name in {str(s) for s in value.free_symbols} else edge
        block = edge.src
        if not isinstance(block, SDFGState):
            return None  # a region before the loop may assign the name inside it


def exit_value(cond, name: str, start, step: int):
    """The value ``name`` leaves the loop with: ``start`` plus ``step`` times the trip count, ``start`` for none.

    Only for ``name < b`` / ``name <= b`` with a positive step, or ``>`` / ``>=`` with a negative one."""
    if not isinstance(cond, sympy.core.relational.Relational) or str(cond.lhs) != name:
        return None
    bound = cond.rhs
    if step > 0 and isinstance(cond, (sympy.StrictLessThan, sympy.LessThan)):
        span = bound - start + (1 if isinstance(cond, sympy.LessThan) else 0)
    elif step < 0 and isinstance(cond, (sympy.StrictGreaterThan, sympy.GreaterThan)):
        span = start - bound + (1 if isinstance(cond, sympy.GreaterThan) else 0)
    else:
        return None
    trips = sympy.Max(0, symbolic.int_ceil(span, abs(step)))
    return start + step * trips


def solved_for(cond, name: str):
    """``cond`` as ``name <rel> bound`` when it is a comparison linear in ``name`` with coefficient 1 or -1, the
    spelling the loop analyses read; else ``cond`` unchanged."""
    if not isinstance(cond, sympy.core.relational.Relational) or isinstance(cond, (sympy.Eq, sympy.Ne)):
        return cond
    var = symbolic.pystr_to_symbolic(name)
    var = next((s for s in cond.free_symbols if str(s) == name), var)
    diff = sympy.expand(cond.lhs - cond.rhs)
    coeff = diff.coeff(var)
    if coeff not in (1, -1) or var in (diff - coeff * var).free_symbols:
        return cond
    bound = -(diff - coeff * var) / coeff
    # ``-i < b`` is ``b' < i``: build it the other way round, then flip it to put ``i`` first.
    return type(cond)(var, bound) if coeff == 1 else type(cond)(bound, var).reversed


def rewrite(loop: LoopRegion, cond, cond_symbol: str | None, name: str, edge, step: int, defining) -> None:
    """Make ``name`` the loop's counter: started from ``defining``'s value, bumped as the update.

    A loop counter is scope-bound and no SDFG symbol, so ``name`` leaves the SDFG's symbols. The defining assignment
    moves into the loop header when it sits on the entry edge; further back, other blocks may still read it.
    """
    del edge.data.assignments[name]
    entries = loop.parent_graph.in_edges(loop)
    if cond_symbol is not None:
        for e in body_edges(loop, cond_symbol) + entries:
            e.data.assignments.pop(cond_symbol, None)
    start = defining.data.assignments[name]
    if defining in entries:
        del defining.data.assignments[name]
    loop.loop_variable = name
    loop.init_statement = CodeBlock(f"{name} = {start}")
    loop.loop_condition = CodeBlock(symbolic.symstr(cond))
    loop.update_statement = CodeBlock(f"{name} = {name} + ({step})")
    if name in loop.sdfg.symbols:
        loop.sdfg.remove_symbol(name)


@xf.explicit_cf_compatible
class WhileToFor(ppl.Pass):
    """Give every while loop whose condition rests on a constant-step counter that counter as its loop variable."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """:returns: The number of loops converted, or ``None`` if none was."""
        # A loop converted ahead of the next one hands it the start value on the edge between them, so repeat.
        converted, progress = 0, True
        while progress:
            progress = False
            for region in list(sdfg.all_control_flow_regions(recursive=True)):
                if isinstance(region, LoopRegion) and convert(region):
                    converted, progress = converted + 1, True
        return converted or None
