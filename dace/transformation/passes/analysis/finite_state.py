# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Is a loop-carried symbol the state of a finite-state machine?

A loop that chains through one symbol is sequential by construction::

    for i in range(N):
        state = trans[state, symbols[i]]
        counts[state] += 1

No subset test removes that dependence: it is on the VALUE of ``state``. The loop is still a scan, because
composing transitions is associative. It is a scan with a carry of bounded size when ``state`` takes finitely many
values: a block of iterations is then the table of its end state per start state, and two tables compose by
lookup. This module decides when the state is finite, and which part of the body computes it.

WHAT IS PROVEN. Every iteration reads ``state`` as an index into a dimension of extent ``E`` (``trans[state, ...]``)
on the straight-line prefix of its body, before anything assigns ``state``. A program whose accesses stay in bounds
therefore has ``0 <= index(state) < E`` at every iteration, including the first, whose ``state`` is the value the
loop is entered with. z3 proves that this constraint confines ``state`` to ``count`` values starting at ``base``,
for every value of the program's other symbols. The in-bounds precondition is the program's own: a run that reads
out of bounds has no defined result to preserve.

WHAT THE STEP IS. The assignments ``state`` depends on, through data and through the conditions that guard them,
form the step ``state' = f(state, i)``. The step may read only arrays the loop does not write, so running it on
any start state, in any order, reads the values the loop would have read. Everything else in the body (the
``counts`` update) stays in the loop, which no longer carries ``state`` once each iteration reads its start state
from the scan.
"""

import ast
import copy
from dataclasses import dataclass

import sympy

from dace import SDFG, data, dtypes, symbolic
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis, smt_dependence


@dataclass(frozen=True, slots=True)
class StepSlice:
    """A loop's carried symbol and the part of its body that computes it.

    :ivar symbol: the carried symbol.
    :ivar values: every symbol and scalar the step defines, ``symbol`` included.
    :ivar tables: the containers the step reads; the loop defines none of them.
    """

    symbol: str
    values: frozenset[str]
    tables: frozenset[str]


@dataclass(frozen=True, slots=True)
class FiniteState:
    """A loop's finite carried state: ``count`` values from ``base``.

    :ivar step: the state and the slice that computes it.
    :ivar base: the smallest value the state takes.
    :ivar count: how many values it takes.
    """

    step: StepSlice
    base: sympy.Basic
    count: sympy.Basic


@dataclass(frozen=True, slots=True)
class Subscript:
    """One ``array[index, ...]`` read inside an expression."""

    array: str
    indices: tuple[str, ...]


def is_scalar(desc: data.Data) -> bool:
    """Whether ``desc`` holds one value."""
    return isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1,))


def subscripts(text: str) -> list[Subscript]:
    """Every subscripted read in the Python expression ``text``."""
    found: list[Subscript] = []
    for node in ast.walk(ast.parse(text, mode="eval")):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            found.append(Subscript(node.value.id, tuple(ast.unparse(part) for part in index)))
    return found


def names(text: str) -> set[str]:
    """Every name the expression or statement ``text`` mentions, array names included."""
    return {node.id for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Name)}


def straight_chain(region: ControlFlowRegion) -> list[ControlFlowBlock] | None:
    """The blocks of ``region`` in execution order, if they run one after the other on unconditional edges."""
    start = region.start_block
    chain = [start]
    while region.out_degree(chain[-1]) == 1:
        edge = region.out_edges(chain[-1])[0]
        if not edge.data.is_unconditional():
            return None
        chain.append(edge.dst)
    if region.out_degree(chain[-1]) != 0 or len(chain) != region.number_of_nodes():
        return None
    return chain


def chain_edge_assignments(region: ControlFlowRegion, block: ControlFlowBlock) -> list[tuple[str, str]]:
    """The assignments on ``block``'s one outgoing edge, in evaluation order; none after the last block."""
    edges = region.out_edges(block)
    return list(edges[0].data.assignments.items()) if edges else []


def assignments_in(region: ControlFlowRegion) -> list[tuple[str, str]]:
    """Every interstate assignment anywhere in ``region``."""
    return [(lhs, rhs) for edge in region.all_interstate_edges() for lhs, rhs in edge.data.assignments.items()]


def assigns(block: ControlFlowBlock, symbol: str) -> bool:
    """Whether ``block`` assigns ``symbol`` on an edge inside it."""
    if not isinstance(block, ControlFlowRegion):
        return False
    return any(lhs == symbol for lhs, rhs in assignments_in(block))


def carried_symbols(loop: LoopRegion, chain: list[ControlFlowBlock]) -> list[str]:
    """The symbols the body assigns and reads, in some iteration, before assigning them."""
    assigned = {lhs for lhs, rhs in assignments_in(loop)}
    defined: set[str] = set()
    carried: list[str] = []
    for block in chain:
        carried += sorted((block.free_symbols & assigned) - defined)
        for lhs, rhs in chain_edge_assignments(loop, block):
            carried += sorted((names(rhs) & assigned) - defined)
            defined.add(lhs)
    return list(dict.fromkeys(carried))


def definitions(region: ControlFlowRegion) -> list[tuple[str, set[str]]]:
    """Every value ``region`` defines with the names it reads: interstate assignments, tasklet writes and copies."""
    found = [(lhs, names(rhs)) for lhs, rhs in assignments_in(region)]
    for state in (block for block in region.all_control_flow_blocks() if isinstance(block, SDFGState)):
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet):
                reads = {edge.data.data for edge in state.in_edges(node) if not edge.data.is_empty()}
                reads |= set(node.free_symbols)
                found += [(edge.data.data, reads) for edge in state.out_edges(node) if not edge.data.is_empty()]
        found += [
            (edge.dst.data, {edge.src.data})
            for edge in state.edges()
            if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)
        ]
    return found


def defines(region: ControlFlowRegion, needed: set[str]) -> bool:
    """Whether ``region`` defines a ``needed`` value."""
    return any(lhs in needed for lhs, reads in definitions(region))


def guarded_assignments(region: ControlFlowRegion, needed: set[str]) -> set[str]:
    """The names read by the conditions that guard a definition of a ``needed`` value."""
    guards: set[str] = set()
    for block in region.all_control_flow_blocks():
        if isinstance(block, LoopRegion) and defines(block, needed):
            for statement in (block.loop_condition, block.init_statement, block.update_statement):
                guards |= names(statement.as_string) if statement is not None else set()
        if not isinstance(block, ConditionalBlock):
            continue
        for condition, branch in block.branches:
            if condition is not None and defines(branch, needed):
                guards |= names(condition.as_string)
    for edge in region.all_interstate_edges():
        if not edge.data.is_unconditional() and any(lhs in needed for lhs in edge.data.assignments):
            guards |= names(edge.data.condition.as_string)
    return guards


def step_slice(loop: LoopRegion, symbol: str) -> tuple[frozenset[str], frozenset[str]] | None:
    """The values ``symbol`` depends on inside ``loop``, through definitions and the guards around them, and the
    containers it reads that the loop does not define. ``None`` when it depends on a non-scalar container the loop
    writes, which the step could not recompute alone."""
    containers = loop.sdfg.arrays
    found = definitions(loop)
    defined = {lhs for lhs, reads in found}
    needed = {symbol}
    reads: set[str] = set()
    while True:
        for lhs, used in found:
            if lhs in needed:
                reads |= used
        reads |= guarded_assignments(loop, needed)
        grown = needed | (reads & defined)
        if grown == needed:
            break
        needed = grown
    if any(name in containers and not is_scalar(containers[name]) for name in needed):
        return None
    return frozenset(needed), frozenset(name for name in reads - defined if name in containers)


def unit_offset(index: str, symbol: str) -> sympy.Basic | None:
    """``c`` when ``index`` is ``symbol + c`` with ``c`` free of ``symbol``, else ``None``."""
    expr = symbolic.pystr_to_symbolic(index)
    sym = symbolic.pystr_to_symbolic(symbol)
    if sympy.simplify(sympy.diff(expr, sym) - 1) != 0:
        return None
    offset = sympy.simplify(expr - sym)
    return None if sym in offset.free_symbols else offset


def proves_range(index: str, extent: sympy.Basic, symbol: str, base: sympy.Basic, count: sympy.Basic) -> bool:
    """z3: ``0 <= index < extent`` confines ``symbol`` to ``[base, base + count)`` for every other symbol."""
    if not smt_dependence.has_z3():
        return False
    z3 = smt_dependence.z3
    cache: dict = {}
    arrays: dict = {}
    term = smt_dependence._sympy_to_z3(symbolic.pystr_to_symbolic(index), cache, arrays)
    bound = smt_dependence._sympy_to_z3(sympy.sympify(extent), cache, arrays)
    low = smt_dependence._sympy_to_z3(sympy.sympify(base), cache, arrays)
    size = smt_dependence._sympy_to_z3(sympy.sympify(count), cache, arrays)
    state = smt_dependence._sympy_to_z3(symbolic.pystr_to_symbolic(symbol), cache, arrays)
    if any(part is None for part in (term, bound, low, size, state)):
        return False
    in_bounds = z3.And(0 <= term, term < bound)
    return smt_dependence.prove_unsat(in_bounds, z3.And(low <= state, state < low + size)) is True


def index_bound(
    loop: LoopRegion, chain: list[ControlFlowBlock], symbol: str, shapes: dict[str, tuple]
) -> tuple[sympy.Basic, sympy.Basic] | None:
    """``(base, count)`` from the first read that indexes a table with ``symbol`` on the body's straight-line prefix,
    before anything assigns ``symbol``."""
    for block in chain:
        if assigns(block, symbol):
            return None
        for lhs, rhs in chain_edge_assignments(loop, block):
            for read in subscripts(rhs):
                if read.array not in shapes or len(read.indices) != len(shapes[read.array]):
                    continue
                for dim, index in enumerate(read.indices):
                    if symbol not in names(index):
                        continue
                    offset = unit_offset(index, symbol)
                    if offset is None:
                        continue
                    extent = shapes[read.array][dim]
                    if proves_range(index, extent, symbol, -offset, extent):
                        return -offset, extent
            if lhs == symbol:
                return None
    return None


def step_slices(loop: LoopRegion) -> list[StepSlice]:
    """Every carried symbol of ``loop`` whose step reads no other carried symbol, with that step.

    Empty when the body is not a straight chain of blocks or the loop does not step by one.
    """
    if loop_analysis.get_loop_stride(loop) != 1 or loop_analysis.loop_jumps(loop):
        return []
    chain = straight_chain(loop)
    if chain is None:
        return []
    carried = carried_symbols(loop, chain)
    found = []
    for symbol in carried:
        sliced = step_slice(loop, symbol)
        if sliced is not None and not sliced[0] & (set(carried) - {symbol}):
            found.append(StepSlice(symbol, sliced[0], sliced[1]))
    return found


def finite_state(loop: LoopRegion) -> FiniteState | None:
    """The finite carried state of ``loop``, or ``None`` when no carried symbol has a step and a table bound."""
    chain = straight_chain(loop)
    shapes = {name: tuple(desc.shape) for name, desc in loop.sdfg.arrays.items()}
    for step in step_slices(loop):
        bound = index_bound(loop, required(chain), step.symbol, shapes)
        if bound is not None:
            return FiniteState(step, bound[0], bound[1])
    return None


def step_dataflow(state: SDFGState, values: frozenset[str]) -> set:
    """The nodes of ``state`` that compute one of ``values``: tasklets and copies writing it, and the access nodes
    they read and write."""
    kept: set = set()
    for edge in state.edges():
        if edge.data.is_empty() or edge.data.data not in values or not isinstance(edge.dst, nodes.AccessNode):
            continue
        if isinstance(edge.src, (nodes.Tasklet, nodes.AccessNode)):
            kept |= {edge.src, edge.dst} | {inner.src for inner in state.in_edges(edge.src)}
    return kept


def step_sdfg(loop: LoopRegion, step: StepSlice) -> SDFG:
    """A copy of ``loop``'s body that keeps only ``step``: its control flow, the definitions of the step's values,
    and the tables they read. Its symbols are the step's, the loop variable and every size the tables need."""
    sdfg = loop.sdfg
    result = SDFG(f"{loop.label}_step")
    body = copy.deepcopy(loop)
    for block in list(body.nodes()):
        result.add_node(block, is_start_block=block is body.start_block)
    for edge in body.edges():
        result.add_edge(edge.src, edge.dst, edge.data)
    for edge in result.all_interstate_edges():
        for lhs in [lhs for lhs in edge.data.assignments if lhs not in step.values]:
            del edge.data.assignments[lhs]
    for block in result.all_control_flow_blocks():
        if isinstance(block, SDFGState):
            block.remove_nodes_from([node for node in block.nodes() if node not in step_dataflow(block, step.values)])
    for name in sorted(step.tables):
        desc = copy.deepcopy(sdfg.arrays[name])
        desc.transient = False
        result.add_datadesc(name, desc)
    for name in sorted(step.values & set(sdfg.arrays)):
        result.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]))
    result.reset_cfg_list()
    for name in sorted(result.free_symbols | set(step.values) | {loop.loop_variable}):
        if name in sdfg.symbols and name not in result.symbols:
            result.add_symbol(name, sdfg.symbols[name])
    if loop.loop_variable not in result.symbols:
        result.add_symbol(loop.loop_variable, dtypes.int64)
    return result
