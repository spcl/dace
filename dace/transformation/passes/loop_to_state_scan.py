# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop that carries a finite-state symbol to a :class:`~dace.libraries.standard.nodes.state_scan.StateScan`.

::

    state = 0                                  state = 0
    for i in range(N):                         seq = StateScan(step, state, 0 <= state < NS)
        state = trans[state, symbols[i]]  ->   for i in range(N):
        counts[state] += 1                         state = seq[i]
                                                   state = trans[state, symbols[i]]
                                                   counts[state] += 1
                                               state = seq[N]

:mod:`~dace.transformation.passes.analysis.finite_state` proves the state finite and slices out the step. The scan
computes every iteration's start state, so the loop reads it instead of carrying it, and a later ``LoopToMap``
makes the loop parallel. The body keeps its own assignments to the state: they are dead across iterations, and the
values they give inside an iteration are the ones the loop computed before.

The pass lifts only a loop whose other carried symbols are sum or product reductions, so the loop it leaves behind
carries nothing a map cannot.
"""

import ast
import copy

import dace
from dace import SDFG, dtypes, symbolic
from dace.libraries.standard.nodes.state_scan import INIT_CONNECTOR, OUT_CONNECTOR, StateScan
from dace.memlet import Memlet
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.finite_state import (
    FiniteState,
    assignments_in,
    carried_symbols,
    finite_state,
    names,
    straight_chain,
)


def is_reduction(loop: LoopRegion, symbol: str) -> bool:
    """Whether ``symbol``'s only assignment in ``loop`` is ``symbol = symbol + e`` or ``symbol * e``, either way
    round, with ``e`` free of ``symbol``."""
    rhs = [value for lhs, value in assignments_in(loop) if lhs == symbol]
    if len(rhs) != 1:
        return False
    expr = ast.parse(rhs[0], mode="eval").body
    if not isinstance(expr, ast.BinOp) or not isinstance(expr.op, (ast.Add, ast.Mult)):
        return False
    sides = [expr.left, expr.right]
    carried = [side for side in sides if isinstance(side, ast.Name) and side.id == symbol]
    others = [side for side in sides if side not in carried]
    return len(carried) == 1 and all(symbol not in names(ast.unparse(side)) for side in others)


def rest_is_parallel(loop: LoopRegion, found: FiniteState) -> bool:
    """Whether every symbol ``loop`` carries besides the state is a reduction."""
    chain = straight_chain(loop)
    return chain is not None and all(
        is_reduction(loop, name) for name in carried_symbols(loop, chain) if name != found.symbol
    )


def step_sdfg(loop: LoopRegion, found: FiniteState) -> SDFG:
    """A copy of ``loop``'s body that keeps only the step: its control flow, the assignments to the step's symbols,
    and the tables they read. Every state is emptied; the step reads no dataflow result."""
    sdfg = loop.sdfg
    step = SDFG(f"{loop.label}_step")
    body = copy.deepcopy(loop)
    for block in list(body.nodes()):
        step.add_node(block, is_start_block=block is body.start_block)
    for edge in body.edges():
        step.add_edge(edge.src, edge.dst, edge.data)
    for edge in step.all_interstate_edges():
        for lhs in [lhs for lhs in edge.data.assignments if lhs not in found.step_symbols]:
            del edge.data.assignments[lhs]
    for block in step.all_control_flow_blocks():
        if isinstance(block, SDFGState):
            block.remove_nodes_from(list(block.nodes()))
    for name in found.tables:
        desc = copy.deepcopy(sdfg.arrays[name])
        desc.transient = False
        step.add_datadesc(name, desc)
    step.reset_cfg_list()
    for name in step.free_symbols | set(found.step_symbols) | {loop.loop_variable}:
        if name in sdfg.symbols and name not in step.symbols:
            step.add_symbol(name, sdfg.symbols[name])
    if loop.loop_variable not in step.symbols:
        step.add_symbol(loop.loop_variable, dtypes.int64)
    return step


def lift(loop: LoopRegion, found: FiniteState) -> None:
    """Replace ``loop``'s carried state by a :class:`StateScan` in front of it."""
    sdfg = loop.sdfg
    parent: ControlFlowRegion = loop.parent_graph
    begin = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop) + 1
    dtype = sdfg.symbols.get(found.symbol, dtypes.int64)
    sequence = sdfg.add_array(f"{found.symbol}_states", [end - begin + 1], dtype, transient=True, find_new_name=True)[0]
    seed = sdfg.add_scalar(f"{found.symbol}_start", dtype, transient=True, find_new_name=True)[0]

    scan_state = parent.add_state_before(loop, f"{loop.label}_state_scan", is_start_block=parent.start_block is loop)
    node = StateScan(f"{loop.label}_state_scan", tuple(sorted(found.tables)))
    node.step = step_sdfg(loop, found)
    node.state = found.symbol
    node.iterate = loop.loop_variable
    node.begin = begin
    node.end = end
    node.base = found.base
    node.count = found.count
    scan_state.add_node(node)
    take = scan_state.add_tasklet("start_state", {}, {"out": None}, f"out = {found.symbol}")
    start = scan_state.add_access(seed)
    scan_state.add_edge(take, "out", start, None, Memlet(f"{seed}[0]"))
    scan_state.add_edge(start, None, node, INIT_CONNECTOR, Memlet(f"{seed}[0]"))
    for name in sorted(found.tables):
        scan_state.add_edge(scan_state.add_read(name), None, node, name, Memlet.from_array(name, sdfg.arrays[name]))
    scan_state.add_edge(
        node, OUT_CONNECTOR, scan_state.add_write(sequence), None, Memlet.from_array(sequence, sdfg.arrays[sequence])
    )

    first = symbolic.symstr(begin, cpp_mode=False)
    body_start = loop.start_block
    head = loop.add_state(f"{loop.label}_start_state", is_start_block=True)
    loop.add_edge(
        head,
        body_start,
        dace.InterstateEdge(assignments={found.symbol: f"{sequence}[{loop.loop_variable} - ({first})]"}),
    )
    last = symbolic.symstr(end - begin, cpp_mode=False)
    parent.add_state_after(loop, f"{loop.label}_end_state", assignments={found.symbol: f"{sequence}[{last}]"})
    sdfg.reset_cfg_list()


class LoopToStateScan(ppl.Pass):
    """Lift loops whose carried symbol is a proven-finite state to :class:`StateScan` library nodes."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> int | None:
        """Lift every eligible loop; returns how many, or ``None`` when none was."""
        lifted = 0
        loops = [cfg for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, LoopRegion)]
        for loop in loops:
            found = finite_state(loop)
            if found is None or not rest_is_parallel(loop, found):
                continue
            lift(loop, found)
            lifted += 1
        return lifted or None
