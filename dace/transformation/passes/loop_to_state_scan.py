# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop that carries a finite-state or GF(2)-affine symbol to a :class:`~dace.libraries.standard.nodes.state_scan.StateScan`.

::

    state = 0                                  state = 0
    for i in range(N):                         seq = StateScan(step, state, 0 <= state < NS)
        state = trans[state, symbols[i]]  ->   for i in range(N):
        counts[state] += 1                         state = seq[i]
                                                   state = trans[state, symbols[i]]
                                                   counts[state] += 1
                                               state = seq[N]

:mod:`~dace.transformation.passes.analysis.finite_state` proves the state finite and slices out the step; failing
that, :mod:`~dace.transformation.passes.analysis.carried_state` proves it GF(2)-affine (a CRC). The scan
computes every iteration's start state, so the loop reads it instead of carrying it, and a later ``LoopToMap``
makes the loop parallel. The body keeps its own assignments to the state: they are dead across iterations, and the
values they give inside an iteration are the ones the loop computed before.

The pass lifts only a loop whose other carried symbols are sum or product reductions, so the loop it leaves behind
carries nothing a map cannot.
"""

import ast

import dace
from dace import SDFG, dtypes, symbolic
from dace.libraries.standard.nodes.state_scan import INIT_CONNECTOR, OUT_CONNECTOR, StateScan, StateScanKind
from dace.memlet import Memlet
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.carried_state import gf2_state
from dace.transformation.passes.analysis.finite_state import (
    StepSlice,
    assignments_in,
    carried_symbols,
    finite_state,
    names,
    step_sdfg,
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


def rest_is_parallel(loop: LoopRegion, symbol: str) -> bool:
    """Whether every symbol ``loop`` carries besides ``symbol`` is a reduction."""
    chain = straight_chain(loop)
    return chain is not None and all(
        is_reduction(loop, name) for name in carried_symbols(loop, chain) if name != symbol
    )


def folds_only(loop: LoopRegion, symbol: str) -> bool:
    """Whether ``symbol`` is a sum or product reduction nothing else in ``loop`` reads: only its exit value is
    used, which is the loop's closed form, not a scan."""
    if not is_reduction(loop, symbol):
        return False
    if any(symbol in names(rhs) for lhs, rhs in assignments_in(loop) if lhs != symbol):
        return False
    return not any(symbol in state.free_symbols for state in loop.states())


def scan_node(loop: LoopRegion) -> tuple[StateScan, StepSlice] | None:
    """A :class:`StateScan` for ``loop``'s carried state, finite if it is, else GF(2)-affine, or ``None``."""
    finite = finite_state(loop)
    if finite is not None:
        node = StateScan(f"{loop.label}_state_scan", tuple(sorted(finite.step.tables)))
        node.kind, node.base, node.count = StateScanKind.FINITE, finite.base, finite.count
        return node, finite.step
    affine = gf2_state(loop)
    if affine is not None:
        node = StateScan(f"{loop.label}_state_scan", tuple(sorted(affine.step.tables)))
        node.kind, node.bits = StateScanKind.GF2_AFFINE, affine.bits
        return node, affine.step
    return None


def lift(loop: LoopRegion, node: StateScan, found: StepSlice) -> None:
    """Replace ``loop``'s carried state by ``node``, in a state in front of it."""
    sdfg = loop.sdfg
    parent: ControlFlowRegion = loop.parent_graph
    begin = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop) + 1
    dtype = sdfg.symbols.get(found.symbol, dtypes.int64)
    sequence = sdfg.add_array(f"{found.symbol}_states", [end - begin + 1], dtype, transient=True, find_new_name=True)[0]
    seed = sdfg.add_scalar(f"{found.symbol}_start", dtype, transient=True, find_new_name=True)[0]

    scan_state = parent.add_state_before(loop, f"{loop.label}_state_scan", is_start_block=parent.start_block is loop)
    node.step = step_sdfg(loop, found)
    node.state = found.symbol
    node.iterate = loop.loop_variable
    node.begin = begin
    node.end = end
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
    """Lift loops whose carried symbol is a proven-finite or GF(2)-affine state to :class:`StateScan` library
    nodes."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self, only: list | None = None):
        super().__init__()
        # When given, only these loops are candidates (the segments a split left behind).
        self._only = only

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> int | None:
        """Lift every eligible loop; returns how many, or ``None`` when none was."""
        lifted = 0
        if self._only is not None and not self._only:
            return None
        loops = [cfg for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, LoopRegion)]
        if self._only is not None:
            only = {id(loop) for loop in self._only}
            loops = [loop for loop in loops if id(loop) in only]
        for loop in loops:
            made = scan_node(loop)
            if made is None or not rest_is_parallel(loop, made[1].symbol) or folds_only(loop, made[1].symbol):
                continue
            lift(loop, made[0], made[1])
            lifted += 1
        return lifted or None
