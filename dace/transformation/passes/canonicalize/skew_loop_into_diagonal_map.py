# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A sequential loop ``i`` around a one-parameter map ``j`` whose every dependence has the uniform distance
``(d, c*d)`` becomes ONE map over the diagonals ``w = j - c*i``, each running its chain of ``i`` in order.

``for i: map j: a[i, j] = a[i - 1, j - 1] + b[i, j]`` (TSVC s119) launches one kernel per ``i``. Every dependence
stays on one diagonal ``j - i``, so the diagonals are independent: after the skew ``j = w + c*i`` no dependence
crosses ``w`` and :class:`~dace.transformation.interstate.move_loop_into_map.MoveLoopIntoMap` moves the loop into
the map, with the bounds of ``i`` clipped to the diagonal. A dependence is the distance between a store and a load
(or a second store) of one container, solved exactly from their affine subscripts; a subscript that is not affine in
``(i, j)``, a pair whose distance is not one constant vector, or two distances of different slopes refuses.

Reference: Wolf & Lam, TPDS '91 (unimodular skewing and interchange).
"""

from typing import Any

import sympy

from dace import SDFG, SDFGState, subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap
from dace.transformation.passes.analysis import loop_analysis


def single_map(loop: LoopRegion) -> tuple[SDFGState, nodes.MapEntry] | None:
    """The body state and its one top-level one-parameter unit-step map, when the body is just that."""
    if len(loop.nodes()) != 1 or not isinstance(loop.nodes()[0], SDFGState):
        return None
    state: SDFGState = loop.nodes()[0]
    scopes = state.scope_dict()
    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scopes[n] is None]
    if len(entries) != 1 or len(entries[0].map.params) != 1 or entries[0].map.range[0][2] != 1:
        return None
    if any(
        scopes[n] is None and not isinstance(n, (nodes.AccessNode, nodes.MapEntry, nodes.MapExit))
        for n in state.nodes()
    ):
        return None
    return state, entries[0]


def accesses(state: SDFGState, entry: nodes.MapEntry) -> tuple[dict[str, list], dict[str, list]] | None:
    """Point subscripts of the containers the map scope reads and writes, or ``None`` on a range or a nested SDFG."""
    reads: dict[str, list] = {}
    writes: dict[str, list] = {}
    inner = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    for node in inner.nodes():
        if isinstance(node, nodes.NestedSDFG):
            return None
    for edge in state.scope_subgraph(entry).edges():
        if edge.data.is_empty():
            continue
        if not (isinstance(edge.src, nodes.Tasklet) or isinstance(edge.dst, nodes.Tasklet)):
            continue
        subset = edge.data.subset
        if subset.num_elements() != 1 or edge.data.wcr is not None:
            return None
        index = [symbolic.pystr_to_symbolic(str(r[0])) for r in subset.ndrange()]
        (writes if isinstance(edge.src, nodes.Tasklet) else reads).setdefault(edge.data.data, []).append(index)
    return reads, writes


def distance(store: list, other: list, i: sympy.Symbol, j: sympy.Symbol) -> tuple[int, int] | None:
    """The constant ``(di, dj)`` with ``store(i, j) == other(i + di, j + dj)`` for every ``(i, j)``, or ``None``."""
    di, dj = sympy.symbols("cpf_skew_di cpf_skew_dj", integer=True)
    equations = [sympy.expand(s - o.subs({i: i + di, j: j + dj}, simultaneous=True)) for s, o in zip(store, other)]
    for expr in (*store, *other):
        if not expr.free_symbols & {i, j}:
            continue
        if not expr.is_polynomial(i, j) or sympy.Poly(expr, i, j).total_degree() > 1:
            return None
    solution = sympy.solve(equations, [di, dj], dict=True)
    if len(solution) != 1 or set(solution[0]) != {di, dj}:
        return None
    vector = solution[0][di], solution[0][dj]
    if not all(v.is_Integer for v in vector):
        return None
    return int(vector[0]), int(vector[1])


def slope(loop: LoopRegion, state: SDFGState, entry: nodes.MapEntry) -> int | None:
    """The one nonzero ``c`` with every dependence distance ``(d, c*d)``, ``d != 0``; ``None`` when there is none."""
    found = accesses(state, entry)
    if found is None:
        return None
    reads, writes = found
    i, j = symbolic.pystr_to_symbolic(loop.loop_variable), symbolic.pystr_to_symbolic(entry.map.params[0])
    slopes: set[int] = set()
    for data, stores in writes.items():
        for store in stores:
            for other in [*reads.get(data, []), *stores]:
                vector = distance(store, other, i, j)
                if vector is None:
                    return None
                di, dj = vector
                if di == 0 and dj == 0:
                    continue
                if di == 0 or dj % di:
                    return None
                slopes.add(dj // di)
    if len(slopes) != 1 or 0 in slopes:
        return None
    return slopes.pop()


@transformation.explicit_cf_compatible
class SkewLoopIntoDiagonalMap(ppl.Pass):
    """Turn each sequential loop over a map whose dependences all lie on one diagonal into a map over the diagonals
    with the loop inside (module docstring)."""

    CATEGORY: str = "Parallelization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        applied = 0
        for loop in [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]:
            applied += self.try_loop(loop)
        return applied or None

    def try_loop(self, loop: LoopRegion) -> bool:
        start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
        if loop.loop_variable is None or start is None or end is None or loop_analysis.get_loop_stride(loop) != 1:
            return False
        body = single_map(loop)
        if body is None:
            return False
        state, entry = body
        c = slope(loop, state, entry)
        if c is None:
            return False
        i, j = loop.loop_variable, entry.map.params[0]
        lo, hi = entry.map.range[0][0], entry.map.range[0][1]
        if {str(s) for s in symbolic.pystr_to_symbolic(f"({lo}) + ({hi})").free_symbols} & {i, j}:
            return False
        w = f"{j}_diag"
        while w in state.sdfg.symbols or w in state.sdfg.arrays:
            w += "_"
        first, last = (c * start, c * end) if c > 0 else (c * end, c * start)
        skewed = f"({w} + {c} * {i})"
        replace_dict(state.scope_subgraph(entry, include_entry=False), {j: skewed})
        for edge in state.out_edges(entry):  # the subgraph without its entry leaves the entry's edges out
            edge.data.replace({j: skewed})
        entry.map.params = [w]
        entry.map.range = type(entry.map.range)([(lo - last, hi - first, 1)])
        sdfg = state.sdfg
        MoveLoopIntoMap.apply_to(sdfg, verify=False, loop=loop)
        # The diagonals' union of positions propagates past the array's edges: the scope's memlets take whole arrays
        exit_node = state.exit_node(entry)
        nested = entry_nested(state, entry)
        for edge in [
            *state.in_edges(entry),
            *state.out_edges(exit_node),
            *state.out_edges(entry),
            *state.in_edges(exit_node),
        ]:
            if edge.data.data is not None and (
                edge.src is nested or edge.dst is nested or edge.src is exit_node or edge.dst is entry
            ):
                edge.data.subset = subsets.Range.from_array(sdfg.arrays[edge.data.data])
                edge.data.dynamic = True
        inner = next(n for n in entry_nested(state, entry).sdfg.all_control_flow_regions() if isinstance(n, LoopRegion))
        # Clip ``i`` to the diagonal: lo <= w + c*i <= hi
        if c > 0:
            first_i, last_i = f"int_ceil(({lo}) - {w}, {c})", f"int_floor(({hi}) - {w}, {c})"
        else:
            first_i, last_i = f"int_ceil({w} - ({hi}), {-c})", f"int_floor({w} - ({lo}), {-c})"
        inner.init_statement = CodeBlock(f"{i} = max({start}, {first_i})")
        inner.loop_condition = CodeBlock(f"{i} <= min({end}, {last_i})")
        return True


def entry_nested(state: SDFGState, entry: nodes.MapEntry) -> nodes.NestedSDFG:
    """The nested SDFG :class:`MoveLoopIntoMap` left directly inside ``entry``."""
    return next(e.dst for e in state.out_edges(entry) if isinstance(e.dst, nodes.NestedSDFG))
