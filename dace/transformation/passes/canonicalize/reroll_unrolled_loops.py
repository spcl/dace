# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Re-roll a manually-unrolled lane chain back into a step-``g`` loop."""

import copy
import re
from typing import Any

import dace
from dace import symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_expr
from dace.sdfg.state import LoopRegion, SDFGState, is_written
from dace.subsets import Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import ValueIds, value_keys
from dace.transformation.passes.canonicalize.split_statements import value_edges
from dace.transformation.transformation import explicit_cf_compatible


class IteratorScaling:
    """Undo record of :func:`unscale_iterator`: the loop's header and every subset it rewrote."""

    __slots__ = ("loop", "factor", "header", "subsets")

    def __init__(self, loop: LoopRegion, factor: int, header: tuple, subsets: list[tuple[Memlet, Subset]]):
        self.loop = loop
        self.factor = factor
        self.header = header
        self.subsets = subsets

    def restore(self) -> None:
        self.loop.init_statement, self.loop.loop_condition, self.loop.update_statement = self.header
        for memlet, subset in self.subsets:
            memlet.subset = subset


def fuse_body_states(loop: LoopRegion) -> None:
    """Fuse the chain of states an unrolled body still has once its counter edges are empty."""
    from dace.transformation.interstate.state_fusion import StateFusion

    fused = True
    while fused and len(loop.nodes()) > 1:
        fused = False
        for edge in list(loop.edges()):
            if (
                not (isinstance(edge.src, SDFGState) and isinstance(edge.dst, SDFGState))
                or not edge.data.is_unconditional()
            ):
                continue
            if edge.data.assignments:
                continue
            if StateFusion.can_be_applied_to(loop.sdfg, first_state=edge.src, second_state=edge.dst):
                StateFusion.apply_to(loop.sdfg, verify=False, first_state=edge.src, second_state=edge.dst)
                fused = True
                break


def names_of(expr) -> set[str]:
    """The free symbol names of a subset, an expression or an interstate edge."""
    return {str(x) for x in expr.free_symbols}


def iterator_outside_subsets(loop: LoopRegion, var: str) -> bool:
    """Does ``var`` appear anywhere in ``loop``'s body but a memlet subset, or the body hold more than tasklets?"""
    if any(var in edge.data.free_symbols for edge in loop.all_interstate_edges()):
        return True
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            if block is not loop and var in names_of(block):
                return True
            continue
        for node in block.nodes():
            if not isinstance(node, (nodes.AccessNode, nodes.Tasklet)):
                return True
            if isinstance(node, nodes.Tasklet) and re.search(rf"\b{re.escape(var)}\b", node.code.as_string or ""):
                return True
    return False


def scaled_subsets(loop: LoopRegion, var: str) -> tuple[int, list[Memlet]] | None:
    """The one coefficient ``m >= 2`` of ``var`` across the body's subsets, and the memlets that carry it."""
    sym = symbolic.pystr_to_symbolic(var)
    factors: set = set()
    memlets: list[Memlet] = []
    for state in loop.states():
        for edge in state.edges():
            memlet = edge.data
            if memlet is None or memlet.subset is None or var not in names_of(memlet.subset):
                continue
            if memlet.other_subset is not None and var in names_of(memlet.other_subset):
                return None
            exprs = [as_expr(bound) for begin, last, _ in memlet.subset.ndrange() for bound in (begin, last)]
            if any((expr - expr.coeff(sym) * sym).has(sym) for expr in exprs):
                return None
            factors.update(expr.coeff(sym) for expr in exprs)
            memlets.append(memlet)
    factors.discard(0)
    factor = _const_int(factors.pop()) if len(factors) == 1 else None
    return (factor, memlets) if factor is not None and factor >= 2 else None


def unscale_iterator(loop: LoopRegion) -> IteratorScaling | None:
    """Rewrite a unit-step loop whose iterator appears only as ``m*i + c`` in subsets into a step-``m`` loop over ``i +
    c``."""
    var = loop.loop_variable
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    if start is None or end is None or iterator_outside_subsets(loop, var):
        return None
    scaled = scaled_subsets(loop, var)
    if scaled is None:
        return None
    factor, memlets = scaled
    sym = symbolic.pystr_to_symbolic(var)
    header = (loop.init_statement, loop.loop_condition, loop.update_statement)
    saved = [(memlet, copy.deepcopy(memlet.subset)) for memlet in memlets]
    for memlet in memlets:
        memlet.subset.replace({sym: sym / factor})
    lo = symbolic.symstr(symbolic.simplify(start * factor))
    hi = symbolic.symstr(symbolic.simplify((end + 1) * factor))
    loop.init_statement = dace.properties.CodeBlock(f"{var} = {lo}")
    loop.loop_condition = dace.properties.CodeBlock(f"{var} < {hi}")
    loop.update_statement = dace.properties.CodeBlock(f"{var} = {var} + {factor}")
    return IteratorScaling(loop, factor, header, saved)


def unrolled_step(loop: LoopRegion) -> tuple[int | None, IteratorScaling | None]:
    """The step the re-roll matchers read: the loop's own, or ``m`` once a scaled iterator is unscaled."""
    stride = loop_analysis.get_loop_stride(loop)
    step = _const_int(stride) if stride is not None else None
    undo = unscale_iterator(loop) if step == 1 else None
    if undo is None:
        return step, None
    return undo.factor, undo


def _const_int(value: symbolic.SymbolicType | int) -> int | None:
    """Return ``value`` as a Python ``int`` if it is a constant integer, else ``None``."""
    try:
        sym = symbolic.pystr_to_symbolic(value) if isinstance(value, str) else symbolic.pystr_to_symbolic(str(value))
        if sym.is_Integer:
            return int(as_expr(sym))
    except Exception:
        return None
    return None


def _offset_of_index(index_expr: str, loop_var: str) -> int | None:
    """Constant offset ``k`` of an index expression ``loop_var + k``."""
    try:
        expr = symbolic.pystr_to_symbolic(index_expr)
    except Exception:
        return None
    if loop_var not in (str(s) for s in expr.free_symbols):
        return None
    return _const_int(as_expr(expr) - as_expr(symbolic.pystr_to_symbolic(loop_var)))


def _index_offset(subset: Subset, loop_var: str) -> int | None:
    """Constant offset ``k`` of a one-dimensional ``loop_var + k`` subset."""
    if subset is None or len(subset) != 1:
        return None
    rb, re_, rs = subset[0]
    if rb != re_:
        return None
    return _offset_of_index(str(rb), loop_var)


def _assignment_offset(rhs: str, loop_var: str) -> int | None:
    """Offset ``k`` of an interstate-assignment RHS ``loop_var + k`` or ``arr[loop_var + k]``."""
    rhs = rhs.strip()
    index_expr = rhs
    if "[" in rhs and rhs.endswith("]"):
        index_expr = rhs[rhs.rindex("[") + 1 : -1]
    return _offset_of_index(index_expr, loop_var)


def lane_signature(
    loop: LoopRegion,
    states: list[SDFGState],
    lane: int,
    private: dict[SDFGState, dict[nodes.Node, None]],
    edges: dict[SDFGState, list[MultiConnectorEdge[Memlet]]],
    aliases: dict[str, str],
    sym_offset: dict[str, int],
    gathers: dict[str, str],
    written: set[str],
    ids: ValueIds,
) -> list | None:
    """What lane ``lane`` hands out, spelled as lane 0 would: equal signatures mean one value computed one offset apart."""
    var = loop.loop_variable
    stored: dict[str, int] = {}

    def name(data: str | None) -> str:
        return aliases.get(data, "") if data in written else data or ""

    def rebased(subset: Subset | None) -> str:
        if subset is None:
            return ""
        exprs = [as_expr(x) for rng in subset.ndrange() for x in rng]
        return str(
            [
                x.subs(
                    {
                        s: s - lane if str(s) == var else symbolic.symbol(f"__{gathers[str(s)]}".replace("[", "_"))
                        for s in x.free_symbols
                        if str(s) == var or str(s) in gathers
                    }
                )
                for x in exprs
            ]
        )

    def edge_key(memlet: Memlet) -> tuple:
        return (name(memlet.data), rebased(memlet.subset), rebased(memlet.other_subset), str(memlet.wcr))

    roots: list = sorted(gathers[sym] for sym, offset in sym_offset.items() if offset == lane)
    for st in states:
        mine = private.get(st, {})

        def node_key(node: nodes.Node) -> Any:
            if isinstance(node, nodes.AccessNode):
                return ("access", str(node.desc(st.sdfg).dtype)) if node in mine else aliases.get(node.data, node.data)
            if node not in mine:
                return "?"
            used = names_of(node) if isinstance(node, nodes.Tasklet) else None
            return None if used is None or used & {var, *gathers} else node.code.as_string.strip()

        keys = value_keys(st, node_key, edge_key, ids, stored, inputs_of=mine)
        if keys is None:
            return None
        out = [e for n in mine for e in value_edges(st.out_edges(n)) if e.dst not in mine]
        out += [e for e in edges.get(st, []) if e.src not in mine and e.dst not in mine]
        roots.extend(ids.setdefault((keys[e.src], e.src_conn or "", *edge_key(e.data)), len(ids)) for e in out)
    return sorted(roots, key=str)


def lane_spacing(distinct: list[int]) -> int | None:
    """The spacing ``g`` of sorted lane offsets ``[0, g, ..., (m - 1) g]`` with ``m >= 2``, else ``None``."""
    if len(distinct) < 2 or distinct[0] != 0 or distinct[1] <= 0:
        return None
    g = distinct[1]
    return g if distinct == [j * g for j in range(len(distinct))] else None


def drop_isolated_access_nodes(state: SDFGState) -> None:
    """Remove the access nodes a dropped lane left without edges."""
    for n in list(state.nodes()):
        if isinstance(n, nodes.AccessNode) and state.degree(n) == 0:
            state.remove_node(n)


def scalar_accumulator_sink(state: SDFGState) -> nodes.AccessNode | None:
    """The state's only sink, if it is a scalar the state also reads: a reduction with no other output."""
    access = [n for n in state.nodes() if isinstance(n, nodes.AccessNode)]
    sinks = [n for n in access if state.out_degree(n) == 0]
    sources = dict.fromkeys(n.data for n in access if state.in_degree(n) == 0)
    if len(sinks) != 1 or sinks[0].data not in sources:
        return None
    desc = state.sdfg.arrays.get(sinks[0].data)
    return sinks[0] if desc is not None and desc.total_size == 1 else None


def forward_reach(state: SDFGState, lane_edges: dict[int, list]) -> dict[nodes.Node, dict[int, None]]:
    """The lanes whose edges reach each node along out-edges."""
    reached_by: dict[nodes.Node, dict[int, None]] = {}
    for d, edges in lane_edges.items():
        frontier = [e.dst for e in edges]
        seen: dict = {}
        while frontier:
            n = frontier.pop()
            if n in seen:
                continue
            seen[n] = None
            reached_by.setdefault(n, {})[d] = None
            frontier.extend(e.dst for e in state.out_edges(n))
    return reached_by


def steps_alike(loop: LoopRegion, states: list[SDFGState], lane_syms: dict[str, int]) -> bool:
    """Whether no body edge decides, and each assignment but a lane symbol's holds the same value however often it runs:
    the re-rolled body runs ``m`` times as often, and only the lanes account for that."""
    varying = {sym for edge in loop.edges() for sym in edge.data.assignments}
    varying |= {n.data for st in states for n in st.data_nodes() if is_written(st, n)}
    for edge in loop.edges():
        if not edge.data.is_unconditional():
            return False
        for sym, rhs in edge.data.assignments.items():
            reads = symbolic.free_symbols_and_functions(rhs) | symbolic.arrays(rhs)
            if sym not in lane_syms and reads & varying:
                return False
    return True


@dace.properties.make_properties
@explicit_cf_compatible
class RerollUnrolledLoops(ppl.Pass):
    """Re-roll hand-unrolled lane chains (step ``S``, ``m`` equally-spaced lanes)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Re-roll every matching unrolled loop in ``sdfg`` and its nested SDFGs."""
        total = 0
        max_iters = 1 + sum(
            1
            for sd in sdfg.all_sdfgs_recursive()
            for r in sd.all_control_flow_regions(recursive=True)
            if isinstance(r, LoopRegion)
        )
        for _ in range(max_iters):
            rerolled = 0
            for sd in sdfg.all_sdfgs_recursive():
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    if not isinstance(cfg, LoopRegion) or not cfg.loop_variable:
                        continue
                    # One stride read per loop
                    step, undo = unrolled_step(cfg)
                    if step is None or step < 2:
                        continue
                    # The step rewrite rebuilds the bound from the start and the end; a loop without both is left.
                    if loop_analysis.get_loop_end(cfg) is None or loop_analysis.get_init_assignment(cfg) is None:
                        if undo is not None:
                            undo.restore()
                        continue
                    if undo is not None and len(cfg.nodes()) > 1:
                        # Fusing the unscaled body's states cannot be undone, so a copy decides whether any matcher
                        # accepts the fused body before the real body is fused.
                        if not self.rerolls_fused_copy(cfg, step):
                            undo.restore()
                            continue
                        fuse_body_states(cfg)
                    if self.reroll(cfg, step):
                        rerolled += 1
                    elif undo is not None:
                        undo.restore()
            if rerolled == 0:
                break
            total += rerolled
        return total or None

    def reroll(self, loop: LoopRegion, step: int) -> bool:
        return (
            self._try_reroll(loop, step)
            or self._try_reroll_chain(loop, step)
            or self._try_reroll_accumulator_reduction(loop, step)
        )

    def rerolls_fused_copy(self, loop: LoopRegion, step: int) -> bool:
        """Whether ``reroll`` accepts a copy of ``loop`` with its body states fused."""
        memo: dict[int, Any] = {}
        probe = copy.deepcopy(loop.sdfg, memo)
        if not probe.cfg_list:
            probe.reset_cfg_list()
        fuse_body_states(memo[id(loop)])
        return self.reroll(memo[id(loop)], step)

    def _body_states(self, loop: LoopRegion) -> list[SDFGState] | None:
        """The loop's body states when the body is made only of states."""
        blocks = list(loop.bfs_nodes(loop.start_block)) if loop.nodes() else []
        if not blocks or len(blocks) != len(loop.nodes()) or not all(isinstance(b, SDFGState) for b in blocks):
            return None
        return blocks

    def _interstate_lane_symbols(self, loop: LoopRegion, loop_var: str) -> dict[str, int] | None:
        """Per-lane symbols defined on the body's interstate edges."""
        word = re.compile(r"\b%s\b" % re.escape(loop_var))
        sym_offset: dict[str, int] = {}
        for edge in loop.edges():
            for sym, rhs in edge.data.assignments.items():
                if not word.search(str(rhs)):
                    continue
                off = _assignment_offset(str(rhs), loop_var)
                if off is None:
                    return None
                sym_offset[sym] = off
        return sym_offset

    def _edge_offset(self, subset: Subset | None, loop_var: str, sym_offset: dict[str, int]) -> tuple[int | None, bool]:
        """Lane offset of a memlet subset, and whether it is a lane edge at all."""
        if subset is None:
            return None, True
        names = dict.fromkeys(str(s) for s in subset.free_symbols)
        if loop_var in names:
            off = _index_offset(subset, loop_var)
            return (off, off is not None)
        lane_syms = dict.fromkeys(n for n in names if n in sym_offset)
        if lane_syms:
            if len(lane_syms) != 1:
                return None, False
            return sym_offset[next(iter(lane_syms))], True
        return None, True

    #: Sympy class names of associative binary operations recognised as the merge tasklet's reduction op.
    ASSOC_SYMPY_KIND: dict[str, str] = {"Add": "+", "Mul": "*", "Min": "min", "Max": "max"}

    def _associative_op_kind(self, node: nodes.Node) -> str | None:
        """If ``node`` is a binary associative-op tasklet over its two inputs, return its op kind (``'+'``, ``'*'``,
        ``'min'``, ``'max'``); else ``None``."""
        if not isinstance(node, nodes.Tasklet):
            return None
        code = node.code.as_string.strip()
        if "=" not in code:
            return None
        lhs, rhs = code.split("=", 1)
        if lhs.strip() != "__out":
            return None
        try:
            expr = symbolic.pystr_to_symbolic(rhs.strip())
        except Exception:
            return None
        conns = dict.fromkeys(node.in_connectors)
        if len(conns) != 2 or dict.fromkeys(str(s) for s in expr.free_symbols) != conns:
            return None
        # ``__in1 - __in2`` is an ``Add`` of ``__in1`` and ``-__in2``, ``__in1 / __in2`` a ``Mul`` with a power
        if len(expr.args) != 2 or not all(arg.is_Symbol for arg in expr.args):
            return None
        return self.ASSOC_SYMPY_KIND.get(type(expr).__name__)

    def _is_transparent_spine(self, state: SDFGState, node: nodes.Node) -> bool:
        """Whether ``node`` carries a value through the reduction without itself being a lane-unique computation or an
        associative merge op."""
        if isinstance(node, nodes.AccessNode):
            desc = state.sdfg.arrays.get(node.data)
            return desc is not None and desc.transient
        if isinstance(node, nodes.Tasklet):
            code = node.code.as_string.strip()
            if "=" not in code:
                return False
            lhs, rhs = code.split("=", 1)
            if lhs.strip() != "__out":
                return False
            try:
                expr = symbolic.pystr_to_symbolic(rhs.strip())
            except Exception:
                return False
            return dict.fromkeys(str(s) for s in expr.free_symbols) == dict.fromkeys(["__inp"])
        return False

    def _carried_accumulator_op(self, state: SDFGState, acc: str) -> str | None:
        """Op kind of a lane's ``acc = acc OP x`` update, or ``None`` if it is not that shape."""
        acc_nodes = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == acc]
        src = [n for n in acc_nodes if state.in_degree(n) == 0]
        snk = [n for n in acc_nodes if state.out_degree(n) == 0]
        if len(acc_nodes) != 2 or len(src) != 1 or len(snk) != 1 or state.out_degree(src[0]) != 1:
            return None
        node = state.out_edges(src[0])[0].dst
        op = self._associative_op_kind(node)
        if op is None:
            return None
        while node is not snk[0]:
            out = state.out_edges(node)
            if len(out) != 1:
                return None
            node = out[0].dst
            if node is not snk[0] and not (self._is_transparent_spine(state, node) and state.in_degree(node) == 1):
                return None
        return op

    def _accumulators_folded_once(self, loop: LoopRegion, accs: dict[str, None], op: str) -> bool:
        """Whether the loop's per-lane accumulators are consumed by ONE binary ``op`` tree."""
        sdfg = loop.sdfg
        body = dict.fromkeys(loop.all_control_flow_blocks(recursive=True))
        for e in sdfg.all_interstate_edges(recursive=True):
            syms = e.data.free_symbols  # once per edge, not once per (edge, accumulator)
            if any(a in syms for a in accs):
                return False
        leaves: dict = {}
        seen_names: dict[str, None] = {}
        host: dict[SDFGState, None] = {}
        for st in sdfg.states():
            if st in body:
                continue
            for n in st.nodes():
                if not (isinstance(n, nodes.AccessNode) and n.data in accs and st.out_degree(n) > 0):
                    continue
                # An unroll REMAINDER loop accumulating into the same lane with the same op does
                # not consume the folded value, it contributes one more term to it.
                if self._carried_accumulator_op(st, n.data) == op:
                    continue
                if st.out_degree(n) != 1 or n.data in seen_names:
                    return False
                leaves[n] = None
                seen_names[n.data] = None
                host[st] = None
        if len(leaves) != len(accs) or len(host) != 1:
            return False
        state = next(iter(host))
        # Walk forward from each accumulator leaf through single-consumer binary ``op`` tasklets
        # and state-local transient scalars until leaving the ``op``-tree. Because ``op`` is
        # associative and commutative (+, *, min, max), all ``m`` accumulators fold together
        # soundly -- even inside a larger same-op fold such as ``(s0 + ... + s10) + tail`` --
        # iff every leaf enters at least one ``op`` tasklet and every leaf reaches the SAME final
        # ``op`` tasklet before exiting (to a non-``op`` consumer, a multi-consumer node, a
        # non-transient container, or a transient read in another state).
        used_outside = {
            n.data for st in sdfg.states() if st is not state for n in st.nodes() if isinstance(n, nodes.AccessNode)
        }
        root_ops: dict = {}
        for leaf in leaves:
            cur: nodes.Node = leaf
            last_op: nodes.Node | None = None
            while True:
                outs = state.out_edges(cur)
                if len(outs) != 1:
                    break
                t = outs[0].dst
                if self._associative_op_kind(t) != op or state.in_degree(t) != 2 or state.out_degree(t) != 1:
                    break
                nxt = state.out_edges(t)[0].dst
                if not isinstance(nxt, nodes.AccessNode):
                    return False
                last_op = t
                cur = nxt
                desc = sdfg.arrays.get(cur.data)
                if (desc is not None and not desc.transient) or cur.data in used_outside:
                    break
            if last_op is None:
                return False
            root_ops[last_op] = None
        return len(root_ops) == 1

    def _shared_nodes(self, state: SDFGState) -> dict[nodes.Node, None]:
        """Boundary access nodes of a state (external arrays + read-only sources)."""
        shared: dict[nodes.Node, None] = {}
        for n in state.nodes():
            if not isinstance(n, nodes.AccessNode):
                continue
            desc = state.sdfg.arrays.get(n.data)
            # Value edges only: an ordering memlet is no producer, so it makes no lane boundary.
            if (
                (desc is not None and not desc.transient)
                or not value_edges(state.in_edges(n))
                or not value_edges(state.out_edges(n))
            ):
                shared[n] = None
        return shared

    def _lane_nodes(
        self, state: SDFGState, lane_edges: list[MultiConnectorEdge[Memlet]], shared: dict[nodes.Node, None]
    ) -> dict[nodes.Node, None]:
        """Internal (non-shared) nodes reachable from a lane's boundary edges."""
        seen: dict[nodes.Node, None] = {}
        frontier: list[nodes.Node] = []
        for e in lane_edges:
            frontier.extend(n for n in (e.src, e.dst) if n not in shared)
        while frontier:
            n = frontier.pop()
            if n in seen:
                continue
            seen[n] = None
            frontier.extend(e.src for e in value_edges(state.in_edges(n)) if e.src not in shared)
            frontier.extend(e.dst for e in value_edges(state.out_edges(n)) if e.dst not in shared)
        return seen

    def _try_reroll(self, loop: LoopRegion, step: int) -> bool:
        """Attempt to re-roll one loop; return whether it was rerolled."""
        loop_var = loop.loop_variable
        states = self._body_states(loop)
        if states is None:
            return False
        sym_offset = self._interstate_lane_symbols(loop, loop_var)
        if sym_offset is None or not steps_alike(loop, states, sym_offset):
            return False

        edge_offsets: dict[SDFGState, dict[MultiConnectorEdge[Memlet], int]] = {}
        for st in states:
            offsets = self.edge_lane_offsets(st, loop_var, sym_offset, reads_only=False)
            if offsets is None:
                return False
            edge_offsets[st] = offsets

        all_offsets = dict.fromkeys(sym_offset.values())
        for st in states:
            all_offsets.update(dict.fromkeys(edge_offsets[st].values()))
        if not all_offsets:
            return False

        distinct = sorted(all_offsets)
        m = len(distinct)
        g = lane_spacing(distinct)
        if g is None:
            return False

        # Position-coverage safety: ``step > m*g`` leaves gaps the re-rolled step-``g`` loop would wrongly fill; ``step
        # < m*g`` overlaps, so a position is touched by more than one lane -- safe only when no array is both read and
        # written (else the re-roll changes a read-modify-write count).
        if step > m * g or step % g != 0:
            return False
        if step < m * g and self._has_read_modify_write(states, edge_offsets):
            return False

        shared = {st: self._shared_nodes(st) for st in states}
        lane_edges: dict[int, dict[SDFGState, list]] = {d: {st: [] for st in states} for d in distinct}
        for st in states:
            for edge, off in edge_offsets[st].items():
                lane_edges[off][st].append(edge)
        lane_nodes = self.lane_partition(states, lane_edges, shared)
        if lane_nodes is None:
            return False
        acc = self.carried_accumulators(loop, states, lane_nodes, shared, edge_offsets)
        # Overlapping lanes fold a shared position into one accumulator per covering lane; the re-roll folds it once.
        if acc is None or (acc and step != m * g):
            return False

        private = {d: {st: lane_nodes[st][d] for st in states} for d in distinct}
        if not self._lanes_match(
            loop, states, private, lane_edges, {name: "acc" for name, _ in acc.values()}, sym_offset
        ):
            return False
        if loop_analysis.read_outside(loop.sdfg, loop, set(sym_offset), self._scratch(private)):
            return False

        # Re-roll: drop every lane but offset 0
        drop_offsets = dict.fromkeys(distinct[1:])
        for st in states:
            for d in drop_offsets:
                for edge in lane_edges[d][st]:
                    if edge in st.edges():
                        st.remove_edge(edge)
                for n in lane_nodes[st][d]:
                    if n in st.nodes():
                        st.remove_node(n)
            drop_isolated_access_nodes(st)
        drop_syms = [s for s, o in sym_offset.items() if o in drop_offsets]
        for edge in loop.edges():
            for s in drop_syms:
                edge.data.assignments.pop(s, None)

        self._rewrite_step(loop, loop_var, g, m)
        return True

    def lane_partition(
        self, states: list[SDFGState], lane_edges: dict[int, dict[SDFGState, list]], shared: dict
    ) -> dict[SDFGState, dict[int, dict]] | None:
        """Per state, the nodes each lane's boundary edges reach, or ``None`` if two lanes meet at a node or a
        non-access node belongs to no lane (the re-rolled body runs ``m`` times as often)."""
        lane_nodes: dict[SDFGState, dict[int, dict]] = {}
        for st in states:
            visited_by: dict = {}
            for d, per_state in lane_edges.items():
                for n in self._lane_nodes(st, per_state[st], shared[st]):
                    visited_by.setdefault(n, {})[d] = None
            per_lane: dict[int, dict] = {d: {} for d in lane_edges}
            for n, lanes_visiting in visited_by.items():
                if len(lanes_visiting) != 1:
                    return None
                per_lane[next(iter(lanes_visiting))][n] = None
            if any(n not in visited_by for n in st.nodes() if not isinstance(n, nodes.AccessNode)):
                return None
            lane_nodes[st] = per_lane
        return lane_nodes

    def carried_accumulators(
        self,
        loop: LoopRegion,
        states: list[SDFGState],
        lane_nodes: dict[SDFGState, dict[int, dict]],
        shared: dict,
        edge_offsets: dict[SDFGState, dict[MultiConnectorEdge[Memlet], int]],
    ) -> dict[int, tuple[str, SDFGState]] | None:
        """Per lane, its carried accumulator of the ``s0 OP= f(i); s1 OP= f(i+g); s = s0 OP s1`` shape; ``{}`` if
        no lane carries one, ``None`` if only some do or the accumulators are not folded by one ``OP`` tree."""
        carried: dict[int, dict[str, SDFGState]] = {d: {} for d in lane_nodes[states[0]]}
        for st in states:
            for d, lane in lane_nodes[st].items():
                for n in lane:
                    for e in st.out_edges(n):
                        if isinstance(e.dst, nodes.AccessNode) and e.dst in shared[st] and e not in edge_offsets[st]:
                            carried[d][e.dst.data] = st
        acc: dict[int, tuple[str, SDFGState]] = {}
        for d, names_in_lane in carried.items():
            if len(names_in_lane) > 1:
                return None
            for name, st in names_in_lane.items():
                acc[d] = (name, st)
        if not acc:
            return acc
        m = len(carried)
        names = dict.fromkeys(name for name, unused_state in acc.values())
        if len(acc) != m or len(names) != m:
            return None
        ops = dict.fromkeys(self._carried_accumulator_op(st, name) for name, st in acc.values())
        if len(ops) != 1 or next(iter(ops)) is None:
            return None
        return acc if self._accumulators_folded_once(loop, names, next(iter(ops))) else None

    def _cut_nodes(self, state: SDFGState) -> dict[nodes.Node, None]:
        """AccessNodes of the loop's VISIBLE containers -- where the value walk stops."""
        arrays = state.sdfg.arrays
        return dict.fromkeys(
            n
            for n in state.nodes()
            if isinstance(n, nodes.AccessNode) and not (n.data in arrays and arrays[n.data].transient)
        )

    def _cut_side_subset(self, edge: MultiConnectorEdge[Memlet], cut_node: nodes.AccessNode) -> Subset | None:
        """The subset of ``edge`` that indexes ``cut_node``'s container."""
        mem = edge.data
        if mem is None or mem.data is None:
            return None
        return mem.subset if mem.data == cut_node.data else mem.other_subset

    def _topological_index(self, state: SDFGState) -> dict[int, int]:
        """Position of every node of ``state`` in a topological order, by node id."""
        try:
            order = list(sdutil.dfs_topological_sort(state))
        except Exception:
            order = []
        idx: dict[int, int] = {}
        for n in order:
            idx.setdefault(state.node_id(n), len(idx))
        for n in state.nodes():
            idx.setdefault(state.node_id(n), len(idx))
        return idx

    def _value_components(
        self, states: list[SDFGState], cut: list[dict[nodes.Node, None]]
    ) -> list[list[tuple[int, nodes.Node]]]:
        """Partition the body's non-boundary nodes into connected VALUE components."""
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        def find(key: tuple[int, int]) -> tuple[int, int]:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(a: tuple[int, int], b: tuple[int, int]) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for si, st in enumerate(states):
            for n in st.nodes():
                if n not in cut[si]:
                    key = (si, st.node_id(n))
                    parent[key] = key
        for si, st in enumerate(states):
            for e in st.edges():
                if e.data is None or e.data.is_empty() or e.src in cut[si] or e.dst in cut[si]:
                    continue
                union((si, st.node_id(e.src)), (si, st.node_id(e.dst)))
        first_seen: dict[str, tuple[int, int]] = {}
        for si, st in enumerate(states):
            for n in st.nodes():
                if not isinstance(n, nodes.AccessNode) or n in cut[si]:
                    continue
                key = (si, st.node_id(n))
                union(first_seen.setdefault(n.data, key), key)
        groups: dict[tuple[int, int], list[tuple[int, nodes.Node]]] = {}
        for key in sorted(parent):
            groups.setdefault(find(key), []).append((key[0], states[key[0]].node(key[1])))
        return list(groups.values())

    def _try_reroll_chain(self, loop: LoopRegion, step: int) -> bool:
        """Re-roll a lane chain whose lanes read past their own position (TSVC s116)."""
        loop_var = loop.loop_variable
        states = self._body_states(loop)
        if states is None:
            return False
        # A per-lane interstate symbol names the offset of a lane's READ, not of its store
        if self._interstate_lane_symbols(loop, loop_var) or not steps_alike(loop, states, {}):
            return False

        cut = [self._cut_nodes(st) for st in states]
        comps = self._value_components(states, cut)
        bounds = self.component_boundaries(states, cut, comps) if comps else None
        found = self.chain_lanes(states, bounds, loop_var) if bounds is not None else None
        if found is None:
            return False
        lanes, order = found
        distinct = sorted(lanes)
        m = len(distinct)
        g = lane_spacing(distinct)
        if g is None:
            return False
        # Exact coverage only: a gap would leave positions the step-``g`` loop wrongly fills, an
        # overlap would store a position twice, and both break the read-ahead argument above.
        if step != m * g:
            return False
        private = {d: {st: {} for st in states} for d in distinct}
        for d in distinct:
            for si, node in comps[lanes[d]]:
                private[d][states[si]][node] = None
        if not self._lanes_match(loop, states, private, {d: {} for d in distinct}, {}, {}):
            return False
        if sorted(distinct, key=lambda d: order[d]) != distinct:
            return False
        if loop_analysis.read_outside(loop.sdfg, loop, set(), self._scratch(private)):
            return False

        for d in distinct[1:]:
            for si, node in comps[lanes[d]]:
                st = states[si]
                if node in st.nodes():
                    st.remove_node(node)
        for st in states:
            drop_isolated_access_nodes(st)
        self._rewrite_step(loop, loop_var, g, m)
        return True

    def component_boundaries(
        self, states: list[SDFGState], cut: list[dict[nodes.Node, None]], comps: list[list[tuple[int, nodes.Node]]]
    ) -> list[list] | None:
        """Per value component, its ``(state index, edge, container node, is_read)`` edges to the visible
        containers, or ``None`` for a bare container-to-container copy, which belongs to no lane."""
        owner: dict[tuple[int, int], int] = {}
        for ci, comp in enumerate(comps):
            for si, node in comp:
                owner[(si, states[si].node_id(node))] = ci
        bounds: list[list] = [[] for _ in comps]
        for si, st in enumerate(states):
            for e in st.edges():
                if e.data is None or e.data.is_empty():
                    continue
                is_read, stores = e.src in cut[si], e.dst in cut[si]
                if is_read and stores:
                    return None
                if not is_read and not stores:
                    continue
                inner = e.dst if is_read else e.src
                bounds[owner[(si, st.node_id(inner))]].append((si, e, e.src if is_read else e.dst, is_read))
        return bounds

    def chain_lanes(
        self, states: list[SDFGState], bounds: list[list], loop_var: str
    ) -> tuple[dict[int, int], dict[int, tuple[int, int]]] | None:
        """The component of each lane by its store offset, and the position of each lane's first store; ``None``
        if a component stores at no single offset or two components store at one."""
        topo = [self._topological_index(st) for st in states]
        lanes: dict[int, int] = {}
        order: dict[int, tuple[int, int]] = {}
        for ci, boundary in enumerate(bounds):
            offsets = []
            for si, e, node, is_read in boundary:
                off, ok = self._edge_offset(self._cut_side_subset(e, node), loop_var, {})
                # A loop-invariant READ (a scalar coefficient) is shared by every lane and stays
                if not ok or (off is None and not is_read):
                    return None
                offsets.append((is_read, off, si, e))
            stored_at = dict.fromkeys(off for is_read, off, unused_si, unused_edge in offsets if not is_read)
            if len(stored_at) != 1:
                return None
            d = next(iter(stored_at))
            if d in lanes:
                return None
            lanes[d] = ci
            order[d] = min(
                (si, topo[si][states[si].node_id(e.dst)]) for is_read, unused_off, si, e in offsets if not is_read
            )
        return lanes, order

    def _try_reroll_accumulator_reduction(self, loop: LoopRegion, step: int) -> bool:
        """Re-roll a hand-unrolled reduction (TSVC s352, s31111) into one step-``g`` lane."""
        loop_var = loop.loop_variable
        states = self._body_states(loop)
        if states is None or len(states) != 1:
            return False
        st = states[0]
        # A pure in-state reduction: no per-lane interstate gather symbols.
        if self._interstate_lane_symbols(loop, loop_var) or not steps_alike(loop, states, {}):
            return False

        read_offsets = self.edge_lane_offsets(st, loop_var, {}, reads_only=True)
        if read_offsets is None:
            return False
        per_lane_edges: dict[int, list] = {}
        for edge, off in read_offsets.items():
            per_lane_edges.setdefault(off, []).append(edge)
        distinct = sorted(per_lane_edges)
        m = len(distinct)
        g = lane_spacing(distinct)
        if g is None:
            return False
        # Contiguous, no-overlap coverage: each position read exactly once.
        if step != m * g:
            return False

        sink = scalar_accumulator_sink(st)
        if sink is None:
            return False
        reached_by = forward_reach(st, {d: per_lane_edges[d] for d in distinct})
        fold_op = self.single_fold_op(st, reached_by)
        spine = self._fold_spine(st, sink, fold_op) if fold_op is not None else None
        if spine is None:
            return False

        # Private: reached from one offset, off the accumulator's spine.
        private = {d: {st: {} for st in states} for d in distinct}
        for n, offs in reached_by.items():
            if len(offs) == 1 and n not in spine:
                private[next(iter(offs))][st][n] = None
        if not self._lanes_match(loop, states, private, {d: {st: per_lane_edges[d]} for d in distinct}, {}, {}):
            return False
        if loop_analysis.read_outside(loop.sdfg, loop, set(), self._scratch({0: {st: reached_by}}) - {sink.data}):
            return False

        # Rewrite.
        drop_offsets = dict.fromkeys(distinct[1:])  # already sorted -- dict.fromkeys keeps that order
        for d in distinct[1:]:
            for edge in per_lane_edges[d]:
                if edge in st.edges():
                    st.remove_edge(edge)
        for n in list(st.nodes()):
            offs = reached_by.get(n)
            if offs and all(o in drop_offsets for o in offs):
                st.remove_node(n)
        self._collapse_merge_tree(st, dict.fromkeys(n for n in st.nodes() if self._associative_op_kind(n) == fold_op))
        drop_isolated_access_nodes(st)

        self._rewrite_step(loop, loop_var, g, m)
        return True

    def edge_lane_offsets(
        self, state: SDFGState, loop_var: str, sym_offset: dict[str, int], reads_only: bool
    ) -> dict[MultiConnectorEdge[Memlet], int] | None:
        """The lane offset of every loop-dependent edge (of every read from an access node, ``reads_only``), or
        ``None`` if an edge's subset has no single offset."""
        offsets: dict[MultiConnectorEdge[Memlet], int] = {}
        for edge in state.edges():
            if edge.data is None or (reads_only and not isinstance(edge.src, nodes.AccessNode)):
                continue
            off, ok = self._edge_offset(edge.data.subset, loop_var, sym_offset)
            if not ok:
                return None
            if off is not None:
                offsets[edge] = off
        return offsets

    def single_fold_op(self, state: SDFGState, reached_by: dict) -> str | None:
        """The one associative op every node two lanes reach merges with (transparent spine carriers aside)."""
        fold_op: str | None = None
        for n, offs in reached_by.items():
            if len(offs) < 2:
                continue
            op = self._associative_op_kind(n)
            if op is None:
                if self._is_transparent_spine(state, n):
                    continue
                return None
            if fold_op is not None and fold_op != op:
                return None
            fold_op = op
        return fold_op

    def _fold_spine(self, state: SDFGState, sink: nodes.AccessNode, op: str) -> dict[nodes.Node, None] | None:
        """The nodes from the accumulator's read to ``sink``, or ``None`` if it passes anything but ``op`` folds."""
        reads = [
            n for n in state.data_nodes() if n.data == sink.data and state.in_degree(n) == 0 and state.out_degree(n) > 0
        ]
        if len(reads) != 1 or state.out_degree(reads[0]) != 1:
            return None
        spine: dict[nodes.Node, None] = {}
        node = state.out_edges(reads[0])[0].dst
        while node is not sink:
            if (self._associative_op_kind(node) != op and not self._is_transparent_spine(state, node)) or (
                state.out_degree(node) != 1
            ):
                return None
            spine[node] = None
            node = state.out_edges(node)[0].dst
        return spine

    def _lanes_match(
        self,
        loop: LoopRegion,
        states: list[SDFGState],
        private: dict[int, dict],
        edges: dict[int, dict],
        aliases: dict[str, str],
        sym_offset: dict[str, int],
    ) -> bool:
        """Whether all lanes have lane 0's non-empty :func:`lane_signature`."""
        ids: ValueIds = {}
        gathers = {
            sym: str(rhs)[: str(rhs).rindex("[")] if "[" in str(rhs) else ""
            for edge in loop.edges()
            for sym, rhs in edge.data.assignments.items()
            if sym in sym_offset
        }
        written = {n.data for st in states for n in st.data_nodes() if is_written(st, n) and n.desc(st.sdfg).transient}
        sigs = [
            lane_signature(loop, states, d, private[d], edges[d], aliases, sym_offset, gathers, written, ids)
            for d in private
        ]
        return bool(sigs[0]) and all(sig == sigs[0] for sig in sigs[1:])

    def _scratch(self, private: dict[int, dict]) -> set[str]:
        """The containers of the lanes' private nodes: a read of one after the loop would see another position's
        value, or none at all if its lane is dropped."""
        return {
            n.data
            for per_state in private.values()
            for lane in per_state.values()
            for n in lane
            if isinstance(n, nodes.AccessNode)
        }

    def _collapse_merge_tree(self, state: SDFGState, merges: dict[nodes.Node, None]) -> None:
        """Splice each surviving merge tasklet to passthrough its one live input."""
        # Process merges in topological order (closer to the leaves first) so a
        # later merge sees its already-collapsed predecessors as direct edges.
        try:
            from dace.sdfg.utils import dfs_topological_sort

            order = [n for n in dfs_topological_sort(state) if n in merges]
        except Exception:
            # ``merges`` is a set of NODE objects
            order = sorted(merges, key=state.node_id)
        for m in order:
            if m not in state.nodes():
                continue
            in_edges = list(state.in_edges(m))
            out_edges = list(state.out_edges(m))
            # A live merge has exactly one of its 2 inputs still wired (the other lane's chain was just removed).
            if len(in_edges) != 1 or len(out_edges) != 1:
                continue
            live_in = in_edges[0]
            out_acc = out_edges[0].dst
            if not isinstance(out_acc, nodes.AccessNode):
                continue
            # Redirect every consumer of ``out_acc`` to read from ``live_in.src`` with that source's port.
            for ce in list(state.out_edges(out_acc)):
                new_memlet = ce.data
                if (
                    new_memlet is not None
                    and isinstance(live_in.src, nodes.AccessNode)
                    and new_memlet.data == out_acc.data
                ):
                    new_memlet = dace.Memlet(
                        data=live_in.src.data, subset=copy.deepcopy(new_memlet.subset), wcr=new_memlet.wcr
                    )
                state.remove_edge(ce)
                state.add_edge(live_in.src, live_in.src_conn, ce.dst, ce.dst_conn, new_memlet)
            state.remove_node(m)
            if state.degree(out_acc) == 0:
                state.remove_node(out_acc)

    def _has_read_modify_write(
        self, states: list[SDFGState], edge_offsets: dict[SDFGState, dict[MultiConnectorEdge[Memlet], int]]
    ) -> bool:
        """Whether any array is both read and written by the lanes."""
        reads: dict = {}
        writes: dict = {}
        for st in states:
            for edge in edge_offsets[st]:
                if isinstance(edge.src, nodes.AccessNode):
                    reads[edge.src.data] = None
                if isinstance(edge.dst, nodes.AccessNode):
                    writes[edge.dst.data] = None
        return any(a in writes for a in reads)

    @staticmethod
    def simplify_int_floor(num: symbolic.SymbolicType, step: symbolic.SymbolicType) -> symbolic.SymbolicType:
        """Pull exact integer multiples of positive integer ``step`` out of ``int_floor(num, step)``.

        For every integer ``a``, integer ``q``, and integer ``k >= 1``:
        ``int_floor(k * q + a, k) == q + int_floor(a, k)``.
        Thus ``int_floor(7 * NBLK - 1, 7)`` simplifies to ``NBLK - 1`` and
        ``int_floor(N - 17, 17)`` simplifies to ``int_floor(N, 17) - 1``, while an unaligned
        tail such as ``int_floor(N - 1, 5)`` stays intact.
        """
        import sympy

        if not (step.is_Integer and int(step) > 0):
            return symbolic.int_floor(num, step)
        k = int(step)
        expanded = sympy.expand(as_expr(num))
        terms = sympy.Add.make_args(expanded)
        quot_terms: list[sympy.Expr] = []
        rem_nonconst: list[sympy.Expr] = []
        const_total = 0
        for term in terms:
            if term.is_Integer:
                const_total += int(term)
                continue
            coeff, rest = term.as_coeff_Mul()
            if coeff.is_Integer:
                q, r = divmod(int(coeff), k)
                if q != 0:
                    quot_terms.append(q * rest)
                if r != 0:
                    rem_nonconst.append(r * rest)
            else:
                rem_nonconst.append(term)
        if not rem_nonconst:
            return sympy.Add(*quot_terms) + (const_total // k)
        # Keep small non-multiple offsets like ``N - 1`` readable; pull out full multiples of ``k``.
        if const_total != 0 and (const_total % k == 0 or abs(const_total) >= k):
            q_c, r_c = divmod(const_total, k)
            quot_terms.append(sympy.Integer(q_c))
            const_total = r_c
        rem = sympy.Add(*rem_nonconst, sympy.Integer(const_total))
        return sympy.Add(*quot_terms) + symbolic.int_floor(rem, step)

    def _rewrite_step(self, loop: LoopRegion, loop_var: str, g: int, m: int) -> None:
        """Rewrite a step-``S`` loop to step ``g`` over the flattened range."""
        loop_end = symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(loop))
        init = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(loop))
        step = symbolic.pystr_to_symbolic(loop_analysis.get_loop_stride(loop))
        quot = self.simplify_int_floor(as_expr(loop_end) - as_expr(init), step)
        last_i = init + step * quot
        new_excl = symbolic.simplify(as_expr(last_i) + m * g)
        loop.update_statement = dace.properties.CodeBlock(f"{loop_var} = {loop_var} + {g}")
        loop.loop_condition = dace.properties.CodeBlock(f"{loop_var} < ({symbolic.symstr(new_excl)})")
