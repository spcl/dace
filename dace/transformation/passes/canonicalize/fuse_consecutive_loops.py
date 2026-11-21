# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse two consecutive loops with identical bodies over adjacent index ranges.

A hand-tiled loop nest arrives split into a *main-body* loop over the largest
multiple-of-``K`` prefix and a step-1 *remainder* loop over the trailing
``< K`` elements::

    for i in range(0, (N // K) * K):   # main body (after re-roll: unit stride)
        acc += a[i]
    for i in range((N // K) * K, N):   # remainder
        acc += a[i]

Both loops run the *same* body over *adjacent* index ranges, so together they
sweep ``range(0, N)``. Left split, a reduction like the one above lifts to two
separate ``Reduce`` library nodes writing the same accumulator -- a shape whose
seed does not chain across the two nodes (the second re-seeds the accumulator
and drops the first's partial sum). Fusing the two loops back into one
``for i in range(0, N)`` removes the split so a single ``Reduce`` is lifted and
the result is exact.

Concatenating two adjacent iteration ranges ``[A, B)`` and ``[B, C)`` under the
*same* stride and the *same* body executes the body for the same index sequence
in the same order as a single loop over their union, so every loop-carried value
is identical. That union is ``[A, C)`` only for ``A <= B <= C``; an empty side
makes it ``[min(A, B), max(B, C))``, which holds in every case and is what the
fused loop runs when the order is not provable. The match is deliberately
conservative -- both loops must be
single-state, unit-stride, directly consecutive (one plain interstate edge
between them, nothing else), and structurally identical up to their iteration
variable -- so it fires only on the re-rolled tile/remainder shape and its kin,
never on unrelated adjacent loops.
"""

import re
from collections.abc import Callable, Container
from typing import Any

import sympy

import dace
from dace import symbolic
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import Edge
from dace.sdfg.narrowing import as_expr
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible

#: Per-SDFG context: data name -> states referencing an AccessNode for it. A merge deletes a
#: whole body state, which changes this, so it is rebuilt once per sweep rather than cached
#: across an ``apply`` (see ``apply_pass``); never rebuilt per candidate pair.
ScratchIndex = dict[str, OrderedSet[SDFGState]]

#: Iterator- and scratch-name-independent key of a body node: ``(kind, payload)``.
NodeKey = tuple[str, str]
#: Interned value keys: two nodes with one id compute one value, in any body they appear in.
ValueIds = dict[tuple, int]

#: Placeholder the iteration variable is normalised to when comparing two loop
#: bodies, so ``a[_loop_it_0]`` and ``a[_loop_it_1]`` compare equal.
ITER_PLACEHOLDER = "__lv__"

#: Placeholder every body-local scratch transient name is normalised to, so two
#: bodies differing only in a frontend-generated intermediate name (e.g.
#: ``s0_plus_a_slice`` vs ``s0_plus_a_slice_0``) compare equal. The carried
#: accumulator and the read/written arrays -- which are referenced OUTSIDE the
#: body and so are not body-local -- keep their real names and must match.
SCRATCH_PLACEHOLDER = "__scratch__"


def _ceil_to_int_floor(expr: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """Rewrite DaCe ``int_ceil(a, k)`` with positive integer constant ``k`` to ``int_floor(a + k - 1, k)``.

    Exact for all integers ``a`` and ``k >= 1``; lets :func:`dace.symbolic.provably_le` reason
    over ``WhileToFor``'s exit values (e.g. ``17 * Max(0, int_ceil(N - 16, 17))``) using its
    existing ``int_floor`` rules.
    """
    return expr.replace(
        lambda x: x.func.__name__ == "int_ceil" and len(x.args) == 2 and x.args[1].is_Integer and int(x.args[1]) > 0,
        lambda x: symbolic.int_floor(x.args[0] + x.args[1] - 1, x.args[1]),
    )


def _int_floor_to_sympy(expr: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """Rewrite DaCe ``int_floor`` / ``int_ceil`` sub-terms as SymPy ``floor`` / ``ceiling``.

    ``int_floor`` and ``int_ceil`` are opaque to :func:`sympy.simplify`, so genuine identities
    such as ``int_floor(N - 11, 11) + 1 == int_floor(N, 11)`` or
    ``int_ceil(N - 16, 17) == int_floor(N, 17)`` are left unproven. SymPy's ``floor`` knows
    ``floor(x + k) == floor(x) + k`` for integer ``k``, which is what an adjacency check between
    a tile bound and a remainder start needs.
    """
    expr = _ceil_to_int_floor(expr)
    expr = expr.replace(
        lambda x: x.func.__name__ == "int_floor" and len(x.args) == 2, lambda x: sympy.floor(x.args[0] / x.args[1])
    )
    return expr.replace(
        lambda x: x.func.__name__ == "int_ceil" and len(x.args) == 2, lambda x: sympy.ceiling(x.args[0] / x.args[1])
    )


def _simplify_extrema(expr: symbolic.SymbolicType, facts: symbolic.Facts | None) -> symbolic.SymbolicType:
    """Drop ``Max`` / ``Min`` operands that ``facts`` proves redundant (e.g. ``Max(0, int_floor(N, 17))``)."""
    expr = _ceil_to_int_floor(as_expr(expr))
    if facts is None or not expr.atoms(sympy.Max, sympy.Min):
        return expr

    def _collapse_max(node: sympy.Max) -> sympy.Expr:
        if len(node.args) == 2:
            a, b = node.args
            if provably_ordered(a, b, facts):
                return b
            if provably_ordered(b, a, facts):
                return a
        return node

    def _collapse_min(node: sympy.Min) -> sympy.Expr:
        if len(node.args) == 2:
            a, b = node.args
            if provably_ordered(a, b, facts):
                return a
            if provably_ordered(b, a, facts):
                return b
        return node

    expr = expr.replace(lambda x: isinstance(x, sympy.Max), _collapse_max)
    expr = expr.replace(lambda x: isinstance(x, sympy.Min), _collapse_min)
    return expr


def _symbolically_equal(
    a: symbolic.SymbolicType, b: symbolic.SymbolicType, facts: symbolic.Facts | None = None
) -> bool:
    """Whether two symbolic expressions are provably equal, floor/ceil/facts-aware.

    :param a: First expression.
    :param b: Second expression.
    :param facts: Optional scope symbol facts (e.g. ``N >= 0`` for ``Max(0, int_ceil(...))``).
    :returns: ``True`` iff ``a - b`` simplifies to zero, else ``False``.
    """
    try:
        # The two expressions come from different sources (two memlets, a subset against a reparsed
        # bound), so a shared name can carry two symbol instances with different dtypes. Subtraction
        # goes through identity, not name, and would leave ``i - i`` uncancelled.
        pa, pb = symbolic.equalize_symbols_across(symbolic.pystr_to_symbolic(a), symbolic.pystr_to_symbolic(b))
        diff = symbolic.simplify(as_expr(pa) - as_expr(pb))
        if diff == 0:
            return True
        if symbolic.simplify(_int_floor_to_sympy(diff)) == 0:
            return True
        if facts is not None:
            sa = _simplify_extrema(pa, facts)
            sb = _simplify_extrema(pb, facts)
            sdiff = symbolic.simplify(as_expr(sa) - as_expr(sb))
            if sdiff == 0 or symbolic.simplify(_int_floor_to_sympy(sdiff)) == 0:
                return True
        return False
    except Exception:
        return False


def value_keys(
    state: SDFGState,
    node_key: Callable[[nodes.Node], Any],
    edge_key: Callable[[dace.Memlet], tuple],
    ids: ValueIds,
    stored: dict[str, int],
    inputs_of: Container[nodes.Node] | None = None,
) -> dict[nodes.Node, int] | None:
    """An interned key per node of ``state``, equal only for nodes computing one value: it folds in the keys of the
    node's sources, wired by connector, so one set of tasklets and memlets connected differently differs.

    :param node_key: What a node is by itself; ``None`` refuses.
    :param edge_key: What a memlet reads.
    :param ids: The interning table, shared by everything compared.
    :param stored: Per container, the key of its last write; carries values into the next state.
    :param inputs_of: The nodes whose sources count, all if ``None``; the key of any other is ``node_key`` alone.
    :returns: The keys, or ``None`` if ``node_key`` refused.
    """
    keys: dict[nodes.Node, int] = {}
    for node in sdutil.dfs_topological_sort(state):
        payload = node_key(node)
        if payload is None:
            return None
        inputs = []
        if inputs_of is None or node in inputs_of:
            # A memlet-path edge has None for a connector, a View's 'views'.
            inputs = [
                (keys[e.src], e.src_conn or "", e.dst_conn or "", *edge_key(e.data))
                for e in state.in_edges(node)
                if not e.data.is_empty()
            ]
        if not inputs and isinstance(node, nodes.AccessNode) and node.data in stored:
            keys[node] = stored[node.data]  # a read of what an access node before it wrote
            continue
        keys[node] = ids.setdefault((payload, tuple(sorted(inputs))), len(ids))
        if inputs and isinstance(node, nodes.AccessNode):
            stored[node.data] = keys[node]
    return keys


def provably_ordered(low: symbolic.SymbolicType, high: symbolic.SymbolicType, facts: symbolic.Facts) -> bool:
    """Whether ``low <= high`` is provable, floor/ceil-aware."""
    low, high = symbolic.equalize_symbols_across(as_expr(low), as_expr(high))
    if symbolic.provably_le(low, high, facts):
        return True
    nlow, nhigh = _ceil_to_int_floor(low), _ceil_to_int_floor(high)
    if (nlow is not low or nhigh is not high) and symbolic.provably_le(nlow, nhigh, facts):
        return True
    if nlow.atoms(sympy.Max, sympy.Min) or nhigh.atoms(sympy.Max, sympy.Min):
        slow = _simplify_extrema(nlow, facts)
        shigh = _simplify_extrema(nhigh, facts)
        if (slow != nlow or shigh != nhigh) and symbolic.provably_le(slow, shigh, facts):
            return True
    return bool(symbolic.simplify(_int_floor_to_sympy(nhigh - nlow)).is_nonnegative)


def _normalize(text: str, loop_var: str) -> str:
    """Replace whole-word occurrences of ``loop_var`` in ``text`` with the
    canonical placeholder so two bodies differing only in iterator name match."""
    return re.sub(r"\b%s\b" % re.escape(loop_var), ITER_PLACEHOLDER, text)


def _canon_data(name: str, local_scratch: dict[str, None]) -> str:
    """Map a body-local scratch transient to the canonical placeholder; leave
    carried / external names (accumulator, arrays) untouched."""
    return SCRATCH_PLACEHOLDER if name in local_scratch else name


def _node_key(node: nodes.Node, loop_var: str, local_scratch: dict[str, None]) -> NodeKey:
    """A structural key for a body node, iterator- and scratch-name-independent."""
    if isinstance(node, nodes.AccessNode):
        return ("access", _canon_data(node.data, local_scratch))
    return ("tasklet", _normalize(node.code.as_string.strip(), loop_var))


@explicit_cf_compatible
class FuseConsecutiveLoops(ppl.Pass):
    """Fuse two directly-consecutive, identical-bodied, unit-stride loops whose
    iteration ranges are adjacent (``[A, B)`` followed by ``[B, C)``) into a
    single loop over ``[A, C)``."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Fuse every matching consecutive-loop pair in ``sdfg`` and its nested
        SDFGs, repeating until no pair matches (a chain of tiles collapses one
        adjacency per sweep).

        :param sdfg: SDFG to mutate in place.
        :returns: The number of fusions performed, or ``None`` if none.
        """
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                # Rebuilt once per sweep, not per candidate pair: a merge deletes a body
                # state, which is exactly what this index tracks, so it goes stale the
                # moment one fires -- the next sweep iteration rebuilds it fresh.
                scratch_index = self._build_scratch_index(sd)
                for cfg in list(sd.all_control_flow_regions()):
                    if self._fuse_one(cfg, scratch_index):
                        fused += 1
                        changed = True
                        break
        return fused or None

    def _build_scratch_index(self, sd: ControlFlowRegion) -> ScratchIndex:
        """Map every AccessNode data name in ``sd`` to the states referencing it."""
        index: ScratchIndex = {}
        for st in sd.states():
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode):
                    index.setdefault(n.data, OrderedSet()).add(st)
        return index

    def _fuse_one(self, cfg: ControlFlowRegion, scratch_index: ScratchIndex) -> bool:
        """Find and fuse one consecutive-loop pair inside ``cfg``.

        :param cfg: The control-flow region to search (one level; loops nested
                    deeper are reached via ``all_control_flow_regions``).
        :param scratch_index: Current sweep's data-name -> states map (see ``apply_pass``).
        :returns: ``True`` if a pair was fused.
        """
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out_edges = cfg.out_edges(first)
            if len(out_edges) != 1:
                continue
            link = out_edges[0]
            second = link.dst
            if isinstance(second, LoopRegion) and second is not first:
                if len(cfg.in_edges(second)) != 1:
                    continue
                # The connecting edge must be pure sequencing: no assignments and a
                # trivial condition, so nothing runs (or is decided) between them.
                if link.data.assignments:
                    continue
                if link.data.condition is not None and link.data.condition.as_string not in ("1", "True"):
                    continue
                if self._adjacent_identical(first, second, scratch_index):
                    self._merge(cfg, first, second, link)
                    return True
            elif isinstance(second, SDFGState) and self._try_fuse_split_accumulator(cfg, first, second, link, scratch_index):
                return True
        return False

    def _adjacent_identical(self, first: LoopRegion, second: LoopRegion, scratch_index: ScratchIndex) -> bool:
        """Whether ``first`` then ``second`` are unit-stride, structurally identical, cover adjacent index ranges
        ``[A, B)`` and ``[B, C)``, and nothing outside reads the iterator value the fusion changes."""
        for loop in (first, second):
            if not loop.loop_variable:
                return False
            stride = loop_analysis.get_loop_stride(loop)
            if stride is None or symbolic.simplify(stride) != 1:
                return False
            start = loop_analysis.get_init_assignment(loop)
            end = loop_analysis.get_loop_end(loop)
            if start is None or end is None:
                return False
        # Adjacency: first's exclusive end (last value + 1) == second's start.
        facts = SymbolResolver().facts_at(first.parent_graph)
        first_end_excl = symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(first))) + 1)
        second_start = loop_analysis.get_init_assignment(second)
        if not _symbolically_equal(first_end_excl, second_start, facts):
            return False
        if not self._bodies_match(first, second, scratch_index):
            return False
        # A shared name exits with the value it had after ``second``, which the fused loop keeps.
        names = (first, second) if first.loop_variable != second.loop_variable else ()
        return not any(loop_analysis.counter_used_outside_loop(l.loop_variable, l, l.sdfg) for l in names)

    #: Associative+commutative reduction ops eligible for split-accumulator tail folding, mapped to their identity.
    _SPLIT_ACC_IDENTITY: dict[str, tuple[str, float]] = {"Add": ("+", 0.0), "Mul": ("*", 1.0)}

    @staticmethod
    def _assoc_op(node: nodes.Node) -> str | None:
        """Op string (``'+'`` or ``'*'``) if ``node`` is a binary associative+commutative tasklet; else ``None``."""
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
        if len(expr.args) != 2 or not all(arg.is_Symbol for arg in expr.args):
            return None
        info = FuseConsecutiveLoops._SPLIT_ACC_IDENTITY.get(type(expr).__name__)
        return info[0] if info is not None else None

    @staticmethod
    def _scalar_identity_init(state: SDFGState) -> tuple[str, str] | None:
        """If ``state`` only initializes one transient scalar to ``0.0`` (``'+'``) or ``1.0`` (``'*'``),
        return ``(name, op)``; else ``None``."""
        all_nodes = list(state.nodes())
        if len(all_nodes) != 2 or len(state.edges()) != 1:
            return None
        edge = state.edges()[0]
        if not (isinstance(edge.src, nodes.Tasklet) and isinstance(edge.dst, nodes.AccessNode)):
            return None
        desc = state.sdfg.arrays.get(edge.dst.data)
        if desc is None or not desc.transient or desc.shape != (1,):
            return None
        code = edge.src.code.as_string.strip()
        if "=" not in code:
            return None
        lhs, rhs = code.split("=", 1)
        if lhs.strip() != edge.src_conn:
            return None
        try:
            val = float(rhs.strip())
        except ValueError:
            return None
        if val == 0.0:
            return edge.dst.data, "+"
        if val == 1.0:
            return edge.dst.data, "*"
        return None

    @staticmethod
    def _single_carried_scalar_op(state: SDFGState, name: str) -> str | None:
        """Op string if ``state`` carries scalar ``name`` via ``name = name OP x``; else ``None``."""
        acc_nodes = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == name]
        src = [n for n in acc_nodes if state.in_degree(n) == 0]
        snk = [n for n in acc_nodes if state.out_degree(n) == 0]
        if len(acc_nodes) != 2 or len(src) != 1 or len(snk) != 1 or state.out_degree(src[0]) != 1:
            return None
        node = state.out_edges(src[0])[0].dst
        op = FuseConsecutiveLoops._assoc_op(node)
        if op is None:
            return None
        while node is not snk[0]:
            outs = state.out_edges(node)
            if len(outs) != 1:
                return None
            node = outs[0].dst
            if node is not snk[0]:
                if isinstance(node, nodes.AccessNode):
                    desc = state.sdfg.arrays.get(node.data)
                    if desc is None or not desc.transient or state.in_degree(node) != 1:
                        return None
                elif isinstance(node, nodes.Tasklet):
                    if state.in_degree(node) != 1:
                        return None
                    code = node.code.as_string.strip()
                    if "=" not in code or code.split("=", 1)[0].strip() != "__out":
                        return None
                else:
                    return None
        return op

    def _try_fuse_split_accumulator(
        self,
        cfg: ControlFlowRegion,
        first: LoopRegion,
        init_state: SDFGState,
        link1: Edge[InterstateEdge],
        scratch_index: ScratchIndex,
    ) -> bool:
        """Fuse ``first (acc OP= f(i)) -> init_state (tail = id) -> second (tail OP= f(i)) -> post (... acc OP tail)``.

        Hand-unrolled reductions often accumulate the remainder into a fresh ``tail = 0.0`` and
        add ``tail`` into the final sum after the remainder loop. Redirecting ``second`` into
        ``acc`` and dropping the redundant ``+ tail`` exposes the standard adjacent-loop merge.
        """
        if link1.data.condition is not None and link1.data.condition.as_string not in ("1", "True"):
            return False
        if len(cfg.in_edges(init_state)) != 1 or len(cfg.out_edges(init_state)) != 1:
            return False
        link2 = cfg.out_edges(init_state)[0]
        if link2.data.assignments or (
            link2.data.condition is not None and link2.data.condition.as_string not in ("1", "True")
        ):
            return False
        second = link2.dst
        if not isinstance(second, LoopRegion) or second is first or len(cfg.in_edges(second)) != 1:
            return False
        # ``WhileToFor`` leaves the first loop's exit-value assignment to ``i`` on ``link1`` when
        # ``init_state`` sits between the two loops; it is dead when ``second`` immediately
        # re-initializes ``i`` and ``init_state`` never reads ``i``.
        if link1.data.assignments:
            if set(link1.data.assignments) != {second.loop_variable}:
                return False
            if second.loop_variable in init_state.used_symbols(all_symbols=True):
                return False
        out2 = cfg.out_edges(second)
        if len(out2) != 1 or out2[0].data.assignments or not out2[0].data.is_unconditional():
            return False
        post = out2[0].dst
        if not isinstance(post, SDFGState) or len(cfg.in_edges(post)) != 1:
            return False

        init_info = self._scalar_identity_init(init_state)
        if init_info is None:
            return False
        tail, op = init_info

        for loop in (first, second):
            if not loop.loop_variable:
                return False
            stride = loop_analysis.get_loop_stride(loop)
            if stride is None or symbolic.simplify(stride) != 1:
                return False
            if loop_analysis.get_init_assignment(loop) is None or loop_analysis.get_loop_end(loop) is None:
                return False
        facts = SymbolResolver().facts_at(first.parent_graph)
        first_end_excl = symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(first))) + 1)
        second_start = loop_analysis.get_init_assignment(second)
        if not _symbolically_equal(first_end_excl, second_start, facts):
            return False

        s1 = self._single_body_state(first)
        s2 = self._single_body_state(second)
        if s1 is None or s2 is None:
            return False
        if self._single_carried_scalar_op(s2, tail) != op:
            return False
        # ``tail`` must appear ONLY in init_state, s2, and post (never on interstate edges).
        if scratch_index.get(tail) != OrderedSet([init_state, s2, post]):
            return False
        sdfg = first.sdfg
        if any(tail in e.data.free_symbols for e in sdfg.all_interstate_edges(recursive=True)):
            return False

        # Find the matching carried scalar ``acc`` in ``s1`` (same op, same dtype, not ``tail``).
        tail_desc = sdfg.arrays.get(tail)
        acc_candidates = [
            name
            for name in dict.fromkeys(n.data for n in s1.nodes() if isinstance(n, nodes.AccessNode) and n.data != tail)
            if sdfg.arrays.get(name) is not None
            and sdfg.arrays[name].transient
            and sdfg.arrays[name].shape == (1,)
            and sdfg.arrays[name].dtype == tail_desc.dtype
            and self._single_carried_scalar_op(s1, name) == op
        ]
        if len(acc_candidates) != 1:
            return False
        acc = acc_candidates[0]
        if acc in {n.data for n in s2.nodes() if isinstance(n, nodes.AccessNode)}:
            return False
        if any(acc in e.data.free_symbols for e in sdfg.all_interstate_edges(recursive=True)):
            return False
        # Redirecting ``second`` into ``acc`` changes ``acc``'s post-loop value, so ``acc`` must
        # not be read in any state other than ``s1`` and ``post`` (pre-loop init states only write it).
        for st in scratch_index.get(acc, ()):
            if st in (s1, post):
                continue
            if any(isinstance(n, nodes.AccessNode) and n.data == acc and st.out_degree(n) > 0 for n in st.nodes()):
                return False

        # Verify ``s1`` and ``s2`` compute identical bodies once ``tail`` is mapped to ``acc``.
        ids: ValueIds = {}
        sig1 = self._state_signature(s1, first.loop_variable, self._local_scratch(first, s1, scratch_index), ids)
        sig2 = self._state_signature(
            s2, second.loop_variable, self._local_scratch(second, s2, scratch_index), ids, rename={tail: acc}
        )
        if sig1 is None or sig1 != sig2:
            return False

        names = (first, second) if first.loop_variable != second.loop_variable else ()
        if any(loop_analysis.counter_used_outside_loop(l.loop_variable, l, l.sdfg) for l in names):
            return False

        # In ``post``, both ``acc`` and ``tail`` must enter the SAME single connected ``op``-tree
        # as single-consumer read-only leaves, and ``acc`` must not be read anywhere after ``post``
        # without going through that fold.
        tail_nodes = [n for n in post.nodes() if isinstance(n, nodes.AccessNode) and n.data == tail]
        acc_nodes = [n for n in post.nodes() if isinstance(n, nodes.AccessNode) and n.data == acc]
        if len(tail_nodes) != 1 or len(acc_nodes) != 1:
            return False
        tail_leaf, acc_leaf = tail_nodes[0], acc_nodes[0]
        if (
            post.in_degree(tail_leaf) != 0
            or post.out_degree(tail_leaf) != 1
            or post.in_degree(acc_leaf) != 0
            or post.out_degree(acc_leaf) != 1
        ):
            return False
        tail_tasklet = post.out_edges(tail_leaf)[0].dst
        if self._assoc_op(tail_tasklet) != op or post.in_degree(tail_tasklet) != 2 or post.out_degree(tail_tasklet) != 1:
            return False
        used_outside_post = {
            n.data for st in sdfg.states() if st is not post for n in st.nodes() if isinstance(n, nodes.AccessNode)
        }
        root_ops: dict = {}
        for leaf in (acc_leaf, tail_leaf):
            cur: nodes.Node = leaf
            last_op: nodes.Node | None = None
            while True:
                outs = post.out_edges(cur)
                if len(outs) != 1:
                    break
                t = outs[0].dst
                if self._assoc_op(t) != op or post.in_degree(t) != 2 or post.out_degree(t) != 1:
                    break
                nxt = post.out_edges(t)[0].dst
                if not isinstance(nxt, nodes.AccessNode):
                    return False
                last_op = t
                cur = nxt
                desc = sdfg.arrays.get(cur.data)
                if (desc is not None and not desc.transient) or cur.data in used_outside_post:
                    break
            if last_op is None:
                return False
            root_ops[last_op] = None
        if len(root_ops) != 1:
            return False

        # Commit:
        # 1. Rename ``tail`` -> ``acc`` inside ``s2``.
        for n in s2.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == tail:
                n.data = acc
        for e in s2.edges():
            if e.data is not None and e.data.data == tail:
                e.data.data = acc

        # 2. Splice ``tail`` out of ``tail_tasklet`` in ``post`` (``other OP tail -> dst`` becomes ``other -> dst``).
        other_edge = next(e for e in post.in_edges(tail_tasklet) if e.src is not tail_leaf)
        out_edge = post.out_edges(tail_tasklet)[0]
        new_mem = dace.Memlet(
            data=out_edge.data.data,
            subset=out_edge.data.subset,
            other_subset=other_edge.data.subset,
        )
        post.add_edge(other_edge.src, other_edge.src_conn, out_edge.dst, out_edge.dst_conn, new_mem)
        post.remove_node(tail_tasklet)
        post.remove_node(tail_leaf)

        # 3. Drop ``init_state`` and fuse ``first`` + ``second``.
        cfg.remove_node(init_state)
        new_link = cfg.add_edge(first, second, InterstateEdge())
        self._merge(cfg, first, second, new_link)
        return True

    def _single_body_state(self, loop: LoopRegion) -> SDFGState | None:
        """The loop's one non-empty body state, or ``None`` if the body is not a
        single compute state (empty connective states are tolerated, their edges
        must not decide or assign anything: those are not compared)."""
        blocks = list(loop.nodes())
        if not all(isinstance(b, SDFGState) for b in blocks):
            return None
        non_empty = [b for b in blocks if b.nodes()]
        if len(non_empty) != 1:
            return None
        if any(e.data.assignments or not e.data.is_unconditional() for e in loop.edges()):
            return None
        return non_empty[0]

    def _bodies_match(self, first: LoopRegion, second: LoopRegion, scratch_index: ScratchIndex) -> bool:
        """Whether the two loops' single body states compute the same values into the same places,
        up to their iteration variable and body-local scratch names."""
        s1 = self._single_body_state(first)
        s2 = self._single_body_state(second)
        if s1 is None or s2 is None:
            return False
        ids: ValueIds = {}
        sig1 = self._state_signature(s1, first.loop_variable, self._local_scratch(first, s1, scratch_index), ids)
        sig2 = self._state_signature(s2, second.loop_variable, self._local_scratch(second, s2, scratch_index), ids)
        return sig1 is not None and sig1 == sig2

    def _local_scratch(self, loop: LoopRegion, body_state: SDFGState, scratch_index: ScratchIndex) -> dict[str, None]:
        """Transient data names used ONLY inside ``body_state`` -- i.e. not
        referenced by any other block of the owning SDFG (not carried across
        iterations, not read/written outside the loop). These are frontend
        scratch intermediates whose names are irrelevant to what the body
        computes and so are normalised away when comparing two bodies."""
        root = loop
        while root.parent_graph is not None:
            root = root.parent_graph
        local: dict[str, None] = {}
        for n in body_state.nodes():
            if isinstance(n, nodes.AccessNode) and len(scratch_index.get(n.data, ())) == 1:
                desc = root.arrays.get(n.data)
                if desc is not None and desc.transient:
                    local[n.data] = None
        return local

    def _state_signature(
        self,
        state: SDFGState,
        loop_var: str,
        local_scratch: dict[str, None],
        ids: ValueIds,
        rename: dict[str, str] | None = None,
    ) -> list[int] | None:
        """The sorted value keys of ``state``'s sinks, ``None`` if a node's meaning is not in its key:
        scope, nested SDFG and library nodes carry it in properties and contents."""
        rmap = rename or {}

        def _data_name(name: str) -> str:
            return _canon_data(rmap.get(name, name), local_scratch)

        def node_key(node: nodes.Node) -> NodeKey | None:
            if not isinstance(node, (nodes.AccessNode, nodes.Tasklet)):
                return None
            if isinstance(node, nodes.AccessNode) and node.data in local_scratch:
                desc = state.sdfg.arrays[node.data]
                return ("scratch", f"{desc.dtype} {desc.shape}")
            if isinstance(node, nodes.AccessNode):
                return ("access", _data_name(node.data))
            return _node_key(node, loop_var, local_scratch)

        def edge_key(memlet: dace.Memlet) -> tuple:
            subsets = ("" if s is None else _normalize(str(s), loop_var) for s in (memlet.subset, memlet.other_subset))
            return (_data_name(memlet.data) if memlet.data else "", *subsets, str(memlet.wcr))

        keys = value_keys(state, node_key, edge_key, ids, {})
        return None if keys is None else sorted(keys[n] for n in state.sink_nodes())

    def _merge(self, cfg: ControlFlowRegion, first: LoopRegion, second: LoopRegion, link: Edge[InterstateEdge]) -> None:
        """Extend ``first`` over the union range and splice ``second`` out.

        ``first`` keeps its body and iterator; only its exclusive upper bound is
        widened to ``second``'s. ``second``'s successor edges are re-homed onto
        ``first`` and ``second`` is removed.

        :param cfg: The region owning both loops.
        :param first: The surviving loop (its body is kept).
        :param second: The loop to remove (its body is a duplicate of ``first``).
        :param link: The ``first -> second`` sequencing edge (removed with ``second``).
        """
        var = first.loop_variable
        facts = SymbolResolver().facts_at(first.parent_graph)
        start = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(first))
        adjacent = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(second))
        new_end_excl = symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(second))) + 1)
        if not provably_ordered(start, adjacent, facts):
            first.init_statement = dace.properties.CodeBlock(f"{var} = {symbolic.symstr(sympy.Min(start, adjacent))}")
        if not provably_ordered(adjacent, new_end_excl, facts):
            new_end_excl = sympy.Max(adjacent, new_end_excl)
        first.loop_condition = dace.properties.CodeBlock(f"{var} < ({symbolic.symstr(new_end_excl)})")
        # Independent within each range says nothing across them (``A[i + K] = A[i]`` over ``[0, K)`` and ``[K, 2K)``).
        first.known_parallel = False

        out_edges = list(cfg.out_edges(second))
        cfg.remove_node(second)  # also drops the ``first -> second`` link edge
        for e in out_edges:
            cfg.add_edge(first, e.dst, e.data)


__all__ = ["FuseConsecutiveLoops", "GuardedFusionPlan", "plan_guarded_fusion", "commit_guarded_fusion"]

# #
# Guarded fusion: the same adjacent-range rewrite for loops whose bodies DIFFER.
# #
#
# ``FuseConsecutiveLoops`` above fuses a chain only when the bodies are identical, because then
# the merge is free -- widen one bound, drop the twin. When the bodies differ the index sequence
# argument is unchanged (concatenating ``[A, B)`` and ``[B, C)`` under the same stride visits the
# same indices in the same order), but the merged loop has to pick a body per iteration, so the
# bodies move into a ``ConditionalBlock`` keyed on the iterator.
#
# That rewrite is NOT unconditionally desirable, which is why it is a plan/commit pair rather
# than part of the pass above: if one sibling is a recurrence and the other is DOALL, fusing them
# makes the whole range a recurrence and LOSES the parallel half, and nothing downstream splits
# that back apart. Only a caller that has already proved it wins something -- ``WavefrontSkew``,
# which needs the merged 2-D space to find a diagonal -- may commit it.


class GuardedFusionPlan:
    """Sibling loops that fuse into one guarded loop over the union of their ranges.

    ``guards[k]`` are the constraints (each meant ``>= 0``) that hold inside sibling ``k`` --
    exactly the branch conditions :func:`commit_guarded_fusion` will emit, so an analysis run
    against the plan sees the same iteration space as the committed graph.

    ``var`` is the iterator the fused loop uses -- the first sibling's. Siblings normalized
    apart (``normalize_loops_and_maps`` gives every loop a PRIVATE ``_loop_it_N``) keep their own
    names in ``loop_vars`` and are renamed onto ``var`` at commit.
    """

    __slots__ = ("loops", "loop_vars", "var", "lo", "hi", "bounds", "guards")

    def __init__(
        self,
        loops: list[LoopRegion],
        loop_vars: list[str],
        var: str,
        lo: symbolic.SymbolicType,
        hi: symbolic.SymbolicType,
        bounds: list[tuple[symbolic.SymbolicType, symbolic.SymbolicType]],
        guards: list[list[symbolic.SymbolicType]],
    ) -> None:
        self.loops = loops
        self.loop_vars = loop_vars
        self.var = var
        self.lo = lo
        self.hi = hi
        self.bounds = bounds
        self.guards = guards


def plan_guarded_fusion(region: ControlFlowRegion) -> GuardedFusionPlan | None:
    """The maximal chain of adjacent-range sibling loops in ``region``, or ``None``.

    Refuses unless the chain is everything ``region`` holds (bar empty states): the callers
    reason about a two-level nest, and a stray computation beside the chain would be swept into
    a branch it does not belong to. Does not mutate.
    """
    loops = [b for b in region.nodes() if isinstance(b, LoopRegion)]
    if len(loops) < 2:
        return None
    for b in region.nodes():
        if b not in loops and isinstance(b, SDFGState) and len(list(b.nodes())) > 0:
            return None

    # Order the chain by its sequencing edges, starting from the loop nothing else points at.
    heads = [l for l in loops if not any(isinstance(e.src, LoopRegion) for e in region.in_edges(l))]
    if len(heads) != 1:
        return None
    chain: list[LoopRegion] = [heads[0]]
    while True:
        out = region.out_edges(chain[-1])
        if len(out) != 1:
            break
        link, nxt = out[0], out[0].dst
        if not isinstance(nxt, LoopRegion) or nxt in chain:
            break
        if len(region.in_edges(nxt)) != 1:
            break
        # Pure sequencing only: nothing may run, or be decided, between two fused siblings.
        if link.data.assignments:
            return None
        if link.data.condition is not None and link.data.condition.as_string not in ("1", "True"):
            return None
        chain.append(nxt)
    if len(chain) != len(loops):
        return None

    var = chain[0].loop_variable
    loop_vars = [l.loop_variable for l in chain]
    # Renaming a later sibling onto ``var`` must not capture: refuse if ``var`` already means
    # something inside that sibling.
    if any(var in l.free_symbols for l in chain[1:]):
        return None
    bounds: list[tuple[symbolic.SymbolicType, symbolic.SymbolicType]] = []
    for loop in chain:
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None or symbolic.simplify(stride) != 1:
            return None
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        if start is None or end is None:
            return None
        bounds.append((symbolic.pystr_to_symbolic(start), symbolic.pystr_to_symbolic(end)))
    for (_, prev_end), (nxt_start, _) in zip(bounds, bounds[1:]):
        if not _symbolically_equal(symbolic.simplify(prev_end + 1), nxt_start):
            return None
    # The guards partition the union only when no sibling's range is inverted.
    facts = SymbolResolver().facts_at(region)
    if not all(provably_ordered(lo, hi + 1, facts) for lo, hi in bounds):
        return None

    v = symbolic.pystr_to_symbolic(var)
    guards = [[symbolic.simplify(v - lo), symbolic.simplify(hi - v)] for (lo, hi) in bounds]
    return GuardedFusionPlan(chain, loop_vars, var, bounds[0][0], bounds[-1][1], bounds, guards)


def commit_guarded_fusion(plan: GuardedFusionPlan, region: ControlFlowRegion) -> LoopRegion:
    """Replace ``plan``'s chain in ``region`` with one loop over the union range whose body is a
    ``ConditionalBlock`` selecting the original body per iteration. Returns the new loop.

    The branch conditions are the chain's own sub-ranges, tested in order, with the last sibling
    as the ``else`` -- so the partition is total by construction and no iteration falls through.
    """
    var = plan.var
    sdfg = region.sdfg
    # Read before any mutation: if the chain led the region, the fused loop has to inherit that,
    # and ``add_node`` is the only sanctioned way to say so.
    was_start = region.start_block is plan.loops[0]
    merged = LoopRegion(
        f"{plan.loops[0].label}_fused",
        condition_expr=f"{var} < ({symbolic.symstr(plan.hi + 1)})",
        loop_var=var,
        initialize_expr=f"{var} = {symbolic.symstr(plan.lo)}",
        update_expr=f"{var} = {var} + 1",
        sdfg=sdfg,
    )
    selector = ConditionalBlock(f"{var}_range_select", sdfg=sdfg)
    for k, loop in enumerate(plan.loops):
        if plan.loop_vars[k] != var:
            loop.replace_dict({plan.loop_vars[k]: var})
        branch = ControlFlowRegion(f"{loop.label}_range", sdfg=sdfg)
        # Snapshot before moving: ``add_node`` re-homes each block's ``parent_graph``, so the
        # edge list has to be read off the loop while it still owns them.
        start, blocks, edges = loop.start_block, list(loop.nodes()), list(loop.edges())
        for blk in blocks:
            branch.add_node(blk, is_start_block=blk is start)
        for e in edges:
            branch.add_edge(e.src, e.dst, e.data)
        last = k == len(plan.loops) - 1
        selector.add_branch(None if last else f"{var} < ({symbolic.symstr(plan.bounds[k][1] + 1)})", branch)
    merged.add_node(selector, is_start_block=True)

    in_edges = list(region.in_edges(plan.loops[0]))
    out_edges = list(region.out_edges(plan.loops[-1]))
    region.add_node(merged, is_start_block=was_start, ensure_unique_name=True)
    for e in in_edges:
        region.add_edge(e.src, merged, e.data)
    for e in out_edges:
        region.add_edge(merged, e.dst, e.data)
    for loop in plan.loops:
        region.remove_node(loop)
    return merged
