# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a hand-written symmetric matrix-multiply nest to a ``Symm`` BLAS node.

polybench ``symm`` computes ``C := alpha*A*B + beta*C`` with ``A`` symmetric,
written by hand as a per-column in-place triangular accumulation rather than the
BLAS ``xSYMM`` primitive::

    for j, i:                              # a 2-D parallel map over (i in 0:M, j in 0:N)
        temp2 = 0
        for k in 0:i:                      # triangular inner reduction
            C[k, j] += alpha * B[i, j] * A[i, k]
            temp2   +=          B[k, j] * A[i, k]
        C[i, j] = beta*C[i, j] + alpha*B[i, j]*A[i, i] + alpha*temp2

The frontend emits this as a 2-D map whose body is a single NestedSDFG; the
in-place scatter surfaces at the NestedSDFG boundary as a triangular slice-WCR
``C[0:i, j]`` onto the SAME output ``C`` that the finalize step also writes
point-wise at ``C[i, j]``. That pairing -- a triangular self-scatter and a
point-write onto one output, fed by a symmetric operand ``A`` referenced only on
its ``[i, 0:i]`` lower triangle plus the ``[i, i]`` diagonal, and a second matrix
``B`` -- is the ``symm`` fingerprint. Recognising it and emitting a :class:`Symm`
node dispatches to the vendor ``dsymm`` / ``cublasDsymm`` kernel and replaces the
sequential in-place triangular accumulation with the optimized primitive.

The same kernel has a second, equally common spelling -- the npbench slice form,
which is what the corpus carries -- where the triangular accumulation is written with
column slices and an explicit inner product instead of a scalar ``k`` loop::

    C *= beta[0]                                       # a separate prologue state
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]  # scatter onto the rows above i
            temp2[j]  = B[:i, j] @ A[i, :i]            # a Dot/MatMul library node
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2

That one is NOT a map with a NestedSDFG body -- it is a two-level ``LoopRegion`` nest
whose statements the frontend spreads over staging temporaries -- so it is matched the
way ``loop_to_syrk`` matches its nest: by RESOLVING the dataflow of each body state to
a sympy expression and comparing that against what ``symm`` is defined to compute (see
:mod:`~dace.transformation.passes.canonicalize.rank_k_match`). The ``beta`` prescale is
a separate statement there and is deliberately left alone: the lift emits ``Symm`` with
a compile-time ``beta=1``, so ``C *= beta`` followed by ``C := alpha*A*B + C`` composes
to the same ``C := alpha*A*B + beta*C``.

Both matches are deliberately conservative -- any deviation is a clean no-op. The
map form must run BEFORE ``normalize_reduction`` so it sees the raw frontend boundary
(which that stage would otherwise rewrite); the slice form needs the body states
already fused, so the pass is scheduled a second time next to ``loop_to_syrk``. Only
the polybench orientation (``side='L'``, ``uplo='L'``) is recognised; other
orientations fall through untouched.
"""
from typing import Dict, List, NamedTuple, Optional, Tuple

import sympy

from dace import SDFG, SDFGState, memlet as mm, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.rank_k_match import (StateValueResolver, equals, expressions_equal,
                                                                  is_single_element, loop_extent, loop_invariant,
                                                                  nontransient_written, outer_loop_candidates, reaches,
                                                                  replace_loop_with_state, root_sdfg_of,
                                                                  single_body_state, sink_write_subset, written_arrays)
from dace.transformation.transformation import explicit_cf_compatible

#: Stand-ins for the two slice indices while a body state's value expression is resolved:
#: the row inside the ``C[0:i, j]`` scatter, and the column inside the ``C[i, 0:N]`` finalize.
SLICE_ROW = sympy.Symbol("__symm_p")
SLICE_COL = sympy.Symbol("__symm_q")

#: The frontend stages a library-node result through at most a copy AccessNode plus a
#: forwarding tasklet; anything longer is not the shape this matcher understands.
MAX_STAGING_HOPS = 4


def _eq(a, b) -> bool:
    """Symbolic equality of two index expressions (strings or sympy)."""
    try:
        return bool(symbolic.simplify(symbolic.pystr_to_symbolic(str(a)) - symbolic.pystr_to_symbolic(str(b))) == 0)
    except Exception:
        return False


def _axes(subset) -> Optional[List[Tuple[object, object, object]]]:
    """The ``(begin, end, step)`` tuple of every axis of a 2-D ``Range``, else None."""
    if not isinstance(subset, Range) or len(subset) != 2:
        return None
    return list(subset.ndrange())


def _is_point(axis, p) -> bool:
    """Axis is the single point ``p`` (``begin == end == p``, unit step)."""
    b, e, s = axis
    return _eq(b, p) and _eq(e, p) and _eq(s, 1)


def _is_lower_tri(axis, p) -> bool:
    """Axis is the half-open triangular range ``0:p`` (``begin 0``, ``end p-1``)."""
    b, e, s = axis
    return _eq(b, 0) and _eq(e, symbolic.pystr_to_symbolic(str(p)) - 1) and _eq(s, 1)


def _is_scalar_point(subset) -> bool:
    """Subset is a single element of a length-1 array (a scalar coefficient read)."""
    if not isinstance(subset, Range):
        return False
    return all(_eq(b, e) for b, e, _ in subset.ndrange())


class SymmMatch:
    """Extracted operands of a recognised ``symm`` nest."""

    def __init__(self, a: str, b: str, c: str, alpha: str, beta: str):
        self.a, self.b, self.c, self.alpha, self.beta = a, b, c, alpha, beta


@explicit_cf_compatible
class LoopToSymm(ppl.Pass):
    """Lift a hand-written symmetric matrix-multiply map nest to a ``Symm`` node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in list(sd.states()):
                for node in list(state.nodes()):
                    if isinstance(node, nodes.MapEntry) and node in state.nodes() and self._try_lift(sd, state, node):
                        count += 1
        for parent, loop in outer_loop_candidates(sdfg):
            if loop not in parent.nodes():
                continue  # already spliced out (defensive)
            match = match_slice_form(parent, loop)
            if match is not None:
                replace_slice_form(parent, loop, match)
                count += 1
        return count or None

    def _try_lift(self, sdfg: SDFG, state: SDFGState, me: nodes.MapEntry) -> bool:
        match = self._match(sdfg, state, me)
        if match is None:
            return False
        self._replace(sdfg, state, me, match)
        return True

    def _match(self, sdfg: SDFG, state: SDFGState, me: nodes.MapEntry) -> Optional[SymmMatch]:
        if len(me.map.params) != 2:
            return None
        mx = state.exit_node(me)
        body = state.all_nodes_between(me, mx)
        if body is None or len(body) != 1:
            return None
        # ``all_nodes_between`` returns a SET; the len==1 guard above makes this pick unambiguous today, but
        # take it by a stable key so relaxing the guard cannot silently make the choice hash-order dependent.
        nsdfg = min(body, key=state.node_id)
        if not isinstance(nsdfg, nodes.NestedSDFG):
            return None

        # Group the NestedSDFG's boundary memlets (in map-parameter terms) by array.
        ins: Dict[str, List] = {}
        for e in state.in_edges(nsdfg):
            if e.data is not None and e.data.data is not None:
                ins.setdefault(e.data.data, []).append(e.data.subset)
        outs: Dict[str, List] = {}
        for e in state.out_edges(nsdfg):
            if e.data is not None and e.data.data is not None:
                outs.setdefault(e.data.data, []).append((e.data.subset, e.data.wcr))

        # Output ``C``: exactly one array, written by a triangular slice-WCR
        # ``C[0:p_row, p_col]`` and a point-write ``C[p_row, p_col]``.
        if len(outs) != 1:
            return None
        c = next(iter(outs))
        tri = point = None
        for subset, wcr in outs[c]:
            ax = _axes(subset)
            if ax is None:
                return None
            if wcr is not None:
                tri = ax
            else:
                point = ax
        if tri is None or point is None:
            return None
        # p_col is the point axis of the slice-WCR; p_row is its triangular axis.
        p_col, p_row = None, None
        params = dict.fromkeys(me.map.params)
        for i, ax in enumerate(tri):
            b, e, _ = ax
            if _eq(b, e) and str(b) in params:
                p_col = str(b)
                p_row_axis = tri[1 - i]
                # the other axis must be the triangle 0:p_row for some param p_row
                for cand in sorted(p for p in params if p != p_col):  # stable order regardless of dict order
                    if _is_lower_tri(p_row_axis, cand):
                        p_row = cand
                break
        if p_row is None or p_col is None or p_row == p_col:
            return None
        # The point-write must be exactly ``C[p_row, p_col]``.
        if not (_is_point(point[0], p_row) and _is_point(point[1], p_col)):
            return None
        # C is also read point-wise at [p_row, p_col].
        if c not in ins or not any(
                _axes(s) and _is_point(_axes(s)[0], p_row) and _is_point(_axes(s)[1], p_col) for s in ins[c]):
            return None

        # Symmetric operand A: read on its lower triangle [p_row, 0:p_row] and its
        # [p_row, p_row] diagonal (and nowhere else).
        a = self._find_symmetric(ins, p_row, exclude=dict.fromkeys([c]))
        if a is None:
            return None
        # Matrix B: read at [p_row, p_col] and on the column [0:p_row, p_col].
        b = self._find_b(ins, p_row, p_col, exclude=dict.fromkeys([c, a]))
        if b is None:
            return None
        # alpha, beta: two distinct scalar (length-1) inputs.
        scalars = [
            name for name, subs in ins.items() if name not in (a, b, c) and all(_is_scalar_point(s) for s in subs)
        ]
        if len(scalars) != 2:
            return None
        alpha, beta = self._order_coeffs(sdfg, nsdfg, scalars)
        if alpha is None:
            return None
        return SymmMatch(a, b, c, alpha, beta)

    def _find_symmetric(self, ins: Dict[str, List], p_row: str, exclude) -> Optional[str]:
        for name, subs in ins.items():
            if name in exclude:
                continue
            has_diag = any(_axes(s) and _is_point(_axes(s)[0], p_row) and _is_point(_axes(s)[1], p_row) for s in subs)
            has_tri = any(
                _axes(s) and _is_point(_axes(s)[0], p_row) and _is_lower_tri(_axes(s)[1], p_row) for s in subs)
            if has_diag and has_tri:
                return name
        return None

    def _find_b(self, ins: Dict[str, List], p_row: str, p_col: str, exclude) -> Optional[str]:
        for name, subs in ins.items():
            if name in exclude:
                continue
            has_pt = any(_axes(s) and _is_point(_axes(s)[0], p_row) and _is_point(_axes(s)[1], p_col) for s in subs)
            has_col = any(
                _axes(s) and _is_lower_tri(_axes(s)[0], p_row) and _is_point(_axes(s)[1], p_col) for s in subs)
            if has_pt and has_col:
                return name
        return None

    def _order_coeffs(self, sdfg: SDFG, nsdfg: nodes.NestedSDFG, scalars: List[str]) -> Tuple[Optional[str], str]:
        """``(alpha, beta)``: alpha scales the product (it reaches the inner reduction
        map), beta scales the prior C (it reaches only the finalize step). Distinguish
        them by which crosses into an inner Map scope in the NestedSDFG body."""
        inner = nsdfg.sdfg
        conn_of = {e.data.data: e.dst_conn for e in _boundary_in(sdfg, nsdfg)}
        alpha = None
        for name in scalars:
            conn = conn_of.get(name)
            if conn is not None and _reaches_map_scope(inner, conn):
                alpha = name
        if alpha is None:
            return None, scalars[0]
        beta = next(n for n in scalars if n != alpha)
        return alpha, beta

    def _replace(self, sdfg: SDFG, state: SDFGState, me: nodes.MapEntry, match: SymmMatch) -> None:
        from dace.libraries.blas.nodes.symm import Symm
        mx = state.exit_node(me)
        nsdfg = min(state.all_nodes_between(me, mx), key=state.node_id)  # set -> stable pick (see _match)
        # One read AccessNode per array feeding the map; the frontend may stage the
        # same array through several duplicate read nodes -- keep one, drop the rest.
        reads = {e.data.data: e.src for e in state.in_edges(me) if isinstance(e.src, nodes.AccessNode)}
        writes = {e.data.data: e.dst for e in state.out_edges(mx) if isinstance(e.dst, nodes.AccessNode)}
        boundary = dict.fromkeys([
            *(e.src for e in state.in_edges(me) if isinstance(e.src, nodes.AccessNode)),
            *(e.dst for e in state.out_edges(mx) if isinstance(e.dst, nodes.AccessNode)),
        ])

        node = Symm(me.map.label + "_symm", side="L", uplo="L", alpha=1, beta=1, alpha_input=True, beta_input=True)
        state.add_node(node)

        def full(name: str) -> mm.Memlet:
            return mm.Memlet(data=name, subset=Range([(0, s - 1, 1) for s in sdfg.arrays[name].shape]))

        state.add_edge(reads[match.a], None, node, "_a", full(match.a))
        state.add_edge(reads[match.b], None, node, "_b", full(match.b))
        state.add_edge(reads[match.c], None, node, "_c", full(match.c))
        state.add_edge(reads[match.alpha], None, node, "_alpha", mm.Memlet(f"{match.alpha}[0]"))
        state.add_edge(reads[match.beta], None, node, "_beta", mm.Memlet(f"{match.beta}[0]"))
        state.add_edge(node, "_c", writes[match.c], None, full(match.c))

        state.remove_node(nsdfg)
        state.remove_node(me)
        state.remove_node(mx)
        # Drop any boundary read/write node the map alone kept alive.
        for an in boundary:
            if an in state.nodes() and state.degree(an) == 0:
                state.remove_node(an)


def _boundary_in(sdfg: SDFG, nsdfg: nodes.NestedSDFG):
    for st in sdfg.states():
        if nsdfg in st.nodes():
            return st.in_edges(nsdfg)
    return []


def _reaches_map_scope(inner: SDFG, conn: str) -> bool:
    """Whether the NestedSDFG input ``conn`` is read inside a Map scope of ``inner``
    (alpha multiplies the per-``k`` product; beta only the outer finalize). Follows
    the connector's data node directly, or through one passthrough AccessNode, to a
    MapEntry."""
    for st in inner.all_states():
        for dn in st.data_nodes():
            if dn.data != conn:
                continue
            for e in st.out_edges(dn):
                if isinstance(e.dst, nodes.MapEntry):
                    return True
                if isinstance(e.dst, nodes.AccessNode) and any(
                        isinstance(e2.dst, nodes.MapEntry) for e2 in st.out_edges(e.dst)):
                    return True
    return False


class SymmSliceMatch(NamedTuple):
    """Operands of a recognised npbench-slice ``symm`` nest."""
    c: str
    a: str
    b: str
    alpha: str
    temp: str


def col_slice(subset, rows, col) -> bool:
    """``subset`` is the column prefix ``[0:rows, col]``."""
    if subset is None or len(subset) != 2:
        return False
    (rb, re_, rs), (cb, ce, cs) = subset.ndrange()
    return (equals(rb, 0) and equals(re_,
                                     symbolic.pystr_to_symbolic(str(rows)) - 1) and equals(rs, 1) and equals(cb, col)
            and equals(ce, col) and equals(cs, 1))


def row_slice(subset, row, cols) -> bool:
    """``subset`` is the row prefix ``[row, 0:cols]``."""
    if subset is None or len(subset) != 2:
        return False
    (rb, re_, rs), (cb, ce, cs) = subset.ndrange()
    return (equals(rb, row) and equals(re_, row) and equals(rs, 1) and equals(cb, 0)
            and equals(ce,
                       symbolic.pystr_to_symbolic(str(cols)) - 1) and equals(cs, 1))


def point_of(subset, index) -> bool:
    """``subset`` is the single 1-D element ``[index]``."""
    if subset is None or len(subset) != 1:
        return False
    b, e, s = next(iter(subset.ndrange()))
    return equals(b, index) and equals(e, index) and equals(s, 1)


def write_node(state: SDFGState, name: str) -> Optional[nodes.AccessNode]:
    """The state's single AccessNode for ``name`` that has a producer."""
    found = [n for n in state.data_nodes() if n.data == name and state.in_degree(n) > 0]
    return found[0] if len(found) == 1 else None


def inner_loop_then_state(outer: LoopRegion) -> Optional[Tuple[LoopRegion, SDFGState]]:
    """Split ``outer``'s body into its one inner LoopRegion and its one non-empty
    trailing state, requiring the loop to run FIRST -- the trailing statement consumes
    the vector the loop accumulated, so the other order is a different program."""
    inner: Optional[LoopRegion] = None
    tail: Optional[SDFGState] = None
    for block in outer.nodes():
        if isinstance(block, LoopRegion):
            if inner is not None:
                return None
            inner = block
        elif isinstance(block, SDFGState):
            if not block.nodes():
                continue  # empty connective state
            if tail is not None:
                return None
            tail = block
        else:
            return None
    if inner is None or tail is None or not reaches(outer, inner, tail):
        return None
    return inner, tail


def staged_library_node(state: SDFGState, node: nodes.AccessNode) -> Optional[nodes.LibraryNode]:
    """Walk back from ``node`` through the frontend's forwarding copies to the library
    node that produced its value, or ``None`` if anything else intervenes."""
    current = node
    for _ in range(MAX_STAGING_HOPS):
        producers = [e for e in state.in_edges(current) if e.data is not None and not e.data.is_empty()]
        if len(producers) != 1:
            return None
        src = producers[0].src
        if isinstance(src, nodes.LibraryNode):
            return src
        if isinstance(src, nodes.Tasklet):
            ins = [e for e in state.in_edges(src) if e.data is not None and not e.data.is_empty()]
            if len(ins) != 1 or len(src.out_connectors) != 1:
                return None
            if (src.code.as_string or "").strip() != f"{next(iter(src.out_connectors))} = {ins[0].dst_conn}":
                return None
            src = ins[0].src
        if not isinstance(src, nodes.AccessNode):
            return None
        current = src
    return None


def library_operand_reads(state: SDFGState, sdfg: SDFG,
                          node: nodes.LibraryNode) -> Optional[List[Tuple[str, subsets.Subset]]]:
    """The ``(array, subset)`` each operand of ``node`` ultimately reads, resolving the
    frontend's per-operand staging copy back to the array it was filled from."""
    out: List[Tuple[str, subsets.Subset]] = []
    for edge in state.in_edges(node):
        if edge.data is None or edge.data.is_empty() or not isinstance(edge.src, nodes.AccessNode):
            return None
        desc = sdfg.arrays.get(edge.src.data)
        if desc is not None and desc.transient:
            stage = [e for e in state.in_edges(edge.src) if e.data is not None and not e.data.is_empty()]
            if len(stage) != 1 or not isinstance(stage[0].src, nodes.AccessNode):
                return None
            out.append((stage[0].data.data, stage[0].data.subset))
        else:
            out.append((edge.src.data, edge.data.subset))
    return out


def match_finalize_state(state: SDFGState, root: SDFG, i: str, n, c: str) -> Optional[Tuple[str, str, str, str]]:
    """Match ``C[i, 0:N] += alpha[0]*B[i, 0:N]*A[i, i] + alpha[0]*t[0:N]``.

    :returns: ``(a, b, alpha, t)`` -- the symmetric operand, the second matrix, the
              coefficient and the scratch vector -- or ``None``.
    """
    if nontransient_written(state, root) != dict.fromkeys([c]):
        return None
    cw = write_node(state, c)
    if cw is None or not row_slice(sink_write_subset(state, cw), i, n):
        return None
    resolver = StateValueResolver(state)
    i_sym = symbolic.pystr_to_symbolic(i)
    try:
        value = resolver.value_at(cw, [i_sym, SLICE_COL])
    except ValueError:
        return None
    c_leaf = alpha_sym = a_sym = b_sym = t_sym = None
    alpha = a = b = t = None
    for sym, read in resolver.leaves.items():
        desc = root.arrays.get(read.array)
        if desc is None:
            return None
        if is_single_element(desc):
            if alpha is not None or len(read.index) != 1 or not equals(read.index[0], 0):
                return None
            alpha, alpha_sym = read.array, sym
        elif len(read.index) == 1:
            if t is not None or not desc.transient or not equals(read.index[0], SLICE_COL):
                return None
            t, t_sym = read.array, sym
        elif len(read.index) == 2:
            first, second = read.index
            if read.array == c:
                if c_leaf is not None or not (equals(first, i_sym) and equals(second, SLICE_COL)):
                    return None
                c_leaf = sym
            elif not equals(first, i_sym):
                return None
            elif equals(second, i_sym):
                if a is not None:
                    return None
                a, a_sym = read.array, sym
            elif equals(second, SLICE_COL):
                if b is not None:
                    return None
                b, b_sym = read.array, sym
            else:
                return None
        else:
            return None
    if c_leaf is None or alpha is None or a is None or b is None or t is None or a == b:
        return None
    if not expressions_equal(value, c_leaf + alpha_sym * b_sym * a_sym + alpha_sym * t_sym):
        return None
    return a, b, alpha, t


def match_scatter_state(state: SDFGState, root: SDFG, i: str, j: str, match: SymmSliceMatch) -> bool:
    """Whether ``state`` is ``C[0:i, j] += alpha[0]*B[i, j]*A[i, 0:i]`` together with
    ``t[j] = B[0:i, j] @ A[i, 0:i]``, over the operands ``match`` already fixed."""
    if nontransient_written(state, root) != dict.fromkeys([match.c]):
        return False
    cw = write_node(state, match.c)
    if cw is None or not col_slice(sink_write_subset(state, cw), i, j):
        return False
    resolver = StateValueResolver(state)
    i_sym, j_sym = symbolic.pystr_to_symbolic(i), symbolic.pystr_to_symbolic(j)
    try:
        value = resolver.value_at(cw, [SLICE_ROW, j_sym])
    except ValueError:
        return False
    c_leaf = alpha_sym = a_sym = b_sym = None
    for sym, read in resolver.leaves.items():
        desc = root.arrays.get(read.array)
        if desc is None or len(read.index) != (1 if is_single_element(desc) else 2):
            return False
        if read.array == match.alpha:
            if alpha_sym is not None or not equals(read.index[0], 0):
                return False
            alpha_sym = sym
        elif read.array == match.c:
            if c_leaf is not None or not (equals(read.index[0], SLICE_ROW) and equals(read.index[1], j_sym)):
                return False
            c_leaf = sym
        elif read.array == match.a:
            if a_sym is not None or not (equals(read.index[0], i_sym) and equals(read.index[1], SLICE_ROW)):
                return False
            a_sym = sym
        elif read.array == match.b:
            if b_sym is not None or not (equals(read.index[0], i_sym) and equals(read.index[1], j_sym)):
                return False
            b_sym = sym
        else:
            return False
    if c_leaf is None or alpha_sym is None or a_sym is None or b_sym is None:
        return False
    if not expressions_equal(value, c_leaf + alpha_sym * a_sym * b_sym):
        return False
    return matches_inner_product(state, root, i, j, match)


def matches_inner_product(state: SDFGState, root: SDFG, i: str, j: str, match: SymmSliceMatch) -> bool:
    """Whether the state also stores ``t[j] = B[0:i, j] @ A[i, 0:i]`` -- the second half
    of the scatter statement, which the frontend emits as a Dot/MatMul library node
    rather than as resolvable dataflow."""
    from dace.libraries.blas.nodes.dot import Dot  # local: the BLAS package imports transformations
    from dace.libraries.blas.nodes.matmul import MatMul
    tw = write_node(state, match.temp)
    if tw is None or not point_of(sink_write_subset(state, tw), symbolic.pystr_to_symbolic(j)):
        return False
    lib = staged_library_node(state, tw)
    if not isinstance(lib, (Dot, MatMul)):
        return False
    operands = library_operand_reads(state, root, lib)
    if operands is None or len(operands) != 2:
        return False
    column = [s for name, s in operands if name == match.b and col_slice(s, i, j)]
    row = [s for name, s in operands if name == match.a and row_slice(s, i, i)]
    return len(column) == 1 and len(row) == 1


def transient_dead_outside(loop: LoopRegion, root: SDFG, name: str) -> bool:
    """Whether ``name`` is a transient no state outside ``loop`` READS.

    The lift splices the nest away, so every value it produced into ``name`` is gone.
    The corpus zero-initialises the scratch vector before the nest, which is a write and
    therefore harmless; a read afterwards would mean the value outlives the nest.
    """
    desc = root.arrays.get(name)
    if desc is None or not desc.transient:
        return False
    inside = dict.fromkeys(id(state) for state in loop.all_states())
    for state in root.all_states():
        if id(state) in inside:
            continue
        if any(dn.data == name and state.out_degree(dn) > 0 for dn in state.data_nodes()):
            return False
    return True


def shape_is(root: SDFG, name: str, rows, cols) -> bool:
    """Whether ``name`` is the 2-D array ``rows x cols``."""
    desc = root.arrays.get(name)
    if desc is None or len(desc.shape) != 2:
        return False
    return equals(desc.shape[0], rows) and equals(desc.shape[1], cols)


def match_slice_form(parent: ControlFlowRegion, loop: LoopRegion) -> Optional[SymmSliceMatch]:
    """Recognise the npbench slice spelling of ``symm`` rooted at ``loop``."""
    root = root_sdfg_of(parent)
    m = loop_extent(loop)
    if m is None:
        return None
    split = inner_loop_then_state(loop)
    if split is None:
        return None
    inner, finalize = split
    n = loop_extent(inner)
    if n is None:
        return None
    scatter = single_body_state(inner)
    if scatter is None:
        return None
    outputs = list(nontransient_written(finalize, root))
    if len(outputs) != 1:
        return None
    c = outputs[0]
    operands = match_finalize_state(finalize, root, loop.loop_variable, n, c)
    if operands is None:
        return None
    a, b, alpha, temp = operands
    match = SymmSliceMatch(c=c, a=a, b=b, alpha=alpha, temp=temp)
    if not match_scatter_state(scatter, root, loop.loop_variable, inner.loop_variable, match):
        return None
    if not (shape_is(root, c, m, n) and shape_is(root, a, m, m) and shape_is(root, b, m, n)):
        return None
    if not loop_invariant(loop, (a, b, alpha)):
        return None
    written = dict.fromkeys(name for state in loop.all_states() for name in written_arrays(state))
    written.pop(c, None)
    if any(not transient_dead_outside(loop, root, name) for name in written):
        return None
    return match


def replace_slice_form(parent: ControlFlowRegion, loop: LoopRegion, match: SymmSliceMatch) -> None:
    """Splice the recognised nest out and emit the ``Symm`` node in its place.

    ``beta`` stays a compile-time ``1``: the ``C *= beta[0]`` prescale is a statement
    OUTSIDE this nest and is left where it is, so the two compose to the BLAS
    ``C := alpha*A*B + beta*C``.
    """
    from dace.libraries.blas.nodes.symm import Symm  # local: the BLAS package imports transformations
    root = root_sdfg_of(parent)
    state = replace_loop_with_state(parent, loop, loop.label + "_symm")
    node = Symm(loop.label + "_symm", side="L", uplo="L", alpha=1, beta=1, alpha_input=True, beta_input=False)
    state.add_node(node)

    def full(name: str) -> mm.Memlet:
        # Fresh Range per edge -- DaCe forbids two memlets sharing one subset.
        return mm.Memlet(data=name, subset=Range([(0, s - 1, 1) for s in root.arrays[name].shape]))

    state.add_edge(state.add_read(match.a), None, node, "_a", full(match.a))
    state.add_edge(state.add_read(match.b), None, node, "_b", full(match.b))
    state.add_edge(state.add_read(match.c), None, node, "_c", full(match.c))
    state.add_edge(state.add_read(match.alpha), None, node, "_alpha", mm.Memlet(f"{match.alpha}[0]"))
    state.add_edge(node, "_c", state.add_write(match.c), None, full(match.c))


__all__ = ["LoopToSymm"]
