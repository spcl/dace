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

The match is deliberately conservative -- any deviation is a clean no-op -- and
runs BEFORE ``normalize_reduction`` so it sees the raw frontend boundary (which
that stage would otherwise rewrite). Only the polybench orientation (``side='L'``,
``uplo='L'``) is recognised; other orientations fall through untouched.
"""
from typing import Dict, List, Optional, Tuple

from dace import SDFG, SDFGState, memlet as mm, symbolic
from dace.sdfg import nodes
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


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
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in list(sd.states()):
                for node in list(state.nodes()):
                    if isinstance(node, nodes.MapEntry) and node in state.nodes() and self._try_lift(sd, state, node):
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
        nsdfg = next(iter(body))
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
        params = set(me.map.params)
        for i, ax in enumerate(tri):
            b, e, _ = ax
            if _eq(b, e) and str(b) in params:
                p_col = str(b)
                p_row_axis = tri[1 - i]
                # the other axis must be the triangle 0:p_row for some param p_row
                for cand in params - {p_col}:
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
        a = self._find_symmetric(ins, p_row, exclude={c})
        if a is None:
            return None
        # Matrix B: read at [p_row, p_col] and on the column [0:p_row, p_col].
        b = self._find_b(ins, p_row, p_col, exclude={c, a})
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
        nsdfg = next(iter(state.all_nodes_between(me, mx)))
        # One read AccessNode per array feeding the map; the frontend may stage the
        # same array through several duplicate read nodes -- keep one, drop the rest.
        reads = {e.data.data: e.src for e in state.in_edges(me) if isinstance(e.src, nodes.AccessNode)}
        writes = {e.data.data: e.dst for e in state.out_edges(mx) if isinstance(e.dst, nodes.AccessNode)}
        boundary = ({e.src
                     for e in state.in_edges(me) if isinstance(e.src, nodes.AccessNode)}
                    | {e.dst
                       for e in state.out_edges(mx) if isinstance(e.dst, nodes.AccessNode)})

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


__all__ = ["LoopToSymm"]
