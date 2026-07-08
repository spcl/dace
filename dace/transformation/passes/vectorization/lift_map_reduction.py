# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a scalar reduction carried across an innermost map to a ``Reduce`` libnode.

The spmv row reduction ``for idx: acc = acc + data[idx] * x[indices[idx]]`` is an
innermost map whose body (an indirect-access NestedSDFG that cannot be inlined)
reads scalar ``acc[0]`` at map entry and writes it at map exit -- a loop-carried
RMW. Legacy 1-D ``VectorizeCPU`` mis-vectorizes this once the reduced trip exceeds
the vector width (partial sums never folded), so express it as the canonical
product-map + ``Reduce`` the vectorizer lowers correctly.

Feed-identity lift, WITHOUT touching the opaque body:

1. Accumulator is pre-seeded to the op identity; feed that identity (broadcast,
   not carried) into the map, so the body computes ``identity (op) expr == expr``
   -- just the per-iteration product, gather included.
2. Per-iteration result -> fresh 1-D buffer ``acc_buf[idx-lb]`` instead of the scalar.
3. A ``Reduce`` libnode folds ``acc_buf`` into the accumulator
   (``implementation="vectorized"`` -> :class:`ExpandReduceVectorized`, a
   self-contained ``horizontal_reduce_<op>`` kernel with a scalar tail).

The product-fill map is then an ordinary gather + product map the vectorizer
strides (scalar remainder keeps the gather tail in-range); ``Reduce`` carries its
own vectorized fold + scalar tail.

Detection: :func:`recognize_map_reduction`. Only accumulators pre-initialised to
the op identity are lifted, so seeding the fold with the identity reproduces the
original ``init (op) fold``.
"""
import ast
import copy
from typing import Optional

import dace
from dace import dtypes, nodes, symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.reductions import (
    IDENTITY,
    MapReductionInfo,
    recognize_map_reduction,
)

#: Reduction-op token for each ``add_reduce``-friendly ``ReductionType``. Mirrors
#: the ``+`` / ``*`` restriction of :data:`_WCR_LAMBDA` (see its docstring).
_REDTYPE_OP = {
    dtypes.ReductionType.Sum: "+",
    dtypes.ReductionType.Product: "*",
}

#: Reduction-op token -> ``Reduce`` libnode WCR lambda. Restricted to ``+`` / ``*``:
#: the only ops this lift is tested for, with finite identities (0 / 1) that
#: ``add_reduce`` lowers cleanly. ``max`` / ``min`` (identities -inf / +inf) and
#: bitwise ops are excluded -- their identities fail the finite-float gate below,
#: so they are rejected here rather than mis-lifted with a non-finite identity.
_WCR_LAMBDA = {
    "+": "lambda a, b: a + b",
    "*": "lambda a, b: a * b",
}


def _const_assign_value(code: str) -> Optional[float]:
    """Numeric value of a ``_out = <number>`` tasklet, or ``None``.

    Accepts a bare constant or a unary ``+``/``-`` on a constant (the frontend
    emits ``__out = 0.0`` for a zero seed).

    :param code: The tasklet's Python code string.
    :returns: The float value, or ``None`` if not a numeric constant assign.
    """
    try:
        tree = ast.parse((code or "").strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    v = tree.body[0].value
    if isinstance(v, ast.UnaryOp) and isinstance(v.op, (ast.UAdd, ast.USub)) and isinstance(v.operand, ast.Constant):
        inner = v.operand.value
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -float(inner) if isinstance(v.op, ast.USub) else float(inner)
        return None
    if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and not isinstance(v.value, bool):
        return float(v.value)
    return None


class PureWCRReductionInfo:
    """Pure-WCR boundary reduction from :func:`_recognize_pure_wcr_reduction`.

    The ``body -> map_exit`` ``write_edge`` carries the scalar accumulator under a
    ``CR:op`` WCR; accumulator is NOT read at map entry (WCR alone folds).
    """

    __slots__ = ("map_entry", "map_exit", "body", "accumulator", "op", "write_edge")

    def __init__(self, map_entry, map_exit, body, accumulator, op, write_edge):
        self.map_entry = map_entry
        self.map_exit = map_exit
        self.body = body
        self.accumulator = accumulator
        self.op = op
        self.write_edge = write_edge


def _recognize_pure_wcr_reduction(state: "dace.SDFGState",
                                  map_entry: "dace.nodes.MapEntry") -> Optional[PureWCRReductionInfo]:
    """Recognise ``acc (op)= f(...)`` as a MapExit WCR with no carry-in.

    Canonical ``acc = sum(A)`` / ``dot += a[i]*b[i]``: single-param, unit-step,
    top-level map whose body (any acyclic sub-DAG; the product may fan several
    tasklets into a private ``_wcr_priv_*`` node) has exactly one scalar
    ``body -> map_exit`` edge writing the accumulator under ``CR:+`` / ``CR:*``,
    where the accumulator is a FIXED scalar slot (subset independent of the map
    param) NOT read at map entry. (Loop-carried RMW is :func:`recognize_map_reduction`.)

    Guards reject look-alikes emitted as the same scalar ``CR:`` MapExit edge:
    indexed scatter / recurrence ``a[i] (op)= ...`` (subset depends on param),
    non-additive fold (only ``+`` / ``*`` round-trip the finite-float identity),
    and nested reduction (trip depends on enclosing param -> lifted buffer's
    symbolic shape out of scope after re-nest; deferred).

    :returns: A :class:`PureWCRReductionInfo`, or ``None``.
    """
    from dace.frontend.operations import detect_reduction_type
    if not isinstance(map_entry, dace.nodes.MapEntry):
        return None
    if len(map_entry.map.params) != 1:
        return None
    _, _, step = map_entry.map.range[-1]
    if (step != 1) and (str(step) != "1"):
        return None
    # Top-level only: a nested reduction's trip is an enclosing map param, so the
    # product buffer's symbolic shape would not be in scope after the re-nest.
    if state.entry_node(map_entry) is not None:
        return None
    map_exit = state.exit_node(map_entry)
    inner = state.all_nodes_between(map_entry, map_exit) or set()
    if any(isinstance(n, dace.nodes.MapEntry) for n in inner):
        return None
    param = map_entry.map.params[0]

    def _scalar_slot(e) -> bool:
        return (e.data is not None and e.data.data is not None and e.data.subset is not None
                and e.data.subset.num_elements() == 1)

    # Exactly one scalar WCR write into the map exit, from any body node (bare
    # ``acc = sum(A)`` flat tasklet OR canonicalised product map via a
    # ``_wcr_priv_*`` node -- both accepted).
    wcr_writes = [e for e in state.in_edges(map_exit) if _scalar_slot(e) and e.data.wcr is not None]
    if len(wcr_writes) != 1:
        return None
    write_edge = wcr_writes[0]
    body = write_edge.src
    acc = write_edge.data.data
    # FIXED scalar accumulator: the write subset must not depend on the map param
    # (else it is an indexed scatter / recurrence, not a scalar fold).
    if param in {str(s) for s in write_edge.data.subset.free_symbols}:
        return None
    desc = state.sdfg.arrays.get(acc)
    if desc is None or not isinstance(desc, (dace.data.Scalar, dace.data.Array)):
        return None
    op = _REDTYPE_OP.get(detect_reduction_type(write_edge.data.wcr))
    if op is None or op not in IDENTITY:
        return None
    # Pure WCR: accumulator NOT read at map entry (else loop-carried RMW) and
    # absent elsewhere in the map scope (no aliasing to silently break).
    if any(e.data is not None and e.data.data == acc for e in state.out_edges(map_entry)):
        return None
    for n in (x for x in inner if x not in (map_entry, map_exit)):
        if isinstance(n, dace.nodes.AccessNode) and n.data == acc:
            return None
        if any(e is not write_edge and e.data is not None and e.data.data == acc for e in state.all_edges(n)):
            return None
    return PureWCRReductionInfo(map_entry, map_exit, body, acc, op, write_edge)


class LiftMapReductionToReduce(ppl.Pass):
    """Lift map-carried scalar reductions to product-map + ``Reduce`` libnode.

    :param vectorized: stamp ``implementation="vectorized"`` on the emitted
        ``Reduce`` so :class:`ExpandReduceVectorized` lowers it (the
        self-contained, no-tile-node CPU vectorized fold). Default ``True``.
    """

    def __init__(self, vectorized: bool = True, pure_wcr_only: bool = False, rmw_only: bool = False):
        super().__init__()
        self._vectorized = vectorized
        #: Lift ONLY pure-WCR boundary reductions (skip RMW recogniser). For the
        #: early pipeline call before ``WCRToAugAssign`` rewrites the WCR away; the
        #: RMW shape is lifted later, after ``LoopToMap`` produces the map.
        self._pure_wcr_only = pure_wcr_only
        #: Lift ONLY the loop-carried RMW; leave the pure-WCR ``acc = sum(A)`` as a
        #: map-exit WCR for codegen to lower directly (CPU ``reduction(op:var)`` /
        #: GPU thread-block reduce + one atomic per block). The multi-dim tile
        #: vectorizer sets this so it never materialises a scalar reduction to a
        #: product buffer; opt-in :class:`LiftWCRMapToBufferAndReduce` still gives
        #: the buffer + ``Reduce`` form. Mutually exclusive with ``pure_wcr_only``.
        self._rmw_only = rmw_only

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Lift every recognised map-carried reduction in ``sdfg`` (recursively).

        :param sdfg: The SDFG to transform in place.
        :returns: The number of reductions lifted, or ``None`` if none.
        """
        targets = [(n, g) for n, g in sdfg.all_nodes_recursive()
                   if isinstance(n, nodes.MapEntry) and isinstance(g, dace.SDFGState)]
        count = 0
        for me, state in targets:
            if me not in state.nodes():
                continue  # removed by an earlier lift in this sweep
            # Pure-WCR boundary reduction (``acc(CR:op)`` at MapExit, no carry-in):
            # canonical ``acc = sum(A)``. Tried before the RMW recogniser (which
            # needs a map-entry carry-in read).
            pure = None if self._rmw_only else _recognize_pure_wcr_reduction(state, me)
            if pure is not None and self._lift_pure_wcr(state, pure):
                count += 1
                continue
            if self._pure_wcr_only:
                continue
            info = recognize_map_reduction(state, me)
            if info is None:
                continue
            if self._lift(state, info):
                count += 1
        return count or None

    def _lift_pure_wcr(self, state: dace.SDFGState, info: "PureWCRReductionInfo") -> bool:
        """Lift a pure-WCR boundary reduction to a product buffer + Reduce.

        Redirect the ``CR:op`` map-exit write to a fresh 1-D buffer (drop the WCR
        -> ordinary per-iteration store the tiler strides), then fold the buffer
        into the accumulator via a ``Reduce`` libnode. The ``Reduce -> acc`` edge
        keeps the ``CR:op`` WCR so ``acc (op)= fold`` survives for any initial acc.

        :returns: ``True`` if lifted, ``False`` on failed precondition (SDFG
            unchanged).
        """
        sdfg = state.sdfg
        me, mx, acc, op, write_edge = (info.map_entry, info.map_exit, info.accumulator, info.op, info.write_edge)
        param = me.map.params[0]
        lb, ub, _ = me.map.range[-1]
        trip = symbolic.simplify(ub - lb + 1)
        dtype = sdfg.arrays[acc].dtype
        wcr = _WCR_LAMBDA.get(op)
        if wcr is None:
            return False
        try:
            identity_val = float(IDENTITY[op])
        except (TypeError, ValueError, KeyError):
            return False

        # Locate the map_exit -> acc sink edge that the WCR write drains into.
        write_out_conn = "OUT_" + write_edge.dst_conn[len("IN_"):]
        mx_out = [e for e in state.out_edges(mx) if e.src_conn == write_out_conn]
        if len(mx_out) != 1:
            return False
        mx_out_edge = mx_out[0]
        acc_node = mx_out_edge.dst
        if not (isinstance(acc_node, nodes.AccessNode) and acc_node.data == acc):
            return False

        # --- all preconditions hold; mutate from here on ---
        buf, _ = sdfg.add_transient(f"_red_buf_{acc}", (trip, ), dtype, find_new_name=True)
        # Per-iteration result -> product buffer (drop the WCR carry).
        write_edge.data = dace.Memlet(f"{buf}[{param} - ({lb})]")
        buf_node = state.add_access(buf)
        state.remove_edge(mx_out_edge)
        state.add_edge(mx, write_out_conn, buf_node, None, dace.Memlet(f"{buf}[0:{trip}]"))

        # Reduce(buf) -> acc, vectorized, WCR-accumulated into the prior acc.
        red = state.add_reduce(wcr, axes=[0], identity=identity_val)
        if self._vectorized:
            red.implementation = "vectorized"
        state.add_edge(buf_node, None, red, None, dace.Memlet(f"{buf}[0:{trip}]"))
        out_mem = dace.Memlet(f"{acc}[0]")
        out_mem.wcr = wcr
        state.add_edge(red, None, acc_node, None, out_mem)
        return True

    @staticmethod
    def _split_inout_connector(state: dace.SDFGState, info: MapReductionInfo):
        """Give the accumulator distinct in/out connectors on the body NSDFG.

        When the body reads+writes the accumulator through one *inout* connector
        (s4115 ``sum_val``: same name in in/out_connectors), the feed-identity lift
        would push two outer arrays (identity in, buffer out) through one connector
        -- invalid. Split: rename inner write-side access nodes to a fresh array on
        a new output connector; leave the read on the original.

        :param state: State holding the body node.
        :param info: Recognised reduction; its ``write_edge`` is replaced on split.
        :returns: (possibly new) ``body -> map_exit`` write edge, or ``None`` if the
            inout connector could not be split safely.
        """
        body = info.body
        read_conn, write_conn = info.read_edge.dst_conn, info.write_edge.src_conn
        if not isinstance(body, nodes.NestedSDFG) or read_conn != write_conn:
            return info.write_edge  # already distinct (spmv) or a flat tasklet body
        conn = write_conn
        inner = body.sdfg
        idesc = inner.arrays[conn]
        new_inner, _ = inner.add_array(f"{conn}_acc_w",
                                       idesc.shape,
                                       idesc.dtype,
                                       storage=idesc.storage,
                                       transient=False,
                                       find_new_name=True)
        renamed = False
        for st in inner.all_states():
            for an in list(st.data_nodes()):
                if an.data == conn and st.in_degree(an) >= 1 and st.out_degree(an) == 0:
                    an.data = new_inner
                    for e in st.in_edges(an):
                        if e.data is not None and e.data.data == conn:
                            e.data.data = new_inner
                    renamed = True
        if not renamed:
            del inner.arrays[new_inner]
            return None
        body.add_out_connector(new_inner)
        we = info.write_edge
        new_we = state.add_edge(body, new_inner, we.dst, we.dst_conn, copy.deepcopy(we.data))
        state.remove_edge(we)
        if not any(e.src_conn == conn for e in state.out_edges(body)):
            body.remove_out_connector(conn)
        info.write_edge = new_we
        return new_we

    def _lift(self, state: dace.SDFGState, info: MapReductionInfo) -> bool:
        """Perform the feed-identity lift for one recognised reduction.

        :param state: The state holding the reduction map.
        :param info: The recognised reduction.
        :returns: ``True`` if the lift was applied, ``False`` if a structural /
            semantic precondition failed (the SDFG is left unchanged).
        """
        sdfg = state.sdfg
        me, mx = info.map_entry, info.map_exit
        acc = info.accumulator
        param = me.map.params[0]
        lb, ub, _ = me.map.range[-1]
        trip = symbolic.simplify(ub - lb + 1)
        dtype = sdfg.arrays[acc].dtype

        wcr = _WCR_LAMBDA.get(info.op)
        if wcr is None:
            return False
        try:
            identity_val = float(info.identity)
        except (TypeError, ValueError):
            return False  # defensive: only +/* reach here, with finite identities 0/1

        # --- validate every precondition BEFORE mutating (atomic lift) ---
        # Locate the accumulator's map-entry feed and post-map sink. The inout
        # split retargets only the body->map_exit *src* connector, so these lookups
        # (keyed off the unchanged ``write_edge.dst_conn`` / read connector) stay valid.
        read_in_conn = "IN_" + info.read_edge.src_conn[len("OUT_"):]
        me_in = [e for e in state.in_edges(me) if e.dst_conn == read_in_conn]
        write_out_conn = "OUT_" + info.write_edge.dst_conn[len("IN_"):]
        mx_out = [e for e in state.out_edges(mx) if e.src_conn == write_out_conn]
        if len(me_in) != 1 or len(mx_out) != 1:
            return False
        me_in_edge, mx_out_edge = me_in[0], mx_out[0]
        acc_in_node, acc_out_node = me_in_edge.src, mx_out_edge.dst
        if not (isinstance(acc_in_node, nodes.AccessNode) and acc_in_node.data == acc):
            return False
        if not (isinstance(acc_out_node, nodes.AccessNode) and acc_out_node.data == acc):
            return False

        # Correctness gate: accumulator pre-seeded to the op identity, so seeding
        # the fold with the identity reproduces ``init (op) fold``.
        init_edges = state.in_edges(acc_in_node)
        if not init_edges:
            return False
        for ie in init_edges:
            if not isinstance(ie.src, nodes.Tasklet):
                return False
            val = _const_assign_value(ie.src.code.as_string)
            if val is None or val != identity_val:
                return False

        # --- all preconditions hold; mutate from here on ---
        # Split a shared inout accumulator connector so identity (in) and product
        # buffer (out) ride separate ports.
        write_edge = self._split_inout_connector(state, info)
        if write_edge is None:
            return False

        buf, _ = sdfg.add_transient(f"_red_buf_{acc}", (trip, ), dtype, find_new_name=True)
        zero, _ = sdfg.add_scalar(f"_red_zero_{acc}", dtype, transient=True, find_new_name=True)

        # READ side: rename pre-map accumulator + its seed writes to the fresh
        # identity scalar, so ``acc`` has a single writer (the Reduce) -- no
        # two-access-node aliasing.
        acc_in_node.data = zero
        me_in_edge.data = dace.Memlet(f"{zero}[0]")
        info.read_edge.data = dace.Memlet(f"{zero}[0]")
        for ie in init_edges:
            ie.data = dace.Memlet(f"{zero}[0]")

        # WRITE side: per-iteration result -> product buffer (drop the carry).
        write_edge.data = dace.Memlet(f"{buf}[{param} - ({lb})]")
        buf_node = state.add_access(buf)
        state.remove_edge(mx_out_edge)
        state.add_edge(mx, write_out_conn, buf_node, None, dace.Memlet(f"{buf}[0:{trip}]"))

        # Reduce(buf) -> acc, vectorized.
        red = state.add_reduce(wcr, axes=[0], identity=identity_val)
        if self._vectorized:
            red.implementation = "vectorized"
        state.add_edge(buf_node, None, red, None, dace.Memlet(f"{buf}[0:{trip}]"))
        state.add_edge(red, None, acc_out_node, None, dace.Memlet(f"{acc}[0]"))

        # If the reduced trip depends on data-dependent symbols (spmv
        # ``row_start``/``row_end`` = ``indptr[i]`` / ``indptr[i+1]``, bound by an
        # interstate-edge assignment from a scalar), wrap product-map/buffer/Reduce
        # in a single-iteration map whose dynamic-range connectors re-define them
        # from their scalar sources -- keeping them in scope through re-nesting. An
        # interstate binding does not survive ``ExpandNestedSDFGInputs``' re-nest
        # (symbol demanded from a parent that no longer defines it); a
        # dynamic-range connector rides a data edge and does.
        self._scope_dynamic_range_symbols(state, me, mx, buf_node, red)

        sdfg.validate()
        return True

    @staticmethod
    def _scope_dynamic_range_symbols(state: dace.SDFGState, me: nodes.MapEntry, mx: nodes.MapExit,
                                     buf_node: nodes.AccessNode, red: nodes.LibraryNode) -> None:
        """Wrap the lifted product-map + buffer + ``Reduce`` in a single-iteration
        map that re-defines the product-map's data-dependent range symbols as
        dynamic-range connectors.

        Only ``me`` range symbols that are (a) SDFG symbols AND (b) bound by an
        interstate-edge assignment to a plain ``Scalar`` are scoped; a genuine
        free/global symbol (``N`` in ``0:N``) is already in scope -> no wrap. The
        wrap reuses the original symbol names on the connectors: they shadow the
        interstate-bound symbols with the SAME value, so SDFG-level buffer-shape
        references stay satisfied while the in-scope copy survives the re-nest.

        :param state: State holding the lifted reduction.
        :param me: Product-fill map entry (range carries the symbols).
        :param mx: Product-fill map exit.
        :param buf_node: Product buffer access node.
        :param red: The ``Reduce`` library node.
        """
        sdfg = state.sdfg
        range_syms = {str(s) for s in me.map.range.free_symbols} & set(sdfg.symbols)
        if not range_syms:
            return

        # Resolve each range symbol to its interstate-edge scalar source.
        scalar_src = {}
        for ie in sdfg.edges():
            for sym, rhs in ie.data.assignments.items():
                if sym in range_syms and rhs is not None:
                    rhs = rhs.strip()
                    if rhs in sdfg.arrays and isinstance(sdfg.arrays[rhs], dace.data.Scalar):
                        scalar_src[sym] = rhs
        if set(scalar_src) != range_syms:
            return  # not all symbols scalar-bound dynamic ranges; nothing to scope

        cluster = (set(state.all_nodes_between(me, mx)) | {me, mx, buf_node, red})
        me_w, mx_w = state.add_map("reduce_scope", {"__reduce_scope_it": "0:1"})

        # Dynamic-range connectors re-defining the symbols from their scalars.
        for sym, src in scalar_src.items():
            me_w.add_in_connector(sym)
            state.add_edge(state.add_access(src), None, me_w, sym, dace.Memlet(f"{src}[0]"))

        # Route every cluster<->outside edge through the wrap map. Internal edges
        # and the dynamic-range feeds just added are left alone.
        idx = 0
        for e in list(state.all_edges(*cluster)):
            if e.src in (me_w, mx_w) or e.dst in (me_w, mx_w):
                continue
            src_in, dst_in = e.src in cluster, e.dst in cluster
            if src_in == dst_in:
                continue  # internal edge or unrelated
            idx += 1
            ic, oc = f"IN_rsc{idx}", f"OUT_rsc{idx}"
            gate = me_w if dst_in else mx_w
            gate.add_in_connector(ic)
            gate.add_out_connector(oc)
            state.add_edge(e.src, e.src_conn, gate, ic, copy.deepcopy(e.data))
            state.add_edge(gate, oc, e.dst, e.dst_conn, copy.deepcopy(e.data))
            state.remove_edge(e)

        # The buffer's symbolic shape is only valid inside the wrap scope where
        # the symbols are defined; allocate it there.
        sdfg.arrays[buf_node.data].lifetime = dace.dtypes.AllocationLifetime.Scope


class LiftWCRMapToBufferAndReduce(LiftMapReductionToReduce):
    """Opt-in: lift a pure-WCR boundary reduction to a product buffer + ``Reduce``.

    Canonical ``acc = sum(A)`` (scalar ``CR:op`` WCR at a MapExit, no carry-in) is
    by default left AS a map-exit WCR for direct codegen (CPU ``reduction(op:var)``
    OpenMP clause; GPU thread-block reduce + one atomic per block). This pass is the
    alternative: redirect the per-iteration result to a fresh 1-D buffer + fold with
    a ``Reduce`` libnode (product-fill map the tiler strides + device/vectorized
    ``Reduce``). Run explicitly by callers who want the buffer + libnode form (e.g.
    to pick a specific ``Reduce`` expansion); NOT in the default multi-dim
    vectorization pipeline.

    Thin alias of :class:`LiftMapReductionToReduce` restricted to the pure-WCR lift.
    """

    def __init__(self, vectorized: bool = True):
        super().__init__(vectorized=vectorized, pure_wcr_only=True)
