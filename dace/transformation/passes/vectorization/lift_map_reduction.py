# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a scalar reduction carried across an innermost map to a ``Reduce`` libnode.

The spmv "row reduction" ``for idx: acc = acc + data[idx] * x[indices[idx]]``
is, in SDFG form, an innermost map whose single body node (an *indirect-access*
NestedSDFG that cannot be inlined) reads a scalar ``acc[0]`` at the map entry
and writes it back at the map exit -- a loop-carried read-modify-write. The
legacy 1-D ``VectorizeCPU`` mis-vectorizes this RMW once the reduced trip
exceeds the vector width (the per-chunk partial sums are never folded), so the
reduction must instead be expressed as the canonical product-map + ``Reduce``
shape the vectorizer already lowers correctly.

This pass performs that lift **without touching the opaque body**, via a
*feed-identity* rewrite:

1. The carried accumulator is pre-seeded to the reduction identity; we feed
   that identity (broadcast, no longer carried) into the map in place of the
   loop-carried value, so the body now computes ``identity (op) expr == expr``
   -- i.e. just the per-iteration product, gather included.
2. The per-iteration result is written to a fresh 1-D buffer ``acc_buf[idx-lb]``
   instead of back onto the scalar.
3. A ``Reduce`` libnode folds ``acc_buf`` into the accumulator
   (``implementation="vectorized"`` -> :class:`ExpandReduceVectorized`, a
   self-contained ``horizontal_reduce_<op>`` kernel with a scalar tail).

The product-fill map is then an ordinary gather + product map the vectorizer
strides correctly (scalar remainder on the reduced dim keeps the gather tail
in-range); the ``Reduce`` carries its own vectorized fold + scalar tail.

Detection is :func:`recognize_map_reduction` (an extension of the flat
:func:`recognize_reduction` utility); only reductions whose accumulator is
pre-initialised to the operator identity are lifted (so seeding the fold with
the identity reproduces the original ``init (op) fold`` semantics).
"""
import ast
import copy
from typing import Optional, Tuple

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

#: Reduction-op token -> the ``Reduce`` libnode WCR lambda string. Restricted to
#: ``+`` / ``*`` -- the only ops this lift is designed and tested for (the spmv
#: row-sum, plus product), and whose identities (0 / 1) are finite numerics that
#: ``add_reduce`` lowers cleanly. ``max`` / ``min`` (identities -inf / +inf) and
#: the bitwise ops are deliberately excluded: their identities do not round-trip
#: through the finite-float gate below, so they are rejected here at the WCR
#: lookup rather than mis-lifted with a non-finite identity.
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
    """A pure-WCR boundary reduction recognised by :func:`_recognize_pure_wcr_reduction`.

    The ``body -> map_exit`` ``write_edge`` carries the scalar accumulator with a
    ``CR:op`` WCR and the accumulator is *not* read at the map entry (no
    loop-carried RMW -- the WCR alone expresses the fold).
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
    """Recognise ``acc (op)= f(...)`` expressed as a MapExit WCR with no carry-in.

    The canonical ``acc = sum(A)`` shape: a single-param, unit-step innermost map
    with one flat-tasklet body whose ``body -> map_exit`` edge writes a scalar
    accumulator under a ``CR:+`` / ``CR:*`` WCR, and where that accumulator is
    *not* also read at the map entry. (The loop-carried RMW shape -- accumulator
    read at entry + written at exit -- is :func:`recognize_map_reduction`.)

    :returns: A :class:`PureWCRReductionInfo`, or ``None`` if not recognised.
    """
    from dace.frontend.operations import detect_reduction_type
    if not isinstance(map_entry, dace.nodes.MapEntry):
        return None
    if len(map_entry.map.params) != 1:
        return None
    _, _, step = map_entry.map.range[-1]
    if (step != 1) and (str(step) != "1"):
        return None
    map_exit = state.exit_node(map_entry)
    inner = state.all_nodes_between(map_entry, map_exit) or set()
    if any(isinstance(n, dace.nodes.MapEntry) for n in inner):
        return None
    body_nodes = [n for n in inner if n not in (map_entry, map_exit)]
    if len(body_nodes) != 1 or not isinstance(body_nodes[0], dace.nodes.Tasklet):
        return None
    body = body_nodes[0]

    def _scalar_slot(e) -> bool:
        return (e.data is not None and e.data.data is not None and e.data.subset is not None
                and e.data.subset.num_elements() == 1)

    wcr_writes = [
        e for e in state.in_edges(map_exit) if e.src is body and _scalar_slot(e) and e.data.wcr is not None
    ]
    if len(wcr_writes) != 1:
        return None
    write_edge = wcr_writes[0]
    acc = write_edge.data.data
    desc = state.sdfg.arrays.get(acc)
    if desc is None or not isinstance(desc, (dace.data.Scalar, dace.data.Array)):
        return None
    op = _REDTYPE_OP.get(detect_reduction_type(write_edge.data.wcr))
    if op is None or op not in IDENTITY:
        return None
    # Pure WCR: the accumulator must NOT be read at the map entry (else it is a
    # loop-carried RMW -- handled by the other recogniser) and must not appear on
    # any other body edge (no aliasing we would silently break).
    if any(e.data is not None and e.data.data == acc for e in state.out_edges(map_entry)):
        return None
    other_acc = [
        e for e in state.all_edges(body) if e is not write_edge and e.data is not None and e.data.data == acc
    ]
    if other_acc:
        return None
    return PureWCRReductionInfo(map_entry, map_exit, body, acc, op, write_edge)


class LiftMapReductionToReduce(ppl.Pass):
    """Lift map-carried scalar reductions to product-map + ``Reduce`` libnode.

    :param vectorized: stamp ``implementation="vectorized"`` on the emitted
        ``Reduce`` so :class:`ExpandReduceVectorized` lowers it (the
        self-contained, no-tile-node CPU vectorized fold). Default ``True``.
    """

    def __init__(self, vectorized: bool = True, pure_wcr_only: bool = False):
        super().__init__()
        self._vectorized = vectorized
        #: When set, lift ONLY pure-WCR boundary reductions (skip the
        #: loop-carried RMW recogniser). Used for the early pipeline call that
        #: must run before ``WCRToAugAssign`` rewrites the WCR away; the RMW
        #: shape is lifted later, after ``LoopToMap`` has produced the map.
        self._pure_wcr_only = pure_wcr_only

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
            # Pure-WCR boundary reduction (``acc(CR:op)`` at the MapExit, no
            # carry-in read) is lifted directly to a product buffer + vectorized
            # Reduce; this is the canonical ``acc = sum(A)`` map that DaCe emits
            # for ``acc(1, lambda x, y: x + y)``. Tried first so it fires before
            # the RMW recogniser (which needs a map-entry carry-in read).
            pure = _recognize_pure_wcr_reduction(state, me)
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

        The body writes a per-iteration value to the accumulator through a
        ``CR:op`` WCR at the map exit (no carry-in read). We redirect that write
        to a fresh 1-D buffer (dropping the WCR -- it becomes an ordinary
        per-iteration store the tiler strides) and fold the buffer into the
        accumulator with a ``Reduce`` libnode. The ``Reduce -> acc`` edge keeps
        the ``CR:op`` WCR so the original ``acc (op)= fold`` semantics survive
        for any initial accumulator value (the test seeds ``acc = 0``).

        :returns: ``True`` if lifted, ``False`` if a precondition failed (SDFG
            left unchanged).
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

        When the body reads and writes the accumulator through a single *inout*
        connector (the s4115 ``sum_val`` shape -- same name in ``in_connectors``
        and ``out_connectors``), the feed-identity lift would feed two different
        outer arrays (the identity in, the buffer out) through one connector,
        which is invalid. Split it: rename the inner write-side access nodes to a
        fresh array and expose it as a new output connector, leaving the read on
        the original connector.

        :param state: The state holding the body node.
        :param info: The recognised reduction (its ``write_edge`` is replaced
            when a split occurs).
        :returns: The (possibly new) ``body -> map_exit`` write edge, or ``None``
            if the inout connector could not be split safely.
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
        # Locate the carried accumulator's map-entry feed and the post-map sink.
        # The inout split below only retargets the body->map_exit *src* connector,
        # never the map-entry feed or the map_exit->sink edge, so these lookups
        # (keyed off the unchanged ``info.write_edge.dst_conn`` / read connector)
        # stay valid across it.
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

        # Correctness gate: the accumulator must be pre-seeded to the op
        # identity, so seeding the fold with the identity reproduces the
        # original ``init (op) fold`` value.
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
        # Normalise a shared inout accumulator connector into distinct in/out so
        # the identity (in) and the product buffer (out) ride separate ports.
        write_edge = self._split_inout_connector(state, info)
        if write_edge is None:
            return False

        buf, _ = sdfg.add_transient(f"_red_buf_{acc}", (trip, ), dtype, find_new_name=True)
        zero, _ = sdfg.add_scalar(f"_red_zero_{acc}", dtype, transient=True, find_new_name=True)

        # READ side: rename the pre-map accumulator + its seed writes to the
        # fresh identity scalar so the carried scalar now has a single writer
        # (the Reduce below) -- no two-access-node aliasing on ``acc``.
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

        # If the reduced trip depends on data-dependent symbols (spmv's
        # ``row_start``/``row_end`` = ``indptr[i]`` / ``indptr[i+1]``, bound by
        # an interstate-edge assignment from a scalar), keep those symbols in
        # scope for the buffer + Reduce through the downstream re-nesting passes
        # by wrapping the product-map/buffer/Reduce in a single-iteration map
        # whose dynamic-range connectors re-define them from their scalar
        # sources.  An interstate-edge binding does not survive
        # ``ExpandNestedSDFGInputs``' re-nest (the symbol gets demanded from a
        # parent scope that no longer defines it); a dynamic-range connector is
        # carried as a data edge and so does.
        self._scope_dynamic_range_symbols(state, me, mx, buf_node, red)

        sdfg.validate()
        return True

    @staticmethod
    def _scope_dynamic_range_symbols(state: dace.SDFGState, me: nodes.MapEntry, mx: nodes.MapExit,
                                     buf_node: nodes.AccessNode, red: nodes.LibraryNode) -> None:
        """Wrap the lifted product-map + buffer + ``Reduce`` in a single-iteration
        map that re-defines the product-map's data-dependent range symbols as
        dynamic-range connectors.

        Only the symbols of ``me``'s range that are (a) SDFG symbols and (b)
        bound by an interstate-edge assignment to a plain ``Scalar`` source are
        scoped -- a genuine free/global symbol (e.g. ``N`` in a static
        ``0:N`` reduction) is already in scope everywhere and is left untouched
        (no wrap emitted). The wrap reuses the original symbol names on the
        connectors: they shadow the interstate-bound symbols with the SAME
        value, so the buffer-shape references (at SDFG level) stay satisfied by
        the surviving interstate binding while the in-scope copy survives the
        re-nest.

        :param state: The state holding the lifted reduction.
        :param me: The product-fill map entry (its range carries the symbols).
        :param mx: The product-fill map exit.
        :param buf_node: The product buffer access node.
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
            return  # not all symbols are scalar-bound dynamic ranges; nothing to scope

        cluster = (set(state.all_nodes_between(me, mx)) | {me, mx, buf_node, red})
        me_w, mx_w = state.add_map("reduce_scope", {"__reduce_scope_it": "0:1"})

        # Dynamic-range connectors re-defining the symbols from their scalars.
        for sym, src in scalar_src.items():
            me_w.add_in_connector(sym)
            state.add_edge(state.add_access(src), None, me_w, sym, dace.Memlet(f"{src}[0]"))

        # Route every cluster<->outside edge through the wrap map.  Internal
        # edges (both endpoints in the cluster) and the dynamic-range feeds we
        # just added are left alone.
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
