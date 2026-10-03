# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a scalar reduction carried across an innermost map to a ``Reduce`` libnode.

The spmv row reduction ``for idx: acc = acc + data[idx] * x[indices[idx]]`` is an
innermost map whose body (an indirect-access NestedSDFG that cannot be inlined)
reads scalar ``acc[0]`` at map entry and writes it at map exit -- a loop-carried
RMW. The tile vectorizer cannot widen a loop-carried scalar RMW directly (the
partial sums would never fold across lanes), so express it as the canonical
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

from typing import Any

import dace
from dace import dtypes, nodes, symbolic
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.map_predicates import map_body_nodes
from dace.transformation.passes.vectorization.utils.tasklets import single_assignment
from dace.transformation.passes.vectorization.utils.reductions import (
    IDENTITY,
    MapReductionInfo,
    recognize_map_reduction,
)
from dace.ordered import OrderedSet

#: Reduction-op token for each ``add_reduce``-friendly ``ReductionType``. Mirrors
#: the ``+`` / ``*`` restriction of :data:`_WCR_LAMBDA` (see its docstring).
_REDTYPE_OP = {
    dtypes.ReductionType.Sum: "+",
    dtypes.ReductionType.Product: "*",
}

#: Reduction-op token -> ``Reduce`` WCR lambda. Only ``+`` / ``*``: their identities are finite;
#: ``max`` / ``min`` / bitwise fail the finite-float gate below.
_WCR_LAMBDA = {
    "+": "lambda a, b: a + b",
    "*": "lambda a, b: a * b",
}


def _free_syms(expr: object) -> set[str]:
    # The free-symbol NAMES of ``expr``, or an empty set if it does not parse.
    try:
        return {str(s) for s in symbolic.pystr_to_symbolic(str(expr)).free_symbols}
    except Exception:  # noqa: BLE001 -- an unparseable bound is not a symbol dependence we model
        return set()


def _trip_depends_on_enclosing_map(state: dace.SDFGState, map_entry: nodes.MapEntry,
                                   trip: symbolic.SymbolicType) -> bool:
    # True if the reduction map's trip count is sized by a param of a map ENCLOSING it -- following the scope tree out
    # through nested-SDFG boundaries.
    syms = _free_syms(trip)
    cur_state, node = state, map_entry
    while syms and cur_state is not None:
        scope = cur_state.scope_dict()
        parent = scope.get(node)
        while parent is not None:
            if isinstance(parent, nodes.MapEntry) and syms & {str(p) for p in parent.map.params}:
                return True
            parent = scope.get(parent)
        # Top scope of this state -> ascend into the enclosing NestedSDFG node, re-expressing the
        # symbols in the OUTER SDFG's names. A symbol absent from the mapping is already an outer
        # name (a global like ``M``) and carries through unchanged.
        nsdfg_node = cur_state.sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            return False
        syms = set().union(*(_free_syms(nsdfg_node.symbol_mapping[s]) if s in nsdfg_node.symbol_mapping else {s}
                             for s in syms))
        node, cur_state = nsdfg_node, cur_state.sdfg.parent
    return False


def _const_assign_value(code: str) -> float | None:
    # Numeric value of a ``_out = <number>`` tasklet, or ``None``.
    assign = single_assignment(code)
    if assign is None:
        return None
    v = assign.value
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

    def __init__(self, map_entry: nodes.MapEntry, map_exit: nodes.MapExit, body: nodes.Node, accumulator: str, op: str,
                 write_edge: MultiConnectorEdge[Memlet]) -> None:
        self.map_entry = map_entry
        self.map_exit = map_exit
        self.body = body
        self.accumulator = accumulator
        self.op = op
        self.write_edge = write_edge


def _pure_wcr_map_ok(state: "dace.SDFGState",
                     map_entry: "dace.nodes.MapEntry") -> tuple[nodes.MapExit, list[nodes.Node], str] | None:
    # Shared map-level guards for a pure-WCR reduction: single-param, unit-step, top-level (within its state) map whose
    # body holds no nested map.
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
    # Scope membership, not ``all_nodes_between``: an emptied walk holds no MapEntry, so a map
    # nesting another one would pass this innermost guard.
    inner = map_body_nodes(state, map_entry)
    if any(isinstance(n, dace.nodes.MapEntry) for n in inner):
        return None
    return map_exit, inner, map_entry.map.params[0]


def _validate_pure_wcr_write(state: dace.SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit,
                             inner: list[nodes.Node], param: str,
                             write_edge: MultiConnectorEdge[Memlet]) -> PureWCRReductionInfo | None:
    # Per-write guards: one scalar ``body -> map_exit`` WCR edge writing a FIXED (param-independent) scalar accumulator
    # not read at map entry / aliased in scope.
    from dace.frontend.operations import detect_reduction_type
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


def _scalar_wcr_writes(state: dace.SDFGState, map_exit: nodes.MapExit) -> list[MultiConnectorEdge[Memlet]]:
    # Every scalar-slot (single-element) WCR write edge into ``map_exit``.
    return [
        e for e in state.in_edges(map_exit) if e.data is not None and e.data.data is not None
        and e.data.subset is not None and e.data.subset.num_elements() == 1 and e.data.wcr is not None
    ]


def _recognize_pure_wcr_reductions(state: "dace.SDFGState",
                                   map_entry: "dace.nodes.MapEntry") -> "list[PureWCRReductionInfo]":
    # Recognise EVERY independent pure-WCR scalar reduction on ``map_entry``.
    ok = _pure_wcr_map_ok(state, map_entry)
    if ok is None:
        return []
    map_exit, inner, param = ok
    writes = _scalar_wcr_writes(state, map_exit)
    if not writes:
        return []
    if len({e.data.data for e in writes}) != len(writes):
        return []  # two writes to the same accumulator -> not independent
    infos = []
    for write_edge in writes:
        info = _validate_pure_wcr_write(state, map_entry, map_exit, inner, param, write_edge)
        if info is None:
            return []
        infos.append(info)
    return infos


def _recognize_pure_wcr_reduction(state: "dace.SDFGState",
                                  map_entry: "dace.nodes.MapEntry") -> PureWCRReductionInfo | None:
    # Recognise ``acc (op)= f(...)`` as a MapExit WCR with no carry-in.
    infos = _recognize_pure_wcr_reductions(state, map_entry)
    return infos[0] if len(infos) == 1 else None


class LiftMapReductionToReduce(ppl.Pass):
    """Lift map-carried scalar reductions to product-map + ``Reduce`` libnode.

    :param vectorized: stamp ``implementation="vectorized"`` on the emitted
        ``Reduce`` so :class:`ExpandReduceVectorized` lowers it (the
        self-contained, no-tile-node CPU vectorized fold). Default ``True``.
    """

    def __init__(self,
                 vectorized: bool = True,
                 pure_wcr_only: bool = False,
                 rmw_only: bool = False,
                 nested_only: bool = False,
                 wcr_free_output: bool = False) -> None:
        super().__init__()
        self._vectorized = vectorized
        #: Lift ONLY pure-WCR boundary reductions (skip RMW recogniser). For the
        #: early pipeline call before ``WCRToAugAssign`` rewrites the WCR away; the
        #: RMW shape is lifted later, after ``LoopToMap`` produces the map.
        self._pure_wcr_only = pure_wcr_only
        #: Lift only the loop-carried RMW; leave pure-WCR ``acc = sum(A)`` as a map-exit WCR for codegen
        #: (OpenMP reduction / GPU block reduce). Mutually exclusive with ``pure_wcr_only``.
        self._rmw_only = rmw_only
        #: Lift only reductions whose map is inside a nested SDFG: those cannot keep a boundary WCR (no WCR
        #: inside NSDFGs), so they become buffer + ``Reduce``. Used together with ``wcr_free_output``.
        self._nested_only = nested_only
        #: Emit the fold without a WCR: ``Reduce(buf) -> _partial`` then ``acc = acc <op> _partial``, since
        #: an in-NSDFG WCR output edge would be dropped by the tile emitter. Value-preserving.
        self._wcr_free_output = wcr_free_output

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
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
            # ``nested_only``: skip a top-level reduction (kept as a map-exit WCR); only a
            # reduction trapped inside a body NSDFG needs materialising to a buffer + Reduce.
            if self._nested_only and state.sdfg.parent_nsdfg_node is None:
                continue
            # Pure-WCR boundary reductions (``acc(CR:op)`` at MapExit, no carry-in); several independent
            # accumulators (azimint ``s += a[j]; cnt += 1``) lift in place. Tried before the RMW recognizer.
            pures = [] if self._rmw_only else _recognize_pure_wcr_reductions(state, me)
            if pures:
                lifted = 0
                for pure in pures:
                    if self._lift_pure_wcr(state, pure):
                        lifted += 1
                if lifted:
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
        # Lift a pure-WCR boundary reduction to a product buffer + Reduce.
        sdfg = state.sdfg
        me, mx, acc, op, write_edge = (info.map_entry, info.map_exit, info.accumulator, info.op, info.write_edge)
        param = me.map.params[0]
        lb, ub, _ = me.map.range[-1]
        trip = symbolic.simplify(ub - lb + 1)
        if _trip_depends_on_enclosing_map(state, me, trip):
            return False
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
        # The accumulator slot actually written (``acc[i]`` / ``acc[0]``), preserved so the
        # fold writes the same element -- captured before the buffer redirect below.
        acc_subset = copy.deepcopy(mx_out_edge.data.subset)

        # all preconditions hold; mutate from here on
        buf, _ = sdfg.add_transient(f"_red_buf_{acc}", (trip, ), dtype, find_new_name=True)
        # Per-iteration result -> product buffer (drop the WCR carry).
        write_edge.data = dace.Memlet(f"{buf}[{param} - ({lb})]")
        buf_node = state.add_access(buf)
        state.remove_edge(mx_out_edge)
        state.add_edge(mx, write_out_conn, buf_node, None, dace.Memlet(f"{buf}[0:{trip}]"))

        red = state.add_reduce(wcr, axes=[0], identity=identity_val)
        if self._vectorized:
            red.implementation = "vectorized"
        state.add_edge(buf_node, None, red, '_in', dace.Memlet(f"{buf}[0:{trip}]"))
        if self._wcr_free_output:
            # Reduce(buf) -> _partial (plain), then acc = acc <op> _partial (plain RMW). No
            # WCR survives, so the reduction is legal inside a body NSDFG.
            from dace.transformation.dataflow.wcr_conversion import _wcr_augassign_body
            partial, _ = sdfg.add_scalar(f"_red_partial_{acc}", dtype, transient=True, find_new_name=True)
            partial_node = state.add_access(partial)
            state.add_edge(red, '_out', partial_node, None, dace.Memlet(f"{partial}[0]"))
            fold = state.add_tasklet("reduce_accum", OrderedSet(('__in1', '__in2')), {"__out"},
                                     f"__out = {_wcr_augassign_body(wcr)}")
            state.add_edge(state.add_access(acc), None, fold, "__in1",
                           dace.Memlet(data=acc, subset=copy.deepcopy(acc_subset)))
            state.add_edge(partial_node, None, fold, "__in2", dace.Memlet(f"{partial}[0]"))
            state.add_edge(fold, "__out", acc_node, None, dace.Memlet(data=acc, subset=copy.deepcopy(acc_subset)))
            return True
        # Reduce(buf) -> acc, WCR-accumulated into the prior acc (top-level boundary form).
        out_mem = dace.Memlet(data=acc, subset=copy.deepcopy(acc_subset))
        out_mem.wcr = wcr
        state.add_edge(red, '_out', acc_node, None, out_mem)
        return True

    @staticmethod
    def _split_inout_connector(state: dace.SDFGState, info: MapReductionInfo) -> MultiConnectorEdge[Memlet] | None:
        # Give the accumulator distinct in/out connectors on the body NSDFG.
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
        # Perform the feed-identity lift for one recognised reduction.
        sdfg = state.sdfg
        me, mx = info.map_entry, info.map_exit
        acc = info.accumulator
        param = me.map.params[0]
        lb, ub, _ = me.map.range[-1]
        trip = symbolic.simplify(ub - lb + 1)
        if _trip_depends_on_enclosing_map(state, me, trip):
            return False
        dtype = sdfg.arrays[acc].dtype

        wcr = _WCR_LAMBDA.get(info.op)
        if wcr is None:
            return False
        try:
            identity_val = float(info.identity)
        except (TypeError, ValueError):
            return False  # defensive: only +/* reach here, with finite identities 0/1

        # validate every precondition BEFORE mutating (atomic lift)
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
        # Ordering edges seed nothing; overwriting one with a real memlet would leave the
        # accumulator with no writer and read uninitialized memory as the identity.
        slot = copy.deepcopy(info.write_edge.data.subset)  # the fixed accumulator element, not always [0]
        init_edges = [e for e in state.in_edges(acc_in_node) if not e.data.is_empty()]
        if not init_edges:
            return False
        for ie in init_edges:
            if not isinstance(ie.src, nodes.Tasklet):
                return False
            # a seed of another element does not initialise the accumulator
            if ie.data.data != acc or ie.data.subset != slot:
                return False
            val = _const_assign_value(ie.src.code.as_string)
            if val is None or val != identity_val:
                return False

        # all preconditions hold; mutate from here on
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
        state.add_edge(buf_node, None, red, '_in', dace.Memlet(f"{buf}[0:{trip}]"))
        state.add_edge(red, '_out', acc_out_node, None, dace.Memlet(data=acc, subset=slot))

        # A data-dependent trip (spmv ``indptr[i]`` bounds) is wrapped in a single-iteration map whose
        # dynamic-range connectors redefine the symbols; interstate bindings do not survive the re-nest.
        self._scope_dynamic_range_symbols(state, me, mx, buf_node, red)

        return True

    @staticmethod
    def _scope_dynamic_range_symbols(state: dace.SDFGState, me: nodes.MapEntry, mx: nodes.MapExit,
                                     buf_node: nodes.AccessNode, red: nodes.LibraryNode) -> None:
        # Wrap the lifted product-map + buffer + ``Reduce`` in a single-iteration map that re-defines the product-map's
        # data-dependent range symbols as dynamic-range connectors.
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

        # Scope membership, not ``all_nodes_between``: an emptied walk drops the fill body out of
        # the cluster, and every ``me -> body`` edge then reads as leaving the cluster and gets
        # rerouted through the wrap map.
        cluster: OrderedSet[nodes.Node] = OrderedSet(map_body_nodes(state, me))
        cluster |= (me, mx, buf_node, red)
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
