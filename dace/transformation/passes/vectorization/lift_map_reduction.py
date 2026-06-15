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
from typing import Optional

import dace
from dace import nodes, symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.reductions import (
    MapReductionInfo,
    recognize_map_reduction,
)

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


class LiftMapReductionToReduce(ppl.Pass):
    """Lift map-carried scalar reductions to product-map + ``Reduce`` libnode.

    :param vectorized: stamp ``implementation="vectorized"`` on the emitted
        ``Reduce`` so :class:`ExpandReduceVectorized` lowers it (the
        self-contained, no-tile-node CPU vectorized fold). Default ``True``.
    """

    def __init__(self, vectorized: bool = True):
        super().__init__()
        self._vectorized = vectorized

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
            info = recognize_map_reduction(state, me)
            if info is None:
                continue
            if self._lift(state, info):
                count += 1
        return count or None

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

        sdfg.validate()
        return True
