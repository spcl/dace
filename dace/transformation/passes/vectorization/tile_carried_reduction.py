# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a loop-carried scalar reduction over a tiled map into a partial-sum tile.

A reduction whose accumulator threads *through* the tiled map -- read at the
``MapEntry``, combined with a per-lane tile value via an associative op, written
back at the ``MapExit`` to the SAME scalar (subset independent of the tile
params) -- cannot be vectorized as a plain element-wise tile op: ``acc + tile``
is ``Scalar + Tile`` and the partial sums of the W lanes are never folded.

This pass rewrites such a reduction (design "Option B", carried partial tile):

* widen the carried accumulator to a ``(W,)`` partial-sum *tile*, identity-init;
* the body's combine becomes ``Tile <op> Tile`` (a plain ``TileBinop``), carried
  in place across the now-**sequential** tile-loop (lanes parallel within a tile);
* after the map, fold the partial tile to a scalar with a ``TileReduce`` and
  combine the original pre-loop init value (so a non-identity init stays correct).

The recogniser is deliberately tight (see :func:`find_carried_scalar_reductions`)
so it is a strict no-op on every non-reduction kernel: an in-place ``a[i]+=b[i]``
is excluded because its write subset *depends* on the tile param (distinct per
lane -- no carry), and an elementwise kernel has no accumulator threaded through
the map at all.
"""
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import dace
from dace import properties
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation

#: Associative / commutative ops whose partial sums may be folded independently.
SUPPORTED_REDUCE_OPS = ("+", "*", "min", "max")


class CarriedReduction(NamedTuple):
    """One recognised carried-scalar-reduction site.

    :ivar state: Parent state holding the tiled map.
    :ivar map_entry: The tiled map's entry.
    :ivar acc: Data name of the carried scalar accumulator (in the parent SDFG).
    :ivar op: The reduction op (one of :data:`SUPPORTED_REDUCE_OPS`).
    :ivar in_conn: NSDFG input connector the accumulator is read into.
    :ivar out_conn: NSDFG output connector the accumulator is written from.
    """
    state: SDFGState
    map_entry: MapEntry
    nsdfg: NestedSDFG
    acc: str
    op: str
    in_conn: str
    out_conn: str


def _subset_independent_of(subset, params) -> bool:
    """True iff ``subset`` references none of ``params`` (a true cross-iteration
    carry: the same element every iteration). A subset that uses a tile param is
    a per-lane-distinct write (in-place RMW), NOT a reduction carry."""
    if subset is None:
        return False
    pset = set(str(p) for p in params)
    return not (set(str(s) for s in subset.free_symbols) & pset)


def _carry_pair(state: SDFGState, me: MapEntry, mx: MapExit, sdfg: SDFG, params):
    """Yield ``(acc, in_conn, out_conn)`` for every scalar transient that is BOTH
    read in through ``me`` and written back out through ``mx`` to the same data,
    with a carry subset independent of the tile ``params``."""
    # Accumulators read into the body: AccessNode(acc) -> me:IN_acc -> OUT_acc -> body.
    read_in = {}
    for e in state.in_edges(me):
        if not isinstance(e.src, AccessNode):
            continue
        desc = sdfg.arrays.get(e.src.data)
        if not isinstance(desc, dace.data.Scalar) or not desc.transient:
            continue
        if not _subset_independent_of(e.data.subset, params):
            continue
        # The body connector is the OUT_<x> edge's dst_conn off the entry.
        out_es = [oe for oe in state.out_edges(me) if oe.src_conn == "OUT_" + e.dst_conn[len("IN_"):]]
        if len(out_es) != 1:
            continue
        read_in[e.src.data] = out_es[0].dst_conn
    # Accumulators written back out: body -> mx:IN_acc -> OUT_acc -> AccessNode(acc).
    for e in state.out_edges(mx):
        if not isinstance(e.dst, AccessNode):
            continue
        if e.dst.data not in read_in:
            continue
        if not _subset_independent_of(e.data.subset, params):
            continue
        in_es = [ie for ie in state.in_edges(mx) if ie.dst_conn == "IN_" + e.src_conn[len("OUT_"):]]
        if len(in_es) != 1:
            continue
        yield e.dst.data, read_in[e.dst.data], in_es[0].src_conn


def _body_combine_op(nsdfg: NestedSDFG, in_conn: str, out_conn: str) -> Optional[str]:
    """Return the associative op iff the body NSDFG combines the value read on
    ``in_conn`` into the value written on ``out_conn`` via a single
    ``__out = __a <op> __b`` tasklet (one operand tracing to ``in_conn``).
    Otherwise ``None`` -- the carry is not an associative reduction."""
    import ast
    body = nsdfg.sdfg
    # The accumulator enters as AccessNode(in_conn) and exits as AccessNode(out_conn).
    for state in body.states():
        in_ans = [n for n in state.nodes() if isinstance(n, AccessNode) and n.data == in_conn]
        out_ans = [n for n in state.nodes() if isinstance(n, AccessNode) and n.data == out_conn]
        if not in_ans or not out_ans:
            continue
        for t in [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]:
            if len(t.in_connectors) != 2 or len(t.out_connectors) != 1:
                continue
            # One input must trace (directly) to the accumulator-in AccessNode.
            srcs = {e.src.data for e in state.in_edges(t) if isinstance(e.src, AccessNode)}
            if in_conn not in srcs:
                continue
            if len(t.code.code) != 1 or not isinstance(t.code.code[0], ast.Assign):
                continue
            rhs = t.code.code[0].value
            op = _binop_symbol(rhs)
            if op in SUPPORTED_REDUCE_OPS:
                return op
    return None


def _binop_symbol(rhs) -> Optional[str]:
    """Map an ``ast`` BinOp / min|max Call RHS to its op symbol, else ``None``."""
    import ast
    pyop = {ast.Add: "+", ast.Mult: "*"}
    if isinstance(rhs, ast.BinOp) and type(rhs.op) in pyop:
        return pyop[type(rhs.op)]
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id in ("min", "max"):
        return rhs.func.id
    return None


def _identity_literal(op: str, dtype) -> str:
    """Python literal (as a string for a tasklet body) for ``op``'s identity at
    ``dtype``. Mirrors :func:`dace.libraries.tileops.nodes.tile_reduce._identity_literal`
    but as a Python init (the tile init runs as plain dataflow, not C++)."""
    if op == "+":
        return "0"
    if op == "*":
        return "1"
    if op == "min":
        return "1e308"
    if op == "max":
        return "-1e308"
    raise ValueError(f"no identity for reduction op {op!r}")


def _combine_py(op: str) -> str:
    """Python combine expression template ``__o = <op-combine of __i, __r>``."""
    if op in ("+", "*"):
        return f"__i {op} __r"
    return f"{op}(__i, __r)"  # min / max


@properties.make_properties
@transformation.explicit_cf_compatible
class TileCarriedScalarReduction(ppl.Pass):
    """Lower loop-carried scalar reductions over a tiled map to a partial-sum tile.

    Runs AFTER ``NestInnermostMapBodyIntoNSDFG`` (so the body NSDFG + the
    ``__tile_main`` / ``__scalar_tail`` split already exist) and BEFORE
    ``WidenAccesses`` (which then widens the now-tile body transients as usual).
    K=1 only for now (the carry is over a single innermost tiled dim); K>=2 is a
    safe no-op. Strict no-op on every non-reduction kernel (see
    :func:`find_carried_scalar_reductions`)."""

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.Property(dtype=tuple, default=(8, ), desc="Per-dim tile widths, innermost-last.")

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        super().__init__()
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Descriptors
                | ppl.Modifies.Scopes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        if len(self.widths) != 1:
            return None  # K>=2 carried reductions not handled yet -- safe no-op.
        count = 0
        for red in find_carried_scalar_reductions(sdfg, self.widths):
            self._rewrite_one(red)
            count += 1
        return count or None

    def _rewrite_one(self, red: CarriedReduction) -> None:
        """Rewrite one carried scalar reduction into the partial-tile form."""
        state, me, accname, op = red.state, red.map_entry, red.acc, red.op
        mx = state.exit_node(me)
        sdfg = state.sdfg
        W = self.widths[0]
        acc_desc = sdfg.arrays[accname]
        dtype = acc_desc.dtype

        idlit = _identity_literal(op, dtype)
        wcr = f"lambda a, b: a {op} b" if op in ("+", "*") else f"lambda a, b: {op}(a, b)"

        # --- 1. Two (W,) tiles, both identity-init via a tiny map each:
        #   * id_tile  -- the per-iteration carry-IN read (a CONSTANT identity, so the
        #     body's ``identity <op> a`` yields exactly the per-element contribution);
        #   * acc_tile -- the partial-sum accumulator, WCR-reduced at the carry-OUT
        #     (the cross-tile carry: a parallel/sequential map cannot thread a value
        #     through entry->exit, but a WCR write accumulates across iterations). ---
        from dace.transformation.passes.vectorization.split_map_for_tile_remainder import SCALAR_TAIL_MARKER

        def _make_identity_tile(tag):
            nm, _ = sdfg.add_array(f"{accname}_{tag}", (W, ), dtype, transient=True, find_new_name=True)
            an = state.add_access(nm)
            # Marker suffix keeps MarkTileDims / StrideMapByTileWidths from striding this
            # plain [0:W] init map to step W (which would init only lane 0).
            ime, imx = state.add_map(f"{nm}_init{SCALAR_TAIL_MARKER}", {"__li": f"0:{W}"})
            it = state.add_tasklet("ridentity", set(), {"__o"}, f"__o = {idlit}")
            state.add_edge(ime, None, it, None, Memlet())
            state.add_memlet_path(it, imx, an, src_conn="__o", memlet=Memlet(f"{nm}[__li]"))
            return nm, an

        id_name, id_tile = _make_identity_tile("idtile")
        acc_name, _ = _make_identity_tile("ptile")

        # --- 2. Rewire the MAIN map's carry. ---
        # carry-IN: read the constant identity tile (not the carried value).
        me_out = next(e for e in state.out_edges(me) if e.dst is red.nsdfg and e.dst_conn == red.in_conn)
        in_suffix = me_out.src_conn[len("OUT_"):]
        in_e = next(e for e in state.in_edges(me) if e.dst_conn == "IN_" + in_suffix)
        acc_pre = in_e.src  # the original pre-loop init scalar AccessNode
        state.remove_edge(in_e)
        state.add_edge(id_tile, None, me, "IN_" + in_suffix, Memlet(f"{id_name}[0:{W}]"))
        me_out.data = Memlet(data=id_name, subset=f"0:{W}")

        # carry-OUT: WCR-accumulate into the partial-sum tile across tile iterations.
        # Locate the body->exit edge by the (still inout) out connector first.
        mx_in = next(e for e in state.in_edges(mx) if e.src is red.nsdfg and e.src_conn == red.out_conn)
        out_suffix = mx_in.dst_conn[len("IN_"):]
        out_e = next(e for e in state.out_edges(mx) if e.src_conn == "OUT_" + out_suffix)
        acc_mid = out_e.dst  # scalar AccessNode the tail / consumer reads

        # --- 3. Split the body's inout accumulator into a read-side (fed identity)
        #        and a distinct write-side, and widen both to Tile so the combine
        #        is Tile<op>Tile. ---
        new_out = self._split_and_widen_body_carry(red, W)

        # Re-point the body->exit edge onto the new write connector + WCR.
        state.remove_edge(mx_in)
        state.add_edge(red.nsdfg, new_out, mx, mx_in.dst_conn, Memlet(data=acc_name, subset=f"0:{W}", wcr=wcr))
        tile_mid = state.add_access(acc_name)
        state.remove_edge(out_e)
        carry_out = state.add_edge(mx, "OUT_" + out_suffix, tile_mid, None, Memlet(f"{acc_name}[0:{W}]"))
        carry_out.data.wcr = wcr

        # --- 4. The partial-tile carry accumulates via WCR across tile iterations. ---
        # (left at the default/parallel schedule -- WCR handles the cross-iteration
        # conflict; revisit sequential non-atomic accumulation as an optimisation.)

        # --- 5. Fold the partial-sum tile -> scalar with TileReduce, combine init. ---
        from dace.libraries.tileops.nodes import TileReduce
        folded_name, _ = sdfg.add_scalar(f"{accname}_red", dtype, transient=True, find_new_name=True)
        folded = state.add_access(folded_name)
        red_node = TileReduce(name=f"{acc_name}_reduce", widths=(W, ), op=op, has_mask=False)
        state.add_node(red_node)
        state.add_edge(tile_mid, None, red_node, "_src", Memlet(f"{acc_name}[0:{W}]"))
        state.add_edge(red_node, "_dst", folded, None, Memlet(f"{folded_name}[0]"))
        # combine: acc_mid = orig_init <op> folded  (orig_init = acc_pre's value)
        comb = state.add_tasklet("rcombine", {"__i", "__r"}, {"__o"}, f"__o = {_combine_py(op)}")
        state.add_edge(acc_pre, None, comb, "__i", Memlet(f"{acc_pre.data}[0]"))
        state.add_edge(folded, None, comb, "__r", Memlet(f"{folded_name}[0]"))
        state.add_edge(comb, "__o", acc_mid, None, Memlet(f"{acc_mid.data}[0]"))

    def _split_and_widen_body_carry(self, red: CarriedReduction, W: int) -> str:
        """Split the body NSDFG's inout accumulator connector into a read-side
        (kept as ``in_conn``, fed the identity outside) and a DISTINCT write-side
        connector (WCR-accumulated outside), then widen both connector arrays to a
        ``(W,)`` tile (stretching incident carry-edge subsets to ``[0:W]``) so the
        body combine becomes ``Tile<op>Tile``. Returns the write-side connector
        name (unchanged when the body already had separate in/out connectors).

        DaCe requires an inout connector to bind the SAME array on both sides, so
        feeding identity in and WCR-ing a partial tile out is only possible once
        the connector is split."""
        body = red.nsdfg.sdfg
        nsdfg = red.nsdfg
        in_conn, out_conn = red.in_conn, red.out_conn
        new_out = out_conn
        if in_conn == out_conn:
            new_out = f"{out_conn}_wout"
            n = 0
            while new_out in body.arrays:
                n += 1
                new_out = f"{out_conn}_wout_{n}"
            desc = body.arrays[out_conn]
            body.add_array(new_out, desc.shape, desc.dtype, transient=desc.transient)
            # Rename the write-side boundary AccessNode(s) (written, not read on) so
            # the inner write targets a distinct array bound to the new out connector.
            for st in body.states():
                for node in list(st.nodes()):
                    if (isinstance(node, AccessNode) and node.data == out_conn and st.in_degree(node) > 0
                            and st.out_degree(node) == 0):
                        for e in list(st.in_edges(node)):
                            sub = e.data.subset if e.data is not None else None
                            st.remove_edge(e)
                            st.add_edge(e.src, e.src_conn, node, e.dst_conn, Memlet(data=new_out, subset=sub))
                        node.data = new_out
            nsdfg.remove_out_connector(out_conn)
            nsdfg.add_out_connector(new_out)
        # Widen the read-side + write-side connector arrays to a (W,) tile (both a
        # Scalar and a freshly-added len-1 Array for the split write side).
        for cname in {in_conn, new_out}:
            desc = body.arrays.get(cname)
            is_len1_arr = isinstance(desc, dace.data.Array) and all(bool(s == 1) for s in desc.shape)
            if isinstance(desc, dace.data.Scalar) or is_len1_arr:
                body.arrays[cname] = dace.data.Array(desc.dtype, (W, ), transient=desc.transient)
        for st in body.states():
            for node in st.nodes():
                if not isinstance(node, AccessNode) or node.data not in (in_conn, new_out):
                    continue
                for e in list(st.in_edges(node)) + list(st.out_edges(node)):
                    if e.data is not None and e.data.data == node.data:
                        e.data = Memlet(data=node.data, subset=f"0:{W}")
        return new_out


def find_carried_scalar_reductions(sdfg: SDFG, widths: Tuple[int, ...]) -> List[CarriedReduction]:
    """Find every loop-carried scalar reduction over a tiled (non-tail) map.

    Tight predicate (no false positives on non-reductions):

    * the map is an innermost tile map with >= K params, not a remainder tail;
    * a transient scalar is read in through the entry AND written back out
      through the exit to the SAME data, with a carry subset independent of the
      tile params (excludes in-place ``a[i]+=b[i]``, whose subset uses the param);
    * the body combines the read accumulator into the written one via a single
      associative ``+ / * / min / max`` tasklet.
    """
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                       TILE_K1_TAIL_MARKER)
    from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
    K = len(widths)
    out: List[CarriedReduction] = []
    for node, parent in sdfg.all_nodes_recursive():
        if not isinstance(node, MapEntry) or not isinstance(parent, SDFGState):
            continue
        try:
            if not is_innermost_map(parent, node):
                continue
        except (StopIteration, ValueError):
            continue
        if len(node.map.params) < K:
            continue
        if node.map.label.endswith(SCALAR_TAIL_MARKER) or node.map.label.endswith(TILE_K1_TAIL_MARKER):
            continue
        sdfg_of = parent.sdfg
        mx = parent.exit_node(node)
        try:
            scope_nodes = parent.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
        except (StopIteration, ValueError):
            continue
        nsdfgs = [n for n in scope_nodes if isinstance(n, NestedSDFG)]
        if len(nsdfgs) != 1:
            continue
        nsdfg = nsdfgs[0]
        for acc, in_conn, out_conn in _carry_pair(parent, node, mx, sdfg_of, node.map.params):
            op = _body_combine_op(nsdfg, in_conn, out_conn)
            if op is None:
                continue
            out.append(CarriedReduction(parent, node, nsdfg, acc, op, in_conn, out_conn))
    return out
