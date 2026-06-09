# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Convert in-body tasklets to ``TileBinop`` / ``TileUnop`` / ``TileITE``.

After :class:`StageInsideBody` walks every tile-tagged body NSDFG and
stages non-transient AccessNode reads through tile transients, the
body still holds raw tasklets that operate on per-lane scalar values.
This pass walks the same body NSDFGs and replaces each tasklet with
the corresponding tile lib node so the post-expansion pure-loop body
operates on tile-shape register transients (design section 5.1 +
section 6.7).

First-slice scope (this commit):

* BINARY tasklets whose body is exactly ``_o = _a <op> _b`` with
  ``<op>`` in :data:`_SUPPORTED_BINOPS`. Both operands must be Tile
  (the walker stages them through ``TileLoad`` bridges).
* Connector wiring: tasklet ``_a``/``_b``/``_o`` -> :class:`TileBinop`
  ``_a``/``_b``/``_c``; memlet on each edge preserved.

Deferred to subsequent slices:

* UNARY tasklets -> :class:`TileUnop` (mechanical follow-up).
* TERNARY / merge tasklets -> :class:`TileITE`.
* Reduction tasklets -> :class:`TileReduce`.
* Scalar / Symbol operand kinds (currently only Tile + Tile binops).
* Tasklets nested inside multi-state bodies / RMW chains.
"""
from typing import Any, Dict, Optional, Tuple

import dace
from dace import properties
from dace.libraries.tileops import TileBinop, TileITE, TileReduce, TileUnop
from dace.sdfg import SDFG
from dace.sdfg.nodes import MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

#: Subset of binary operators that map directly onto :class:`TileBinop`.
_SUPPORTED_BINOPS = {"+", "-", "*", "/", "min", "max"}

#: Subset of reduction operators that map directly onto :class:`TileReduce`.
#: Subset of :data:`_SUPPORTED_BINOPS` excluding non-associative ``-`` and ``/``.
_SUPPORTED_REDUCE_OPS = {"+", "*", "min", "max"}

#: Mapping ``tasklet-body form`` -> ``TileUnop op label``. Each value names the
#: ``TileUnop.op`` keyword to instantiate; each key form is matched against the
#: tasklet body (after the DaCe paren wrap is stripped). The leading ``-`` form
#: aliases ``neg``.
_SUPPORTED_UNOPS = {
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "floor",
    "ceil",
    "tanh",
}


@properties.make_properties
@transformation.explicit_cf_compatible
class ConvertTaskletsToTileOps(ppl.Pass):
    """Convert in-body tasklets to tile lib nodes (first slice: binary Tile+Tile only).

    :ivar widths: Per-tile-dim widths; mirrors :class:`StageInsideBody`.
    """

    CATEGORY: str = "Vectorization"

    widths = properties.Property(
        dtype=tuple,
        default=(8, ),
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        """Build the pass.

        :param widths: Per-tile-dim widths.
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"ConvertTaskletsToTileOps: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG.

        Mirror of the walker shape used by :class:`StageInsideBody` and
        :class:`PreparePerLaneIndices`.
        """
        K = len(self.widths)
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, MapEntry):
                continue
            if not isinstance(parent, SDFGState):
                continue
            try:
                if not is_innermost_map(parent, node):
                    continue
            except (StopIteration, ValueError):
                continue
            if len(node.map.params) < K:
                continue
            try:
                scope_nodes = parent.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
            except (StopIteration, ValueError):
                continue
            nsdfgs = [n for n in scope_nodes if isinstance(n, NestedSDFG)]
            if len(nsdfgs) != 1:
                continue
            yield parent, nsdfgs[0], node

    def _detect_binop(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str]]:
        """If ``tasklet`` is a simple binary ``_out = _a <op> _b`` body, return
        ``(out_conn, a_conn, b_conn, op)``. Otherwise ``None``.
        """
        if len(tasklet.in_connectors) != 2 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        in_conns = list(tasklet.in_connectors)
        # Try every op in _SUPPORTED_BINOPS against the two operand orderings. DaCe wraps the
        # RHS of an assignment tasklet in parentheses (``_o = (_a + _b)``); accept both forms.
        for op in _SUPPORTED_BINOPS:
            for a, b in (in_conns, list(reversed(in_conns))):
                if op in ("min", "max"):
                    forms = (f"{out_conn} = {op}({a}, {b})", f"{out_conn} = ({op}({a}, {b}))")
                else:
                    forms = (f"{out_conn} = {a} {op} {b}", f"{out_conn} = ({a} {op} {b})")
                if body in forms:
                    return out_conn, a, b, op
        return None

    def _detect_unop(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str]]:
        """If ``tasklet`` is a simple unary ``_out = <op>(_a)`` body (or ``_out = -_a``),
        return ``(out_conn, a_conn, op_label)``. Otherwise ``None``.

        Accepts both the DaCe-emitted parenthesised RHS form and the bare form.
        ``op_label`` is one of :data:`_SUPPORTED_UNOPS`.
        """
        if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        # Negation: ``_o = -_a`` / ``_o = (-_a)`` / ``_o = (- _a)`` (DaCe sometimes inserts a
        # space between the unary minus and the operand).
        for form in (f"{out_conn} = -{a_conn}", f"{out_conn} = (-{a_conn})", f"{out_conn} = (- {a_conn})"):
            if body == form:
                return out_conn, a_conn, "neg"
        # Function-call ops: ``_o = abs(_a)`` / ``_o = math.exp(_a)`` / etc. Accept both bare
        # name and ``math.`` / ``std::`` prefixes (RemoveMathCall strips ``math.`` upstream;
        # this handles the case where it didn't run or wasn't needed).
        for op in _SUPPORTED_UNOPS:
            if op == "neg":
                continue
            prefixes = (op, f"math.{op}", f"std::{op}")
            for pref in prefixes:
                for form in (f"{out_conn} = {pref}({a_conn})", f"{out_conn} = ({pref}({a_conn}))"):
                    if body == form:
                        return out_conn, a_conn, op
        return None

    def _widen_output_transient_for_tile(self, inner_state: SDFGState, out_edge) -> None:
        """Forward-analysis transient widening (per user direction 2026-06-09).

        When a tile-op produces a Tile output (any input is Tile), the destination
        transient must be tile-shape to receive the per-lane stores. If the original
        tasklet's downstream transient is length-1 (a Scalar or shape ``(1,)`` Array
        produced by the python frontend's default scalar tasklet wiring), widen it
        in place to ``(widths,)`` and rewrite the output memlet's subset
        accordingly.

        Per the user: "When expanding transient scalars inside the nsdfg, we should
        expand accordingly by analyzing tasklet types, so that we don't need to
        narrow afterwards." This helper does the analysis at conversion time
        (the producer's input kinds drive the widening), avoiding a post-hoc
        narrowing pass.

        Non-transient destinations are NOT widened (the walker mints a bridge +
        TileStore at the boundary; the bridge is already tile-shape). Already
        tile-shape transients are left alone.
        """
        import dace.data as dd
        if out_edge.data is None or out_edge.data.data is None:
            return
        sdfg = inner_state.sdfg
        desc = sdfg.arrays.get(out_edge.data.data)
        if desc is None or not desc.transient:
            return
        widths = tuple(self.widths)
        # Already tile-shape? Nothing to do.
        if isinstance(desc, dd.Array):
            current_shape = tuple(desc.shape)
            if current_shape == widths:
                return
            # Only widen length-1 Arrays. Anything else is user-shaped; leave alone.
            if not all(bool(dace.symbolic.simplify(s - 1) == 0) for s in current_shape):
                return
        # Widen: replace the descriptor with a tile-shape Array of the same dtype.
        new_desc = dd.Array(dtype=desc.dtype,
                            shape=widths,
                            transient=True,
                            storage=desc.storage if hasattr(desc, "storage") else None)
        sdfg.arrays[out_edge.data.data] = new_desc
        # Rewrite the output memlet to span the full tile.
        subset_str = ", ".join(f"0:{w}" for w in widths)
        out_edge.data.subset = dace.subsets.Range.from_string(subset_str)

    def _operand_kind(self, inner_state: SDFGState, edge) -> str:
        """Classify the operand kind for the lib node based on the source descriptor.

        Per design section 6.5: a CONSTANT-only staged source is a Scalar bridge or a
        length-1 Array; the lib node consuming it sees ``kind="Scalar"`` and emits a
        hardware splat to fill the lane register. Tile-shape staged sources (LINEAR /
        AFFINE / REPLICATE / MODULAR / GATHER) wire as ``kind="Tile"``.

        :param inner_state: Inner SDFGState holding the edge.
        :param edge: The wired in-edge to the tasklet / lib node connector.
        :returns: ``"Scalar"`` if the source's descriptor is a Scalar or a length-1
            Array; ``"Tile"`` otherwise.
        """
        import dace.data as dd
        if edge.data is None or edge.data.data is None:
            return "Tile"
        try:
            desc = inner_state.sdfg.arrays[edge.data.data]
        except KeyError:
            return "Tile"
        if isinstance(desc, dd.Scalar):
            return "Scalar"
        if isinstance(desc, dd.Array):
            shape = tuple(desc.shape)
            # Single-element Array (shape == (1,) or all 1s) is treated as Scalar broadcast.
            if all(bool(dace.symbolic.simplify(s - 1) == 0) for s in shape):
                return "Scalar"
        return "Tile"

    def _detect_reduction(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str]]:
        """If ``tasklet`` is an in-place RMW reduction body ``_acc = _acc <op> _val`` (or
        ``_acc = _val <op> _acc`` for commutative ops), return ``(acc_conn, val_conn, op)``.
        Otherwise ``None``.

        Detection criteria:

        * 2 in-connectors, 1 out-connector.
        * The output connector name matches one of the in-connector names (in-place RMW).
        * Body is exactly ``<out> = <out> <op> <other>`` (or the symmetric form) with op
          in :data:`_SUPPORTED_REDUCE_OPS` (associative subset).

        Per the user direction (2026-06-09): a tile -> scalar reduction is the natural
        store-with-reduction case; the in-body accumulator pattern is the entry point. The
        post-walker shape places the tile-shape input on the ``_val`` edge (from a bridge)
        and the scalar accumulator on the ``_acc`` edges. ``TileReduce`` lowers the
        accumulation across the full tile to a single scalar write at the boundary.
        """
        if len(tasklet.in_connectors) != 2 or len(tasklet.out_connectors) != 1:
            return None
        out_conn = next(iter(tasklet.out_connectors))
        if out_conn not in tasklet.in_connectors:
            return None
        # The "other" in-connector is the tile-shape input feeding the reduction.
        other_conns = [c for c in tasklet.in_connectors if c != out_conn]
        if len(other_conns) != 1:
            return None
        other_conn = other_conns[0]
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        for op in _SUPPORTED_REDUCE_OPS:
            if op in ("min", "max"):
                forms = (
                    f"{out_conn} = {op}({out_conn}, {other_conn})",
                    f"{out_conn} = ({op}({out_conn}, {other_conn}))",
                    f"{out_conn} = {op}({other_conn}, {out_conn})",
                    f"{out_conn} = ({op}({other_conn}, {out_conn}))",
                )
            else:
                forms = (
                    f"{out_conn} = {out_conn} {op} {other_conn}",
                    f"{out_conn} = ({out_conn} {op} {other_conn})",
                    f"{out_conn} = {other_conn} {op} {out_conn}",
                    f"{out_conn} = ({other_conn} {op} {out_conn})",
                )
            if body in forms:
                return out_conn, other_conn, op
        return None

    def _detect_ite(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str]]:
        """If ``tasklet`` is a ternary if-then-else body, return
        ``(out_conn, cond_conn, t_conn, e_conn)``. Otherwise ``None``.

        Matches the Python ternary form (DaCe parenthesises the RHS):
        ``_o = _t if _cond else _e`` -> ``_o = (_t if _cond else _e)``.

        Three in-connectors are required; their roles are inferred from the
        position in the expression (``_t``, ``_cond``, ``_e``).
        """
        if len(tasklet.in_connectors) != 3 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        in_conns = list(tasklet.in_connectors)
        # Try every (t, cond, e) ordering of the three connectors; accept both the parenthesised
        # and bare forms.
        from itertools import permutations
        for t, cond, e in permutations(in_conns, 3):
            for form in (f"{out_conn} = {t} if {cond} else {e}", f"{out_conn} = ({t} if {cond} else {e})"):
                if body == form:
                    return out_conn, cond, t, e
        return None

    def _convert_one(self, inner_state: SDFGState, tasklet: Tasklet) -> bool:
        """Replace ``tasklet`` with a :class:`TileBinop` / :class:`TileUnop` /
        :class:`TileITE` / :class:`TileReduce` lib node if its body matches a
        recognised op shape.

        Dispatch order: unary (1 in) -> reduction (2 in, in-place RMW) -> binop
        (2 in, non-RMW) -> ITE (3 in). The reduction check runs BEFORE binop so
        an RMW accumulator pattern ``_acc = _acc + _val`` is never miscaptured
        as a generic ``TileBinop(+)``.

        :returns: ``True`` on rewrite.
        """
        unop = self._detect_unop(tasklet)
        if unop is not None:
            return self._convert_unop(inner_state, tasklet, unop)
        reduction = self._detect_reduction(tasklet)
        if reduction is not None:
            return self._convert_reduction(inner_state, tasklet, reduction)
        binop = self._detect_binop(tasklet)
        if binop is not None:
            return self._convert_binop(inner_state, tasklet, binop)
        ite = self._detect_ite(tasklet)
        if ite is not None:
            return self._convert_ite(inner_state, tasklet, ite)
        return False

    def _convert_reduction(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        acc_conn, val_conn, op = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = {e.src_conn: e for e in inner_state.out_edges(tasklet)}
        if acc_conn not in in_edges or val_conn not in in_edges or acc_conn not in out_edges:
            return False
        val_edge = in_edges[val_conn]
        out_edge = out_edges[acc_conn]
        reduce_node = TileReduce(name=f"{tasklet.label}_reduce", widths=tuple(self.widths), op=op)
        inner_state.add_node(reduce_node)
        # TileReduce connectors: _src (tile input) -> _dst (scalar accumulator). The acc-input
        # edge dangles: TileReduce reads no separate scalar accumulator -- it accumulates over
        # the full tile in one shot. If the original tasklet's acc_in edge was the initial
        # load of the accumulator from a parent state, the inner_sdfg.states() walk will leave
        # it intact on the source AN; the new _dst edge writes the reduction result on top.
        inner_state.add_edge(val_edge.src, val_edge.src_conn, reduce_node, "_src",
                             dace.Memlet.from_memlet(val_edge.data))
        inner_state.add_edge(reduce_node, "_dst", out_edge.dst, out_edge.dst_conn,
                             dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + list(out_edges.values()):
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_ite(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        out_conn, cond_conn, t_conn, e_conn = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if any(c not in in_edges for c in (cond_conn, t_conn, e_conn)) or not out_edges:
            return False
        out_edge = out_edges[0]
        # Forward-analysis output widening: if ANY of cond/t/e is Tile, the output is Tile.
        if any(self._operand_kind(inner_state, in_edges[c]) == "Tile" for c in (cond_conn, t_conn, e_conn)):
            self._widen_output_transient_for_tile(inner_state, out_edge)
        ite = TileITE(name=f"{tasklet.label}_ite", widths=tuple(self.widths))
        inner_state.add_node(ite)
        # TileITE connectors: _cond / _t / _e -> _o.
        for src_conn_name, dst_conn_name in (
            (cond_conn, "_cond"),
            (t_conn, "_t"),
            (e_conn, "_e"),
        ):
            edge = in_edges[src_conn_name]
            inner_state.add_edge(edge.src, edge.src_conn, ite, dst_conn_name, dace.Memlet.from_memlet(edge.data))
        inner_state.add_edge(ite, "_o", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_binop(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        out_conn, a_conn, b_conn, op = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or b_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        b_edge = in_edges[b_conn]
        # Operand-kind classification from the source's descriptor: Scalar / length-1 Array
        # source = broadcast Scalar operand kind on the lib node (design section 6.5).
        kind_a = self._operand_kind(inner_state, a_edge)
        kind_b = self._operand_kind(inner_state, b_edge)
        # Forward-analysis output-kind rule (design 6.2): if any input is Tile, the output
        # is Tile and the destination transient must be widened to (widths,). If all inputs
        # are Scalar / Symbol, leave the destination as-is (the output is Scalar).
        if "Tile" in (kind_a, kind_b):
            self._widen_output_transient_for_tile(inner_state, out_edge)
        binop = TileBinop(name=f"{tasklet.label}_binop", widths=tuple(self.widths), op=op, kind_a=kind_a, kind_b=kind_b)
        inner_state.add_node(binop)
        inner_state.add_edge(a_edge.src, a_edge.src_conn, binop, "_a", dace.Memlet.from_memlet(a_edge.data))
        inner_state.add_edge(b_edge.src, b_edge.src_conn, binop, "_b", dace.Memlet.from_memlet(b_edge.data))
        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_unop(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        out_conn, a_conn, op = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        kind_a = self._operand_kind(inner_state, a_edge)
        # Forward-analysis output widening: if the input is Tile, widen the output transient.
        if kind_a == "Tile":
            self._widen_output_transient_for_tile(inner_state, out_edge)
        unop = TileUnop(name=f"{tasklet.label}_unop", widths=tuple(self.widths), op=op, kind_a=kind_a)
        inner_state.add_node(unop)
        inner_state.add_edge(a_edge.src, a_edge.src_conn, unop, "_a", dace.Memlet.from_memlet(a_edge.data))
        inner_state.add_edge(unop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_inner(self, inner_sdfg: SDFG) -> int:
        """Walk every state of ``inner_sdfg`` and convert recognised binop tasklets.

        :returns: Number of conversions performed.
        """
        converted = 0
        for inner_state in inner_sdfg.states():
            for node in list(inner_state.nodes()):
                if not isinstance(node, Tasklet):
                    continue
                if self._convert_one(inner_state, node):
                    converted += 1
        return converted

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every tile-tagged body NSDFG; convert recognised tasklets to tile lib nodes.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Pipeline results (unused).
        :returns: Number of tasklets converted, or ``None`` if zero.
        """
        total = 0
        for _state, nsdfg_node, _map_entry in self._body_nsdfgs(sdfg):
            total += self._convert_inner(nsdfg_node.sdfg)
        return total or None
