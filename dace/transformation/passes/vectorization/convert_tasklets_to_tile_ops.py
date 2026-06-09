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
from dace.libraries.tileops import TileBinop, TileITE, TileUnop
from dace.sdfg import SDFG
from dace.sdfg.nodes import MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

#: Subset of binary operators that map directly onto :class:`TileBinop`.
_SUPPORTED_BINOPS = {"+", "-", "*", "/", "min", "max"}

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
        :class:`TileITE` lib node if its body matches a recognised op shape.

        :returns: ``True`` on rewrite.
        """
        # Try unary first (single in connector) -- cheaper detection short-circuits.
        unop = self._detect_unop(tasklet)
        if unop is not None:
            return self._convert_unop(inner_state, tasklet, unop)
        binop = self._detect_binop(tasklet)
        if binop is not None:
            return self._convert_binop(inner_state, tasklet, binop)
        ite = self._detect_ite(tasklet)
        if ite is not None:
            return self._convert_ite(inner_state, tasklet, ite)
        return False

    def _convert_ite(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        out_conn, cond_conn, t_conn, e_conn = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if any(c not in in_edges for c in (cond_conn, t_conn, e_conn)) or not out_edges:
            return False
        out_edge = out_edges[0]
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
        binop = TileBinop(name=f"{tasklet.label}_binop", widths=tuple(self.widths), op=op)
        inner_state.add_node(binop)
        a_edge = in_edges[a_conn]
        b_edge = in_edges[b_conn]
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
        unop = TileUnop(name=f"{tasklet.label}_unop", widths=tuple(self.widths), op=op)
        inner_state.add_node(unop)
        a_edge = in_edges[a_conn]
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
