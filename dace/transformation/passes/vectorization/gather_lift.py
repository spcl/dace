# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Post-walker pass that lifts lane-dep gather index tiles via per-lane SDFG symbols.

Per user direction 2026-06-10: when the walker stages an indirect (gather) access
``A[__sym]`` where ``__sym`` is a lane-dependent symbol (defined via an interstate
edge whose RHS references the tile iter-var), the existing
:func:`materialise_per_lane_index_tile` produces a placeholder index tile in which
every lane reads the SAME ``__sym`` value -- so every lane reads ``A[__sym]`` at
the same address. The numerical mismatch surfaces as: lane 0 gets a value, lanes
1..W-1 get garbage / zero.

This pass walks the SDFG post-walker, finds placeholder gather index tiles, and
expands each one into a proper gather:

1. For each lane ``l in 0..W-1``, add a new SDFG symbol ``<sym>_lane<dim>id_<l>``
   to ``inner_sdfg.symbols`` (via :meth:`LaneIdScheme.make_dim`).

2. Add interstate-edge assignments fanning out the original RHS per lane:
   ``<sym>_lane<dim>id_<l> = <RHS with iter_var -> (iter_var + l)>``.

3. Rewrite the placeholder populate tasklet from a single-expression loop body
   into an unrolled set of writes, one per lane, sourcing from the per-lane
   symbols.

The original lane-dep symbol ``__sym`` is intentionally left alive until
:class:`RemoveUnusedPerLaneSymbols` (which runs as the post-clean) sweeps it. If
nothing else references it, it's removed; if it has other live uses, it stays.

Scope (this commit, K=1 only):

* Detects placeholder gather tiles whose populating tasklet body matches the
  K=1 single-loop shape emitted by :func:`materialise_per_lane_index_tile`.
* Lifts via 1-D fan-out per lane.

K=2 / K=3 (multi-dim dependency) is a follow-up; the detection + lift logic
generalises naturally over the Cartesian product of dependent-dim widths but
adding it now would extend this commit beyond the K=1 e2e gather test.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import dace
from dace import properties, symbolic
from dace.sdfg import SDFG
from dace.sdfg.nodes import NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme
from dace.transformation.passes.vectorization.utils.tile_access import _is_tile_dependent

#: Tasklet-name prefix emitted by :func:`materialise_per_lane_index_tile`.
_MATERIALISE_PREFIX = "materialise_"

#: K-agnostic regex matching the innermost ``_out[<flat>] = (<dtype>)(<expr>);``
#: write of a materialiser body. Widths come from the tile transient's descriptor
#: (looked up via the ``_out`` output edge); the regex is only used to extract the
#: source ``<expr>`` and the cast dtype.
_INNER_WRITE_RE = re.compile(
    r"_out\[[^\]]+\]\s*=\s*\(\s*(?P<dtype>[a-zA-Z_][\w:]*)\s*\)\s*\(\s*(?P<expr>[^)]+?)\s*\)\s*;",
    re.DOTALL,
)


def _find_iedge_defining_symbol(inner_sdfg: SDFG, sym_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Return ``(interstate_edge, rhs_str)`` for the iedge that defines ``sym_name``,
    or ``(None, None)`` if no such iedge exists."""
    for iedge in inner_sdfg.all_interstate_edges():
        if sym_name in iedge.data.assignments:
            return iedge, iedge.data.assignments[sym_name]
    return None, None


def _shift_expr_by_lane(rhs_expr: str, iter_var: str, lane: int) -> str:
    """Substitute ``iter_var`` -> ``(iter_var + lane)`` in ``rhs_expr`` via sympy.

    Uses :meth:`sympy.Basic.xreplace` with the substitution value passed as a real
    sympy expression (not a string -- :meth:`sympy.Basic.subs` with a string RHS
    silently no-ops inside DaCe ``Subscript`` wrappers, since the string isn't
    walked as an expression). Falls back to identity when ``rhs_expr`` doesn't
    parse, in which case the original RHS is preserved unchanged.
    """
    try:
        expr = symbolic.pystr_to_symbolic(rhs_expr)
        shifted = expr.xreplace({symbolic.symbol(iter_var): symbolic.symbol(iter_var) + lane})
        return str(shifted)
    except Exception:  # noqa: BLE001
        return rhs_expr


def _get_tile_widths_from_out_edge(inner_state: SDFGState, tasklet: Tasklet) -> Optional[Tuple[int, ...]]:
    """Return the materialised tile's per-dim widths by inspecting the ``_out``
    output edge's destination AccessNode descriptor.

    Uses :func:`dace.symbolic.collapse_one_dims` to drop ``ONE``-marked
    broadcast dims (design 3.8.2 -- ``(W_0, ONE)`` is reported as ``(W_0,)``).
    Returns ``None`` if not found or the shape contains non-integer
    non-``ONE`` symbols.
    """
    from dace.sdfg.nodes import AccessNode
    from dace.symbolic import collapse_one_dims
    inner_sdfg = inner_state.sdfg
    for e in inner_state.out_edges(tasklet):
        if e.src_conn != "_out":
            continue
        if not isinstance(e.dst, AccessNode):
            continue
        desc = inner_sdfg.arrays.get(e.dst.data)
        if desc is None or not hasattr(desc, "shape"):
            continue
        # Opt into ``treat_one_symbol_as_one``: ``ONE`` is a broadcast marker
        # added by ``materialise_per_lane_index_tile``; the lift logic operates
        # over the structurally-equivalent ``(W_0, ..., W_{K-1})`` shape.
        collapsed = []
        for s in collapse_one_dims(desc.shape, treat_one_symbol_as_one=True):
            try:
                collapsed.append(int(s))
            except (TypeError, ValueError):
                return None
        return tuple(collapsed)
    return None


def _shift_expr_by_multi_lane(rhs_expr: str, iter_var_to_lane: Dict[str, int]) -> str:
    """Substitute each ``iter_var -> (iter_var + lane)`` in ``rhs_expr`` via sympy.
    Identity fallback when the expression doesn't parse."""
    try:
        expr = symbolic.pystr_to_symbolic(rhs_expr)
        repl = {symbolic.symbol(iv): symbolic.symbol(iv) + lane for iv, lane in iter_var_to_lane.items()}
        return str(expr.xreplace(repl))
    except Exception:  # noqa: BLE001
        return rhs_expr


def _lift_one_placeholder_tasklet(inner_state: SDFGState, tasklet: Tasklet, iter_vars: Tuple[str, ...]) -> bool:
    """Detect and expand a single placeholder gather populate tasklet (any K).

    The materialiser emits a K-fold nested loop with one inner write
    ``_out[<flat>] = (<dtype>)(<expr>);``. We gate on the tasklet label prefix
    + a single inner-write match, then look up the tile widths from the
    transient descriptor (no need to parse the for-loop headers).

    For K_dep < K (partial dependency) the per-lane symbols are generated only
    over the dependent dims; the populate body broadcasts each symbol across
    the non-dependent dims (each non-dep cell at the same dep-indices sources
    from the SAME symbol). This is the full-tile-always design with per-dim
    fan-out collapsing where ``ONE`` would mark the broadcast dim once the
    ``ONE``-marker optimisation lands as a follow-up.

    Returns True if the tasklet was a lifted placeholder; False otherwise.
    """
    import itertools
    if not tasklet.label.startswith(_MATERIALISE_PREFIX):
        return False
    body = tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code)
    match = _INNER_WRITE_RE.search(body)
    if match is None:
        return False
    expr_str = match.group("expr").strip()
    dtype_str = match.group("dtype").strip()
    widths = _get_tile_widths_from_out_edge(inner_state, tasklet)
    if widths is None or len(widths) == 0:
        return False
    inner_sdfg = inner_state.sdfg
    try:
        free_syms = set(map(str, symbolic.pystr_to_symbolic(expr_str).free_symbols))
    except Exception:  # noqa: BLE001
        return False
    if len(free_syms) != 1:
        return False
    lane_dep_sym = next(iter(free_syms))
    iter_var_set = set(iter_vars)
    if not _is_tile_dependent(lane_dep_sym, iter_var_set, inner_sdfg):
        return False
    iedge, rhs_template = _find_iedge_defining_symbol(inner_sdfg, lane_dep_sym)
    if iedge is None or rhs_template is None:
        return False
    try:
        rhs_free = set(map(str, symbolic.pystr_to_symbolic(rhs_template).free_symbols))
    except Exception:  # noqa: BLE001
        return False
    # Dependent dims are the iter-var indices whose names appear in the iedge
    # RHS. ``dep_dims`` is the source-order index list; ``dep_widths`` is the
    # corresponding width tuple used to size the per-lane symbol fan-out.
    K = len(iter_vars)
    dep_dims = [d for d, iv in enumerate(iter_vars) if iv in rhs_free]
    if not dep_dims:
        # No iter-var dependency: nothing to fan out, the expression is
        # genuinely constant -- leave the placeholder as-is (the existing
        # for-loop already produces the right value for every lane).
        return False
    if len(dep_dims) != len(widths):
        # Partial K_dep: only fan out over the dependent dims; the non-dep dims
        # broadcast (each non-dep cell at the same dep-indices sources from the
        # SAME symbol, which the unrolled writes below replicate).
        pass
    dep_widths = tuple(widths[d] for d in dep_dims)
    dep_iter_vars = tuple(iter_vars[d] for d in dep_dims)

    # Mint one per-lane SDFG symbol per Cartesian-product cell of the dep dims;
    # name encodes only the dependent-dim indices (``LaneIdScheme.make_multi``
    # with the (dim, lane) chunks in source order).
    per_lane_syms: Dict[Tuple[int, ...], str] = {}
    for dep_indices in itertools.product(*(range(w) for w in dep_widths)):
        chunks = tuple(zip(dep_dims, dep_indices))
        sym_name = LaneIdScheme.make_multi(lane_dep_sym, chunks)
        per_lane_syms[dep_indices] = sym_name
        if sym_name not in inner_sdfg.symbols:
            origin_dtype = inner_sdfg.symbols.get(lane_dep_sym, dace.int64)
            inner_sdfg.add_symbol(sym_name, origin_dtype)
        iter_var_to_lane = dict(zip(dep_iter_vars, dep_indices))
        iedge.data.assignments[sym_name] = _shift_expr_by_multi_lane(rhs_template, iter_var_to_lane)

    # Build the unrolled populate body: one write per (l_0, ..., l_{K-1}) cell.
    # Non-dep cells at the same dep-indices source from the SAME symbol -- the
    # broadcast semantic the ONE-marker would express structurally.
    lines: List[str] = []
    for all_indices in itertools.product(*(range(w) for w in widths)):
        dep_key = tuple(all_indices[d] for d in dep_dims)
        sym_name = per_lane_syms[dep_key]
        # Flat row-major offset (matches the materialiser's emission convention).
        flat_off = 0
        for k, idx in enumerate(all_indices):
            stride = 1
            for q in range(k + 1, len(widths)):
                stride *= widths[q]
            flat_off += idx * stride
        lines.append(f"_out[{flat_off}] = ({dtype_str})({sym_name});")
    tasklet.code = dace.properties.CodeBlock("\n".join(lines), dace.dtypes.Language.CPP)
    return True


@properties.make_properties
@transformation.explicit_cf_compatible
class GatherLift(ppl.Pass):
    """Post-walker pass that lifts lane-dep gather placeholders into per-lane symbol fan-outs.

    Per the locked design from the 2026-06-10 session: the walker stages the gather
    via a placeholder index tile filled with a bare lane-dep symbol. This pass detects
    each placeholder, fans out the underlying iedge RHS into per-lane symbols via
    :meth:`LaneIdScheme.make_dim`, and rewrites the populate tasklet to source per
    lane.

    The original lane-dep symbol survives this pass (the iedge assignment that
    defined it stays intact). :class:`RemoveUnusedPerLaneSymbols`, which runs as the
    final post-clean, sweeps it when no remaining consumer references it.

    :ivar widths: Per-tile-dim widths, mirroring :class:`StageInsideBody` /
        :class:`ConvertTaskletsToTileOps`. Used to size the fan-out per dim.
    """

    CATEGORY: str = "Vectorization Cleanup"

    widths = properties.Property(
        dtype=tuple,
        default=(8, ),
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"GatherLift: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Tasklets | ppl.Modifies.Edges | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _lift_in_inner_sdfg(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> int:
        """Walk every state in ``inner_sdfg`` and lift any K=1 placeholder gather
        tasklets. Returns the number of lifts performed."""
        lifted = 0
        for inner_state in inner_sdfg.states():
            for node in list(inner_state.nodes()):
                if isinstance(node, Tasklet) and _lift_one_placeholder_tasklet(inner_state, node, iter_vars):
                    lifted += 1
        return lifted

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Optional[Dict[str, Any]]) -> Optional[int]:
        """Lift every K=1 placeholder gather populate tasklet in every body NSDFG.

        The walker associates each body NSDFG with a parent ``MapEntry`` whose
        ``params`` end in the K tile iter-vars; the lift uses the last K params as
        the tile iter-vars passed to :func:`_is_tile_dependent`.
        """
        from dace.sdfg.nodes import MapEntry
        from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
        K = len(self.widths)
        total = 0
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
            iter_vars = tuple(node.map.params[-K:])
            try:
                scope_nodes = parent.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
            except (StopIteration, ValueError):
                continue
            for inner_node in scope_nodes:
                if isinstance(inner_node, NestedSDFG):
                    total += self._lift_in_inner_sdfg(inner_node.sdfg, iter_vars)
        return total or None
