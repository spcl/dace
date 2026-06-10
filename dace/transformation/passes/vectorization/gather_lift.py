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

#: Regex for the K=1 placeholder body emitted by ``materialise_per_lane_index_tile``:
#: ``for (...; __l0 < W; ...) { _out[__l0] = (<dtype>)(<expr>); }``. The tasklet-name
#: prefix is the primary detection gate; this regex parses ``W``, dtype, and the source
#: expression out of an already-confirmed materialiser body.
_K1_BODY_RE = re.compile(
    r"for\s*\(\s*[^)]*?__l0\s*<\s*(?P<width>\d+)\s*;[^)]*?\)\s*\{\s*"
    r"_out\[\s*__l0\s*\]\s*=\s*\(\s*(?P<dtype>[a-zA-Z_][\w:]*)\s*\)\s*\(\s*(?P<expr>[^)]+?)\s*\)\s*;\s*\}",
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


def _lift_one_placeholder_tasklet(inner_state: SDFGState, tasklet: Tasklet, iter_vars: Tuple[str, ...]) -> bool:
    """Detect and expand a single placeholder K=1 gather populate tasklet.

    Returns True if the tasklet was a lifted placeholder; False otherwise.
    """
    if not tasklet.label.startswith(_MATERIALISE_PREFIX):
        return False
    body = tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code)
    match = _K1_BODY_RE.search(body)
    if match is None:
        return False
    expr_str = match.group("expr").strip()
    dtype_str = match.group("dtype").strip()
    width = int(match.group("width"))
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
    dep_iter_vars = [iv for iv in iter_vars if iv in rhs_free]
    if len(dep_iter_vars) != 1:
        # K_dep != 1: multi-dim dependency, deferred to a follow-up slice.
        return False
    dep_iter_var = dep_iter_vars[0]
    per_lane_syms: List[str] = []
    for lane in range(width):
        sym_name = LaneIdScheme.make_multi(lane_dep_sym, ((0, lane), ))
        per_lane_syms.append(sym_name)
        if sym_name not in inner_sdfg.symbols:
            origin_dtype = inner_sdfg.symbols.get(lane_dep_sym, dace.int64)
            inner_sdfg.add_symbol(sym_name, origin_dtype)
        iedge.data.assignments[sym_name] = _shift_expr_by_lane(rhs_template, dep_iter_var, lane)
    unrolled = "\n".join(f"_out[{lane}] = ({dtype_str})({per_lane_syms[lane]});" for lane in range(width))
    tasklet.code = dace.properties.CodeBlock(unrolled, dace.dtypes.Language.CPP)
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
