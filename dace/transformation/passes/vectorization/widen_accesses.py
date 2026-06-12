# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unified ``WidenAccesses`` pass -- widens lane-dep symbols, non-transient
boundary subsets, and transient descriptors in ONE pass.

Per user direction 2026-06-10: ``1 widen transients pass does implements the
whole design and not need 2 passes`` -- replaces the two-pass split of
``InferBodyTransientShapes`` (descriptor + boundary memlet widening) and
``WidenScalarsToTiles`` (lane-dep gated descriptor widening).

**Symmetry contract** (locked): the rules for gather (indirect READ
``A[idx[i]]``) and scatter (indirect WRITE ``A[idx[i]] = ...``) are
identical. Any decision the pass makes on the read side applies verbatim
to the write side; tests on both directions enforce the invariant.

Algorithm (5 steps, applied per tile-tagged body NSDFG):

1. **Classify non-transient ANs** (lane-dep or CONSTANT). The AN-side
   subset classifier (``classify_tile_access``) sees per-edge subset; any
   non-CONSTANT classification marks the data name as lane-dep. Applies
   uniformly to read-side AND write-side edges -- symmetry guarantee.

2. **Widen non-transient boundary memlets** for the lane-dep ANs:
   ``A[ii]`` -> ``A[ii:ii+W]`` on subset dims dominated by tile iter-vars.
   Direction-agnostic -- the AN-side subset shape decides, not edge
   direction.

3. **Propagate lane-dep through Tasklets via DFS** (topological order).
   For each tasklet (visited after all its lane-dep inputs are classified):
   * Lane-dep input data name -> output is lane-dep.
   * Tasklet code body references a tile iter-var -> output is lane-dep.
   * Otherwise: scalar-scalar / scalar-invariant-sym -> output stays
     loop-invariant.

4. **Widen lane-dep transient descriptors** from Scalar / ``(1,)`` Array
   to ``Array(widths)``. Update every memlet referencing the data to
   ``[0:W_0, ..., 0:W_{K-1}]``.

5. **Subset-widening strategy hook** -- the choice (gather-dim-only vs
   whole-dim) is encoded by how step 2 widens. Symmetric on both sides.
   Reserved for a future ``widen_strategy`` knob; current implementation
   widens all dims that reference tile iter-vars (gather-dim-only is the
   conservative starting point matching InferBody's prior behavior).

After this pass runs, the downstream chain (``GenerateTileIterationMask`` ->
``InsertTileLoadStore`` -> ``GatherLift`` -> ``ConvertTaskletsToTileOps``)
emits gather and scatter ``as usual`` (user direction 2026-06-10).
"""
from typing import Any, Dict, List, Optional, Set, Tuple

import dace
from dace import data as dd
from dace import dtypes, properties, subsets
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset
from dace.transformation.passes.vectorization.utils.tile_access import PerDimKind, classify_tile_access


def _find_iedge_defining_symbol(inner_sdfg: SDFG, sym_name: str):
    """Return ``(interstate_edge, rhs_str)`` for the iedge defining ``sym_name``,
    or ``(None, None)``. Used by :func:`materialise_per_lane_index_tile` to
    detect the Bypass form ``__sym = idx[i]`` for the inline per-lane fan-out
    (folded GatherLift logic per user direction 2026-06-11)."""
    for iedge in inner_sdfg.all_interstate_edges():
        if sym_name in iedge.data.assignments:
            return iedge, iedge.data.assignments[sym_name]
    return None, None


def emit_per_lane_symbol_fanout(sdfg: SDFG,
                                sym_name: str,
                                iter_vars,
                                widths,
                                iter_var_ubs: Optional[Dict[str, Any]] = None) -> Optional[Dict[Tuple[int, ...], str]]:
    """Emit per-lane SDFG symbols + iedge assignments for a Bypass-form gather.

    Idempotent: if the per-lane symbols already exist (an earlier call has
    seeded them), returns the existing map without re-emitting. Designed so
    :class:`WidenAccesses` step 5 can seed symbols upfront and
    :func:`materialise_per_lane_index_tile` can call this without race --
    same name scheme, same iedge assignments.

    Per user direction 2026-06-11: ``wide subsets should emit laneid symbols
    vida a[idx[ii+0]] ... all the way assign them to their symbols (making
    the original symbol etc)``. WidenAccesses owns this so the lane-id
    expansion is a sibling of subset / other_subset widening.

    Remainder-loop safety: when ``iter_var_ubs`` is provided, the per-lane
    shift ``iv -> iv + lane`` is clamped to ``Min(iv + lane, ub)`` so the
    per-lane read of e.g. ``idx[i + lane]`` never reaches past the array
    bound on the masked-tail remainder. Caller is responsible for passing
    each tile iter-var's inclusive upper bound (from
    ``map_entry.map.range[d][1]``).

    :param sdfg: Inner SDFG that hosts the bare symbol.
    :param sym_name: Bare interstate symbol (e.g. ``__sym``).
    :param iter_vars: Tile iter-var names (length K, innermost-last).
    :param widths: Per-dim tile widths (length K).
    :param iter_var_ubs: Optional ``{iter_var_name: ub_expr}`` map; when
        provided the per-lane shift is clamped to ``Min(iv + lane, ub)``.
    :returns: ``{dep_idx_tuple: plane_sym_name}`` mapping for every
        Cartesian-product cell of the dep dims, or ``None`` when the
        symbol isn't defined by an interstate edge / has no iter-var
        dependency / has no walkable RHS.
    """
    import itertools as _itertools
    from dace import symbolic as _sym
    iedge, rhs_template = _find_iedge_defining_symbol(sdfg, sym_name)
    if iedge is None or rhs_template is None:
        return None
    try:
        rhs_free = set(map(str, _sym.pystr_to_symbolic(rhs_template).free_symbols))
    except Exception:  # noqa: BLE001
        return None
    dep_iter_var_indices = [d for d, iv in enumerate(iter_vars) if iv in rhs_free]
    if not dep_iter_var_indices:
        return None
    dep_widths_iter = tuple(widths[d] for d in dep_iter_var_indices)
    dep_iter_var_names = tuple(iter_vars[d] for d in dep_iter_var_indices)
    per_lane_syms: Dict[Tuple[int, ...], str] = {}
    try:
        rhs_sym = _sym.pystr_to_symbolic(rhs_template)
    except Exception:  # noqa: BLE001
        return None
    import sympy as _sp
    for dep_idx in _itertools.product(*(range(w) for w in dep_widths_iter)):
        chunks = tuple(zip(dep_iter_var_indices, dep_idx))
        plane = LaneIdScheme.make_multi(sym_name, chunks)
        per_lane_syms[dep_idx] = plane
        if plane not in sdfg.symbols:
            origin_dtype = sdfg.symbols.get(sym_name, dace.int64)
            sdfg.add_symbol(plane, origin_dtype)
        if plane not in iedge.data.assignments:
            repl: Dict[Any, Any] = {}
            for iv, lane in zip(dep_iter_var_names, dep_idx):
                shifted = _sym.symbol(iv) + lane
                if iter_var_ubs is not None and iv in iter_var_ubs:
                    # Clamp to in-bounds element so the lane-fanout never reads past
                    # the source array on the masked-tail remainder. The mask still
                    # gates the SCATTER write -- this is purely a safe-read clamp.
                    shifted = _sp.Min(shifted, iter_var_ubs[iv])
                repl[_sym.symbol(iv)] = shifted
            iedge.data.assignments[plane] = str(rhs_sym.xreplace(repl))
    return per_lane_syms


def _emit_unrolled_lift_or_loop(sdfg: SDFG, gather_expr_orig: str, body_expr: str,
                                 widths, dep_mask, dep_dims, iter_vars, flat, idx_dtype):
    """Emit either the unrolled per-lane fan-out (Bypass form) or the loop body.

    Bypass form: ``gather_expr_orig`` is a single bare interstate symbol
    defined by an iedge whose RHS references at least one tile iter-var.
    Delegates to :func:`emit_per_lane_symbol_fanout` (idempotent -- shared
    with :class:`WidenAccesses` step 5) and emits unrolled writes sourcing
    from the per-lane symbols.

    Non-Bypass form: emit the K-fold for-loop the materialiser originally
    produced (var-substituted expression evaluated per lane).
    """
    import re as _re
    import itertools as _itertools
    expr_strip = gather_expr_orig.strip()
    bare_sym = _re.fullmatch(r"[A-Za-z_]\w*", expr_strip)
    if bare_sym is not None:
        sym_name = bare_sym.group(0)
        per_lane_syms = emit_per_lane_symbol_fanout(sdfg, sym_name, iter_vars, widths)
        if per_lane_syms is not None:
            # ``per_lane_syms`` is keyed by Cartesian-product cells over the
            # dep dims. Recover the dep dim list from the first key's length
            # so the unrolled-write loop matches the fanout exactly.
            sample_key = next(iter(per_lane_syms))
            dep_iter_var_indices = list(range(len(iter_vars)))[-len(sample_key):] if sample_key else []
            # Better: emit_per_lane_symbol_fanout used the iedge RHS to pick
            # dep dims; recover deterministically by re-walking. The lookup
            # is cheap and avoids the assumption that dep dims are the last
            # ``len(sample_key)`` iter vars.
            _iedge, rhs_template = _find_iedge_defining_symbol(sdfg, sym_name)
            from dace import symbolic as _sym
            rhs_free = set(map(str, _sym.pystr_to_symbolic(rhs_template).free_symbols))
            dep_iter_var_indices = [d for d, iv in enumerate(iter_vars) if iv in rhs_free]
            lines = []
            for all_idx in _itertools.product(*(range(w) for w in widths)):
                dep_key = tuple(all_idx[d] for d in dep_iter_var_indices)
                plane = per_lane_syms[dep_key]
                off = 0
                for k, idx in enumerate(all_idx):
                    stride = 1
                    for q in range(k + 1, len(widths)):
                        stride *= widths[q]
                    off += idx * stride
                lines.append(f"_out[{off}] = ({idx_dtype.ctype})({plane});")
            return lines
    code_lines = []
    for d in dep_dims:
        depth = dep_dims.index(d)
        code_lines.append(f"{'    ' * depth}for (std::size_t __l{d} = 0; __l{d} < {widths[d]}; ++__l{d}) {{")
    for d, is_dep in enumerate(dep_mask):
        if not is_dep:
            indent = '    ' * len(dep_dims)
            code_lines.append(f"{indent}std::size_t __l{d} = 0;")
    code_lines.append(f"{'    ' * len(dep_dims)}_out[{flat}] = ({idx_dtype.ctype})({body_expr});")
    for depth in reversed(range(len(dep_dims))):
        code_lines.append(f"{'    ' * depth}}}")
    return code_lines


def materialise_per_lane_index_tile(state: SDFGState,
                                    name_hint: str,
                                    gather_expr: str,
                                    tile_iter_vars,
                                    tile_widths,
                                    idx_dtype: dtypes.typeclass = dace.int64,
                                    dep_mask=None) -> str:
    """Materialise a per-lane gather index tile (design 3.8 + 3.8.1).

    Creates an integer transient of shape derived from ``tile_widths`` and
    emits a CPP tasklet at ``state`` that fills it with the per-lane
    evaluation of ``gather_expr``. Per-dim lane dependency is encoded with
    :data:`dace.symbolic.ONE` on broadcast dims (no per-cell variation).

    Owned by :mod:`widen_accesses` per user direction 2026-06-11 (was a
    standalone ``PreparePerLaneIndices`` pre-pass before the fold).

    :param state: State that owns the gather AccessNode (the materialiser
        emits its populate tasklet into the same state).
    :param name_hint: Hint for the transient name; uniquified via
        ``find_new_name=True``.
    :param gather_expr: Gather expression with ``tile_iter_vars`` free.
    :param tile_iter_vars: Name(s) of the tile iter-vars the expression
        depends on. ``str`` for K=1; iterable for K>=2.
    :param tile_widths: Tile width(s); shape matches ``tile_iter_vars``.
    :param idx_dtype: Descriptor dtype; ``int32`` or ``int64`` per design 10.4.
    :param dep_mask: Optional per-dim dependency mask (overrides direct iter-
        var-membership detection); see :func:`compute_per_iter_var_dep_mask`.
    :returns: Name of the materialised transient (AccessNode added to ``state``).
    """
    sdfg = state.sdfg
    if isinstance(tile_iter_vars, str):
        iter_vars = (tile_iter_vars, )
        widths = (int(tile_widths), )
    else:
        iter_vars = tuple(tile_iter_vars)
        widths = tuple(int(w) for w in tile_widths)
    if len(iter_vars) != len(widths):
        raise ValueError(f"materialise_per_lane_index_tile: tile_iter_vars (len={len(iter_vars)}) "
                         f"and tile_widths (len={len(widths)}) must align")
    from dace.symbolic import ONE
    if dep_mask is not None:
        if len(dep_mask) != len(iter_vars):
            raise ValueError(f"materialise_per_lane_index_tile: dep_mask length {len(dep_mask)} "
                             f"!= iter_vars length {len(iter_vars)}")
        dep_mask = tuple(bool(b) for b in dep_mask)
    else:
        try:
            expr_free_syms = {str(s) for s in dace.symbolic.pystr_to_symbolic(gather_expr).free_symbols}
        except Exception:  # noqa: BLE001
            expr_free_syms = set(iter_vars)
        direct_dep_mask = tuple(v in expr_free_syms for v in iter_vars)
        dep_mask = direct_dep_mask if any(direct_dep_mask) else tuple(True for _ in iter_vars)
    shape = tuple(widths[d] if dep_mask[d] else ONE for d in range(len(iter_vars)))
    if "ONE" not in sdfg.constants_prop:
        sdfg.add_constant("ONE", 1, dd.Scalar(dace.int32))
    arr_name, _ = sdfg.add_array(name_hint,
                                 shape=shape,
                                 dtype=idx_dtype,
                                 storage=dtypes.StorageType.Register,
                                 transient=True,
                                 find_new_name=True)
    # Word-boundary substitution: naive ``str.replace(var, ...)`` is
    # character-level and mangles unrelated identifiers that happen to
    # contain ``var`` as a substring (e.g. ``dst_slice_0`` -> ``dst_sl(__l0)ce_0``
    # when ``var == "i"``). Use ``\b`` so only standalone tokens match.
    import re as _re
    import itertools as _itertools
    body_expr = gather_expr
    for d, var in enumerate(iter_vars):
        body_expr = _re.sub(rf"\b{_re.escape(var)}\b", f"(__l{d})", body_expr)
    dep_dims = [d for d, is_dep in enumerate(dep_mask) if is_dep]
    parts = []
    for i, d in enumerate(dep_dims):
        inner = 1
        for q in dep_dims[i + 1:]:
            inner *= widths[q]
        parts.append(f"__l{d}" if inner == 1 else f"(__l{d} * {inner})")
    flat = " + ".join(parts) if parts else "0"
    # GatherLift fold (user direction 2026-06-11: ``We can remove gather lift
    # because we already have to do it when emitting loads``): when
    # ``gather_expr`` is a single bare interstate symbol that's lane-dep, emit
    # UNROLLED per-lane writes sourcing from per-lane symbols here -- avoids
    # the placeholder loop body that confuses ConvertTaskletsToTileOps.
    code_lines = _emit_unrolled_lift_or_loop(
        sdfg=sdfg,
        gather_expr_orig=gather_expr,
        body_expr=body_expr,
        widths=widths,
        dep_mask=dep_mask,
        dep_dims=dep_dims,
        iter_vars=iter_vars,
        flat=flat,
        idx_dtype=idx_dtype,
    )
    tasklet = state.add_tasklet(
        name=f"materialise_{arr_name}",
        inputs=set(),
        outputs={"_out"},
        code="\n".join(code_lines),
        language=dtypes.Language.CPP,
    )
    out_an = state.add_access(arr_name)
    out_subset = ", ".join(f"0:{widths[d]}" if dep_mask[d] else "0:ONE" for d in range(len(iter_vars)))
    state.add_edge(tasklet, "_out", out_an, None, Memlet(f"{arr_name}[{out_subset}]"))
    return arr_name


@properties.make_properties
@transformation.explicit_cf_compatible
class WidenAccesses(ppl.Pass):
    """Unified widening: non-transient boundary subsets + transient descriptors,
    lane-dep gated, symmetric between gather and scatter.

    :ivar widths: Per-tile-dim widths; same shape contract as the staging walker.
    """

    CATEGORY: str = "Vectorization"

    widths = properties.Property(
        dtype=tuple,
        default=(8, ),
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"WidenAccesses: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG.

        Same predicate as :class:`InsertTileLoadStore`. Skips the
        ``__scalar_tail`` postamble (step-1 sequential body) and the
        ``__tile_k1_tail`` postamble (pinned at K=1).
        """
        from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                            TILE_K1_TAIL_MARKER)
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
            if node.map.label.endswith(SCALAR_TAIL_MARKER) or node.map.label.endswith(TILE_K1_TAIL_MARKER):
                continue
            try:
                scope_nodes = parent.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
            except (StopIteration, ValueError):
                continue
            nsdfgs = [n for n in scope_nodes if isinstance(n, NestedSDFG)]
            if len(nsdfgs) != 1:
                continue
            yield parent, nsdfgs[0], node

    # --- Step 1: classify non-transient ANs ---------------------------------
    def _classify_non_transients(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> Set[str]:
        """Return data names of non-transient ANs with at least one non-CONSTANT
        adjacent edge. These seed the lane-dep propagation.

        SYMMETRIC: walks BOTH in-edges (writes into the AN) and out-edges
        (reads from the AN) of every non-transient AN. Any non-CONSTANT
        classification on either side marks the data as lane-dep.
        """
        lane_dep: Set[str] = set()
        for state in inner_sdfg.states():
            for an in state.nodes():
                if not isinstance(an, AccessNode):
                    continue
                desc = inner_sdfg.arrays.get(an.data)
                if desc is None or desc.transient:
                    continue
                if an.data in lane_dep:
                    continue
                for edge in list(state.out_edges(an)) + list(state.in_edges(an)):
                    try:
                        sub = an_side_subset(edge, an, inner_sdfg)
                    except Exception:  # noqa: BLE001 -- helper may refuse exotic edges
                        lane_dep.add(an.data)  # conservative
                        break
                    try:
                        record = classify_tile_access(sub, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                    except Exception:  # noqa: BLE001
                        lane_dep.add(an.data)
                        break
                    if not record.per_dim_kind:
                        lane_dep.add(an.data)
                        break
                    if not all(k == PerDimKind.CONSTANT for k in record.per_dim_kind):
                        lane_dep.add(an.data)
                        break
        return lane_dep

    # --- Step 2: widen non-transient boundary memlets -----------------------
    def _widen_subset_inplace(self,
                              sub,
                              iter_vars: Tuple[str, ...],
                              inner_sdfg: Optional[SDFG] = None) -> Optional["subsets.Range"]:
        """Widen LINEAR/AFFINE/REPLICATE/MODULAR single-element dims of a subset.

        Returns a new :class:`subsets.Range` if any dim widened, else ``None``.
        Centralised so :attr:`Memlet.subset` and :attr:`Memlet.other_subset`
        share the exact same widening logic.

        Per user direction 2026-06-11: ``[X, idx[i]:idx[i]+W, 0:N] is
        incorrect, we can't widen like that ... idx[i] should stay idx[i]
        until we add gathers``. GATHER dims (whose begin is itself an array
        subscript referencing the iter-var, e.g. ``idx[i]``) are LEFT
        UNCHANGED -- ``InsertTileLoadStore`` then routes them through the
        scatter / gather emission with the materialised idx tile sized as
        a K-D tile (same rank as the load/store tile shape) where each
        per-dim slot is independently ``W_d`` (lane-dep) or ``ONE``
        (broadcast). Per user direction 2026-06-12: ``we dont need to
        always prepend or append ONE, it completely depends on the tile
        shape of the load or store, we just need to have same
        dimensionality``. The per-dim classifier returns
        ``PerDimKind.GATHER`` for these and we skip widening them here.
        """
        widths = tuple(self.widths)
        K = len(iter_vars)
        if sub is None:
            return None
        try:
            ranges = list(sub.ranges)
        except Exception:  # noqa: BLE001
            return None
        # Per-dim classification (when we have the inner SDFG context). The
        # classifier returns ``PerDimKind`` per dim of the subset; we skip
        # widening dims classified as GATHER (their begin is itself an
        # array subscript referencing an iter-var).
        per_dim_kinds = None
        if inner_sdfg is not None:
            try:
                record = classify_tile_access(sub, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                per_dim_kinds = record.per_dim_kind
            except Exception:  # noqa: BLE001
                per_dim_kinds = None
        modified = False
        for d in range(len(ranges)):
            beg, end, step = ranges[d]
            try:
                is_single = bool(dace.symbolic.simplify(end - beg) == 0)
            except Exception:  # noqa: BLE001
                is_single = False
            if not is_single:
                continue
            try:
                beg_syms = dace.symbolic.SymExpr(str(beg)).free_symbols
            except Exception:  # noqa: BLE001
                beg_syms = set()
            dominating_k = None
            for k in range(K):
                if dace.symbolic.pystr_to_symbolic(iter_vars[k]) in beg_syms:
                    dominating_k = k
                    break
            if dominating_k is None:
                continue
            # Skip GATHER dims -- begin is ``idx[i]`` etc.; widening to
            # ``idx[i]:idx[i]+W-1`` would be a contiguous-range claim that
            # is FALSE (lanes 0..W-1 access W different addresses).
            if per_dim_kinds is not None and d < len(per_dim_kinds) and per_dim_kinds[d] == PerDimKind.GATHER:
                continue
            w = widths[dominating_k]
            new_end = dace.symbolic.pystr_to_symbolic(f"({beg}) + {w} - 1")
            ranges[d] = (beg, new_end, step)
            modified = True
        return subsets.Range(ranges) if modified else None

    def _widen_non_transient_memlets(self, inner_sdfg: SDFG, name: str,
                                     iter_vars: Tuple[str, ...]) -> bool:
        """Widen single-element memlets on edges incident to a non-transient AN.

        SYMMETRIC: identical for read-side and write-side edges and for
        :attr:`Memlet.subset` / :attr:`Memlet.other_subset`. For each edge
        whose memlet's data is ``name``, widen any iter-var-dominated single-
        element dim on BOTH subsets to ``[beg : beg + W_k - 1]``.
        """
        changed = False
        for inner_state in inner_sdfg.states():
            for edge in inner_state.edges():
                if edge.data is None or edge.data.data != name:
                    continue
                edge_changed = False
                new_sub = self._widen_subset_inplace(edge.data.subset, iter_vars, inner_sdfg)
                if new_sub is not None:
                    edge.data.subset = new_sub
                    edge_changed = True
                new_other = self._widen_subset_inplace(edge.data.other_subset, iter_vars, inner_sdfg)
                if new_other is not None:
                    edge.data.other_subset = new_other
                    edge_changed = True
                if edge_changed:
                    if edge.data.subset is not None:
                        edge.data.volume = edge.data.subset.num_elements()
                    changed = True
        return changed

    # --- Step 3: propagate lane-dep through Tasklets (DFS / topological) ----
    @staticmethod
    def _tasklet_references_iter_var(tasklet: Tasklet, iter_vars: Tuple[str, ...]) -> bool:
        """True iff ``tasklet``'s code body references any tile iter-var name."""
        import re
        if tasklet.code is None:
            return False
        code_str = tasklet.code.as_string or ""
        if not code_str:
            return False
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", code_str))
        return any(v in tokens for v in iter_vars)

    @staticmethod
    def _data_names_of_edge(edge, side: str) -> List[str]:
        """Collect all data names that ``edge`` references on ``side``."""
        names = []
        endpoint = edge.src if side == "src" else edge.dst
        if isinstance(endpoint, AccessNode):
            names.append(endpoint.data)
        if edge.data and edge.data.data:
            names.append(edge.data.data)
        return names

    def _propagate_lane_dep(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...],
                            nt_lane_dep: Set[str]) -> Set[str]:
        """Forward-propagate lane-dep through Tasklets AND AN -> AN copies.

        Per user direction 2026-06-10/11: ``we need to DFS this so that we
        traverse correctly``. Two propagation rules per fixed-point step:

        1. **AN -> AN copy** (e.g. ``src --[src[i:i+W]]--> src_index``): if
           the source data name is lane-dep, the destination transient
           becomes lane-dep. Required when staging-first
           (:class:`StageGlobalArrayThroughScalars`) routes a lane-dep non-
           transient through a bridge Scalar; the tasklet then reads the
           bridge, not the non-transient directly. Widening the bridge to a
           tile lets the existing input-staging audit case in
           :class:`InsertTileLoadStore` accept the CopyND-handled
           ``src[i:i+W] -> bridge[0:W]`` edge.
        2. **Tasklet**: any lane-dep input data OR tile-iter-var in the code
           body marks every output transient as lane-dep.

        Iteration continues until no new transient is added (fixed point).

        :returns: set of transient data names that need tile-shape widening.
        """
        lane_dep_transients: Set[str] = set()
        changed = True
        max_iters = 32
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            for state in inner_sdfg.states():
                # Rule 1: AN -> AN copies.
                for edge in state.edges():
                    if edge.data is None or edge.data.data is None:
                        continue
                    if not isinstance(edge.src, AccessNode) or not isinstance(edge.dst, AccessNode):
                        continue
                    src_name = edge.src.data
                    dst_name = edge.dst.data
                    if src_name not in nt_lane_dep and src_name not in lane_dep_transients:
                        continue
                    desc = inner_sdfg.arrays.get(dst_name)
                    if desc is None or not desc.transient:
                        continue
                    if not self._is_widenable(desc):
                        continue
                    if dst_name not in lane_dep_transients:
                        lane_dep_transients.add(dst_name)
                        changed = True
                # Rule 2: Tasklets.
                for node in state.nodes():
                    if not isinstance(node, Tasklet):
                        continue
                    is_lane_dep = self._tasklet_references_iter_var(node, iter_vars)
                    if not is_lane_dep:
                        for e in state.in_edges(node):
                            for nm in self._data_names_of_edge(e, "src"):
                                if nm in nt_lane_dep or nm in lane_dep_transients:
                                    is_lane_dep = True
                                    break
                            if is_lane_dep:
                                break
                    if not is_lane_dep:
                        continue
                    for e in state.out_edges(node):
                        for nm in self._data_names_of_edge(e, "dst"):
                            desc = inner_sdfg.arrays.get(nm)
                            if desc is None or not desc.transient:
                                continue
                            if not self._is_widenable(desc):
                                continue
                            if nm not in lane_dep_transients:
                                lane_dep_transients.add(nm)
                                changed = True
        return lane_dep_transients

    # --- Step 4: widen lane-dep transient descriptors -----------------------
    @staticmethod
    def _is_widenable(desc) -> bool:
        """``Scalar`` or any length-1 ``Array`` transient -> widenable to tile.

        Per user direction 2026-06-10: ``Lane dep length 1 arrays should be
        treated same way as lane-dep scalars``. Length-1 includes the literal
        ``(1,)`` shape AND any multi-dim shape that simplifies to all-1
        (e.g. ``(1, 1)``, ``(k,)`` where ``k`` is statically 1, etc.). Common
        Python frontend artifacts that semantically behave as scalars get
        the identical treatment as ``dd.Scalar``.
        """
        if isinstance(desc, dd.Scalar):
            return True
        if isinstance(desc, dd.Array):
            shape = tuple(desc.shape)
            if not shape:
                return False
            try:
                return all(bool(dace.symbolic.simplify(s - 1) == 0) for s in shape)
            except Exception:  # noqa: BLE001 -- symbolic simplification may refuse
                return False
        return False

    # --- Step 5: seed per-lane symbols for Bypass-form gathers --------------
    def _seed_per_lane_symbols(self,
                               inner_sdfg: SDFG,
                               iter_vars: Tuple[str, ...],
                               iter_var_ubs: Optional[Dict[str, Any]] = None) -> int:
        """Walk every gather memlet on a non-transient AN; for the Bypass form
        (begin is a bare interstate symbol defined by ``__sym = idx[i]``-shape
        iedge), seed per-lane symbols + iedge assignments via
        :func:`emit_per_lane_symbol_fanout`.

        Per user direction 2026-06-11: ``wide subsets should emit laneid
        symbols ... all the way assign them to their symbols (making the
        original symbol etc)``. Per-lane symbol fanout is now WidenAccesses's
        responsibility (sibling of subset / other_subset widening). The
        downstream :func:`materialise_per_lane_index_tile` is idempotent --
        if symbols are pre-seeded, it consumes them directly; if not, it
        seeds them inline as a fallback.

        :returns: Number of (AN, k) pairs that triggered a seed.
        """
        import re as _re
        widths = tuple(self.widths)
        seeded = 0
        for inner_state in inner_sdfg.states():
            for an in list(inner_state.nodes()):
                if not isinstance(an, AccessNode):
                    continue
                desc = inner_sdfg.arrays.get(an.data)
                if desc is None or desc.transient:
                    continue
                for edge in list(inner_state.out_edges(an)) + list(inner_state.in_edges(an)):
                    try:
                        sub = an_side_subset(edge, an, inner_sdfg)
                    except Exception:  # noqa: BLE001
                        continue
                    if sub is None:
                        continue
                    try:
                        record = classify_tile_access(sub, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                    except Exception:  # noqa: BLE001
                        continue
                    if not record.per_dim_kind or PerDimKind.GATHER not in record.per_dim_kind:
                        continue
                    for k, kind in enumerate(record.per_dim_kind):
                        if kind != PerDimKind.GATHER:
                            continue
                        begin_str = str(sub.ranges[k][0]).strip()
                        if _re.fullmatch(r"[A-Za-z_]\w*", begin_str) is None:
                            continue
                        if emit_per_lane_symbol_fanout(inner_sdfg, begin_str, iter_vars, widths,
                                                       iter_var_ubs=iter_var_ubs) is not None:
                            seeded += 1
        return seeded

    def _widen_transient(self, inner_sdfg: SDFG, name: str) -> bool:
        """Swap descriptor to ``Array(widths)`` + rewrite touching memlets.

        Returns True on rewrite, False if not eligible.
        """
        desc = inner_sdfg.arrays.get(name)
        if desc is None or not desc.transient or not self._is_widenable(desc):
            return False
        widths = tuple(self.widths)
        target_subset = ", ".join(f"0:{w}" for w in widths)
        target_range = subsets.Range.from_string(target_subset)
        inner_sdfg.arrays[name] = dd.Array(
            dtype=desc.dtype,
            shape=widths,
            transient=True,
            storage=desc.storage,
        )
        for state in inner_sdfg.states():
            for edge in state.edges():
                if edge.data is None or edge.data.data != name:
                    continue
                new_sub = subsets.Range(list(target_range.ranges))
                edge.data.subset = new_sub
                # CRITICAL: update ``volume`` too -- codegen reads it (NOT
                # ``subset.num_elements()``) to size the CopyND emission. A
                # stale ``volume=1`` from the original Scalar memlet would
                # cause AN -> AN bridge copies to move only 1 element of the
                # widened W-element tile.
                edge.data.volume = new_sub.num_elements()
        return True

    # --- Driver --------------------------------------------------------------
    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Run the unified widening over every tile-tagged body NSDFG.

        :returns: Total number of widenings (descriptor swaps + memlet rewrites
            on non-transients) across the SDFG, or ``None`` if zero.
        """
        K = len(self.widths)
        total = 0
        for _state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            iter_vars = tuple(map_entry.map.params[-K:])
            inner_sdfg = nsdfg_node.sdfg
            # Per-iter-var inclusive upper bound for the lane-fanout clamp
            # (remainder-loop safety: ``idx[i + lane]`` -> ``idx[Min(i + lane, ub)]``).
            iter_var_ubs: Dict[str, Any] = {}
            try:
                for d in range(K):
                    full_d = len(map_entry.map.params) - K + d
                    iter_var_ubs[iter_vars[d]] = map_entry.map.range[full_d][1]
            except Exception:  # noqa: BLE001
                iter_var_ubs = {}
            # Step 1: classify non-transients (which need lane-dep treatment).
            nt_lane_dep = self._classify_non_transients(inner_sdfg, iter_vars)
            # Step 2: widen non-transient boundary memlets. SYMMETRIC on
            # gather (read) and scatter (write) edges -- direction-agnostic.
            for name in nt_lane_dep:
                if self._widen_non_transient_memlets(inner_sdfg, name, iter_vars):
                    total += 1
            # Step 3: propagate lane-dep through Tasklets (DFS-equivalent
            # fixed-point iteration).
            to_widen = self._propagate_lane_dep(inner_sdfg, iter_vars, nt_lane_dep)
            # Step 4: widen lane-dep transient descriptors.
            for name in to_widen:
                if self._widen_transient(inner_sdfg, name):
                    total += 1
            # Step 5: seed per-lane symbols for Bypass-form gather memlets
            # (user direction 2026-06-11: ``wide subsets should emit laneid
            # symbols``). InsertTileLoadStore / materialiser consume the
            # pre-seeded symbols downstream -- idempotent helper, safe to
            # call again if the materialiser re-encounters the same gather.
            # Pass per-iter-var ub for the remainder-loop OOB-clamp.
            total += self._seed_per_lane_symbols(inner_sdfg, iter_vars, iter_var_ubs=iter_var_ubs)
        return total if total else None
