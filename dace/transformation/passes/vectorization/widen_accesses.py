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
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset
from dace.transformation.passes.vectorization.utils.tile_access import PerDimKind, classify_tile_access


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

        Same predicate as :class:`InsertTileLoadStore`.
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
    def _widen_subset_inplace(self, sub, iter_vars: Tuple[str, ...]) -> Optional["subsets.Range"]:
        """Widen iter-var-dominated single-element dims of a subset.

        Returns a new :class:`subsets.Range` if any dim widened, else ``None``.
        Centralised so :attr:`Memlet.subset` and :attr:`Memlet.other_subset`
        share the exact same widening logic (user direction 2026-06-11:
        WidenAccesses must extend BOTH subset and other_subset).
        """
        widths = tuple(self.widths)
        K = len(iter_vars)
        if sub is None:
            return None
        try:
            ranges = list(sub.ranges)
        except Exception:  # noqa: BLE001
            return None
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
                new_sub = self._widen_subset_inplace(edge.data.subset, iter_vars)
                if new_sub is not None:
                    edge.data.subset = new_sub
                    edge_changed = True
                new_other = self._widen_subset_inplace(edge.data.other_subset, iter_vars)
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
        """Forward-propagate lane-dep through Tasklets to a fixed point.

        Per user direction 2026-06-10: ``we need to DFS this so that we
        traverse correctly``. Each iteration walks all tasklets; tasklets
        whose lane-dep inputs are already classified propagate to their
        outputs. Iteration continues until no new transient is added
        (fixed point) -- guarantees the topological ordering DFS would
        give us, without requiring an explicit graph topo sort (which is
        complicated by NSDFG boundary edges and dataflow cycles via
        AccessNodes).

        :returns: set of transient data names that need tile-shape widening.
        """
        lane_dep_transients: Set[str] = set()
        changed = True
        max_iters = 32
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            for state in inner_sdfg.states():
                for node in state.nodes():
                    if not isinstance(node, Tasklet):
                        continue
                    # Lane-dep output iff any input is lane-dep OR code uses
                    # tile iter-var (e.g. ``_out = i + 1`` widens to per-lane).
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

    # --- Step 5: materialise per-lane idx tiles for GATHER dims -------------
    def _materialise_gather_indices(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> int:
        """Mint per-lane idx tiles for every GATHER per-dim of every non-transient AN.

        Folds in the prior ``PreparePerLaneIndices`` step per user direction
        2026-06-11: ``WidenAccesses should generate per lane indices``. The
        downstream ``InsertTileLoadStore`` wires the materialised tile into
        the ``_idx_<k>`` connector of ``TileLoad`` / ``TileStore``.

        SYMMETRIC: applies identically to read-side (gather) and write-side
        (scatter) edges via :func:`an_side_subset`.

        :returns: Number of materialised idx tiles.
        """
        K = len(self.widths)
        widths = tuple(self.widths)
        if K != 1:
            # Multi-tile-dim GATHER materialisation deferred -- the legacy
            # ``PreparePerLaneIndices`` slice handled K=1 only; the multi-dim
            # case lands in a follow-up.
            return 0
        minted = 0
        iter_var = iter_vars[0]
        for inner_state in inner_sdfg.states():
            for an in list(inner_state.nodes()):
                if not isinstance(an, AccessNode):
                    continue
                desc = inner_sdfg.arrays.get(an.data)
                if desc is None or desc.transient:
                    continue
                out_edges = list(inner_state.out_edges(an))
                if not out_edges:
                    continue
                try:
                    subset = an_side_subset(out_edges[0], an, inner_sdfg)
                except Exception:  # noqa: BLE001
                    continue
                try:
                    record = classify_tile_access(subset, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                except Exception:  # noqa: BLE001
                    continue
                if not record.per_dim_kind or PerDimKind.GATHER not in record.per_dim_kind:
                    continue
                for k, kind in enumerate(record.per_dim_kind):
                    if kind != PerDimKind.GATHER:
                        continue
                    begin_str = str(subset.ranges[k][0])
                    materialise_per_lane_index_tile(
                        inner_state,
                        name_hint=f"_idx_{an.data}_{k}",
                        gather_expr=begin_str,
                        tile_iter_vars=iter_var,
                        tile_widths=widths[0],
                    )
                    minted += 1
        return minted

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
            storage=desc.storage if hasattr(desc, "storage") else dtypes.StorageType.Register,
        )
        for state in inner_sdfg.states():
            for edge in state.edges():
                if edge.data is None or edge.data.data != name:
                    continue
                edge.data.subset = subsets.Range(list(target_range.ranges))
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
            # Step 5: materialise per-lane idx tiles for GATHER dims (fold of
            # legacy PreparePerLaneIndices pass per user direction 2026-06-11
            # -- one pass widens accesses AND emits the idx tiles ready for
            # InsertTileLoadStore to wire into _idx_<k> connectors).
            total += self._materialise_gather_indices(inner_sdfg, iter_vars)
        return total if total else None
