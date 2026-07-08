# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unified ``WidenAccesses`` pass: widen lane-dep symbols, non-transient boundary subsets, and
transient descriptors in ONE pass. Replaces the two-pass ``InferBodyTransientShapes`` +
``WidenScalarsToTiles`` split.

Symmetry contract (locked): gather (indirect READ ``A[idx[i]]``) and scatter (indirect WRITE
``A[idx[i]] = ...``) obey identical rules; tests enforce both directions.

Algorithm (5 steps, per tile-tagged body NSDFG):

1. Classify non-transient ANs via ``classify_tile_access`` on the AN-side subset. Any
   non-CONSTANT -> data name lane-dep. Read + write edges treated uniformly.
2. Widen non-transient boundary memlets of lane-dep ANs: ``A[ii]`` -> ``A[ii:ii+W]`` on
   iter-var-dominated dims. AN-side subset decides, not edge direction.
3. Propagate lane-dep through Tasklets (fixed point): lane-dep input OR iter-var in code body ->
   output lane-dep; else loop-invariant.
4. Widen lane-dep transient descriptors Scalar / ``(1,)`` Array -> ``Array(widths)``; rewrite
   touching memlets to ``[0:W_0,...,0:W_{K-1}]``.
5. Subset-widening strategy hook: gather-dim-only vs whole-dim, encoded by how step 2 widens;
   symmetric. Reserved for a future ``widen_strategy`` knob; current impl widens all iter-var
   dims (gather-dim-only conservative default).

Downstream chain ``GenerateTileIterationMask`` -> ``InsertTileLoadStore`` -> ``GatherLift`` ->
``ConvertTaskletsToTileOps`` then emits gather/scatter.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

import dace
from dace import data as dd
from dace import properties, subsets
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant,
                                                                            lane_dep_transients_widened,
                                                                            no_memlet_dim_mismatch)
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset
from dace.transformation.passes.vectorization.utils.tile_access import PerDimKind, classify_tile_access


def _find_iedge_defining_symbol(inner_sdfg: SDFG, sym_name: str):
    """``(iedge, rhs_str)`` for the iedge defining ``sym_name``, else
    ``(None, None)``. Detects Bypass form ``__sym = idx[i]`` for per-lane fanout."""
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

    Idempotent: returns existing map if symbols already seeded. WidenAccesses owns this (sibling
    of subset/other_subset widening) so downstream :class:`InsertTileLoadStore` gather-index path
    sees a consistent name scheme.

    Remainder safety: with ``iter_var_ubs``, per-lane shift ``iv -> iv + lane`` is clamped
    ``Min(iv + lane, ub)`` so ``idx[i + lane]`` never reads past the array bound on the masked
    tail. Caller passes each iter-var's inclusive ub (``map_entry.map.range[d][1]``).

    :param sdfg: Inner SDFG hosting the bare symbol.
    :param sym_name: Bare interstate symbol (e.g. ``__sym``).
    :param iter_vars: Tile iter-var names (length K, innermost-last).
    :param widths: Per-dim tile widths (length K).
    :param iter_var_ubs: Optional ``{iter_var: ub_expr}``; clamps shift to ``Min(iv + lane, ub)``.
    :returns: ``{dep_idx_tuple: plane_sym_name}`` over the dep-dim Cartesian product, or ``None``
        if the symbol has no iedge definition / no iter-var dependency / no walkable RHS.
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
                    # Clamp in-bounds: lane-fanout never reads past source on the masked tail.
                    # Mask still gates the SCATTER write; safe-read only.
                    shifted = _sp.Min(shifted, iter_var_ubs[iv])
                repl[_sym.symbol(iv)] = shifted
            iedge.data.assignments[plane] = str(rhs_sym.xreplace(repl))
    return per_lane_syms


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
        """Yield ``(state, nsdfg_node, map_entry)`` per tile-tagged body NSDFG.

        Same predicate as :class:`InsertTileLoadStore`. Skips ``__scalar_tail``
        (sequential body) and ``__tile_k1_tail`` (pinned K=1) postambles.
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
                if not is_vectorizable_map(parent, node):
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
        """Non-transient AN data names with >=1 non-CONSTANT adjacent edge; seed
        the lane-dep propagation.

        SYMMETRIC: walks in-edges (writes) AND out-edges (reads); non-CONSTANT on
        either side marks the data lane-dep.
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
                        record = classify_tile_access(sub, iter_vars=iter_vars, inner_sdfg=inner_sdfg, state=state)
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
                              inner_sdfg: Optional[SDFG] = None,
                              state=None) -> Optional["subsets.Range"]:
        """Widen LINEAR/AFFINE/REPLICATE/MODULAR single-element dims of a subset.

        Returns a new :class:`subsets.Range` if any dim widened, else ``None``. Centralised so
        :attr:`Memlet.subset` and :attr:`Memlet.other_subset` share one widening path.

        GATHER dims (begin is an array subscript on the iter-var, e.g. ``idx[i]``) LEFT UNCHANGED:
        widening ``idx[i]`` to a contiguous range is false; ``InsertTileLoadStore`` routes them
        through gather/scatter emission with a materialised idx tile of matching rank. Classifier
        returns ``PerDimKind.GATHER`` -> skip here.
        """
        widths = tuple(self.widths)
        K = len(iter_vars)
        if sub is None:
            return None
        try:
            ranges = list(sub.ranges)
        except Exception:  # noqa: BLE001
            return None
        # Per-dim classification (needs inner SDFG context); skip GATHER dims
        # (begin is an iter-var array subscript).
        per_dim_kinds = None
        if inner_sdfg is not None:
            try:
                record = classify_tile_access(sub, iter_vars=iter_vars, inner_sdfg=inner_sdfg, state=state)
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
            # Skip GATHER dims (see docstring): widening ``idx[i]`` falsely claims contiguity.
            if per_dim_kinds is not None and d < len(per_dim_kinds) and per_dim_kinds[d] == PerDimKind.GATHER:
                continue
            w = widths[dominating_k]
            new_end = dace.symbolic.pystr_to_symbolic(f"({beg}) + {w} - 1")
            ranges[d] = (beg, new_end, step)
            modified = True
        return subsets.Range(ranges) if modified else None

    def _widen_non_transient_memlets(self, inner_sdfg: SDFG, name: str, iter_vars: Tuple[str, ...]) -> bool:
        """Widen single-element memlets on edges incident to a non-transient AN.

        SYMMETRIC over read/write edges and subset/other_subset: per edge whose data is ``name``,
        widen iter-var-dominated single-element dims on BOTH subsets to ``[beg : beg + W_k - 1]``.
        """
        changed = False
        for inner_state in inner_sdfg.states():
            for edge in inner_state.edges():
                if edge.data is None or edge.data.data != name:
                    continue
                edge_changed = False
                new_sub = self._widen_subset_inplace(edge.data.subset, iter_vars, inner_sdfg, state=inner_state)
                if new_sub is not None:
                    edge.data.subset = new_sub
                    edge_changed = True
                new_other = self._widen_subset_inplace(edge.data.other_subset, iter_vars, inner_sdfg, state=inner_state)
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

    @staticmethod
    def _index_promoted_names(inner_sdfg: SDFG) -> Set[str]:
        """Data names consumed as *index symbols*: appear as a free symbol in some interstate-edge
        assignment RHS (``__sym_i_plus_offset1 = i_plus_offset1`` promotes scalar
        ``i_plus_offset1`` to an index symbol used in subsets).

        Such a transient is an address/index, not a data operand -> must NOT widen to a tile even
        when its defining tasklet references an iter-var (tile load consumes it as the per-lane
        base, kept scalar). Excluding these stops the index scalar becoming ``int64_t*`` and
        breaking the ``__sym = i_plus_offset1`` codegen assignment.
        """
        promoted: Set[str] = set()
        names = set(inner_sdfg.arrays.keys())
        for edge in inner_sdfg.all_interstate_edges():
            assigns = edge.data.assignments if edge.data is not None else {}
            for rhs in assigns.values():
                try:
                    expr = dace.symbolic.pystr_to_symbolic(str(rhs))
                    promoted |= {str(s) for s in expr.free_symbols} & names
                except Exception:  # noqa: BLE001 -- unparseable RHS -> token fallback
                    promoted |= {tok for tok in re.findall(r"\b[A-Za-z_]\w*\b", str(rhs)) if tok in names}
        return promoted

    def _propagate_lane_dep(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...], nt_lane_dep: Set[str]) -> Set[str]:
        """Forward-propagate lane-dep through Tasklets AND AN -> AN copies to a fixed point. Two
        rules per step:

        1. AN -> AN copy (``src --[src[i:i+W]]--> src_index``): lane-dep source -> dest transient
           lane-dep. Needed when :class:`StageGlobalArrayThroughScalars` routes a lane-dep
           non-transient through a bridge Scalar; widening the bridge lets
           :class:`InsertTileLoadStore`'s input-staging audit accept the CopyND
           ``src[i:i+W] -> bridge[0:W]`` edge.
        2. Tasklet: any lane-dep input OR tile iter-var in the code body -> every output transient
           lane-dep.

        :returns: transient data names needing tile-shape widening.
        """
        lane_dep_transients: Set[str] = set()
        # Index symbols (scalars promoted to symbols in subsets) are addresses,
        # not data -> stay scalar; excluded even if their tasklet uses an iter-var.
        index_symbols = self._index_promoted_names(inner_sdfg)
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
                    if dst_name in index_symbols:
                        continue  # index/address symbol -> stays scalar
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
                            if nm in index_symbols:
                                continue  # index/address symbol -> stays scalar
                            if nm not in lane_dep_transients:
                                lane_dep_transients.add(nm)
                                changed = True
        return lane_dep_transients

    # --- Step 4: widen lane-dep transient descriptors -----------------------
    @staticmethod
    def _is_widenable(desc) -> bool:
        """``Scalar`` or any length-1 ``Array`` transient -> widenable to tile.

        Length-1 = literal ``(1,)`` OR any shape simplifying to all-1 (``(1, 1)``,
        ``(k,)`` with k statically 1). Such scalar-like frontend artifacts get the
        same treatment as ``dd.Scalar``.
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
        """Seed per-lane symbols + iedge assignments (via :func:`emit_per_lane_symbol_fanout`) for
        every Bypass-form gather memlet on a non-transient AN (begin is a bare interstate symbol
        from ``__sym = idx[i]``). Downstream :class:`InsertTileLoadStore` gather-index path
        consumes them.

        :returns: number of (AN, k) pairs seeded.
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
                        record = classify_tile_access(sub,
                                                      iter_vars=iter_vars,
                                                      inner_sdfg=inner_sdfg,
                                                      state=inner_state)
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
                        if emit_per_lane_symbol_fanout(inner_sdfg,
                                                       begin_str,
                                                       iter_vars,
                                                       widths,
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
                # CRITICAL: update ``volume`` too -- codegen sizes CopyND from it, NOT
                # ``subset.num_elements()``. Stale ``volume=1`` from the Scalar memlet would copy
                # only 1 element of the W-element tile.
                edge.data.volume = new_sub.num_elements()
                # Widen ``other_subset`` symmetrically (AN -> AN copy ``a[i] -> b[0]``); else
                # validator trips ``Dimensionality mismatch between src/dst subsets``.
                if edge.data.other_subset is not None:
                    edge.data.other_subset = subsets.Range(list(target_range.ranges))
        return True

    # --- Driver --------------------------------------------------------------
    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Run the unified widening over every tile-tagged body NSDFG.

        :returns: Total widenings (descriptor swaps + memlet rewrites on non-transients) across
            the SDFG, or ``None`` if zero.
        """
        K = len(self.widths)
        total = 0
        for _state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            iter_vars = tuple(map_entry.map.params[-K:])
            inner_sdfg = nsdfg_node.sdfg
            # Per-iter-var inclusive ub for the lane-fanout clamp (remainder
            # safety: ``idx[i + lane]`` -> ``idx[Min(i + lane, ub)]``).
            iter_var_ubs: Dict[str, Any] = {}
            try:
                for d in range(K):
                    full_d = len(map_entry.map.params) - K + d
                    iter_var_ubs[iter_vars[d]] = map_entry.map.range[full_d][1]
            except Exception:  # noqa: BLE001
                iter_var_ubs = {}
            # Step 1: classify non-transients (which need lane-dep treatment).
            nt_lane_dep = self._classify_non_transients(inner_sdfg, iter_vars)
            # Step 2: widen non-transient boundary memlets. SYMMETRIC over
            # gather/scatter edges.
            for name in nt_lane_dep:
                if self._widen_non_transient_memlets(inner_sdfg, name, iter_vars):
                    total += 1
            # Step 3: propagate lane-dep through Tasklets (fixed point).
            to_widen = self._propagate_lane_dep(inner_sdfg, iter_vars, nt_lane_dep)
            # Step 4: widen lane-dep transient descriptors.
            for name in to_widen:
                if self._widen_transient(inner_sdfg, name):
                    total += 1
            # Step 5: seed per-lane symbols for Bypass-form gathers. Idempotent;
            # InsertTileLoadStore/materialiser consume them. Pass per-iter-var ub
            # for the remainder OOB-clamp.
            total += self._seed_per_lane_symbols(inner_sdfg, iter_vars, iter_var_ubs=iter_var_ubs)
        # Post-conditions (always run).
        assert_invariant(no_memlet_dim_mismatch(sdfg), "WidenAccesses",
                         "memlet subset and other_subset have matching dimensionality")
        assert_invariant(lane_dep_transients_widened(sdfg, K, tuple(self.widths)), "WidenAccesses",
                         "lane-dep transients widened to (W_0,...,W_{K-1}) or kept as Scalar bridge")
        return total if total else None
