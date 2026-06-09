# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Forward-propagation pre-shape of body-NSDFG transients.

Per user direction 2026-06-09: "When expanding transient scalars inside the
nsdfg, we should expand accordingly by analyzing tasklet types, so that we
don't need to narrow afterwards. Depending on the copy-in's / writes /
access patterns of the non-transients [pick] scalar / len-1 array or
full tile."

This pass runs **before** :class:`StageInsideBody`. For every tile-tagged
body NSDFG, it walks the inner dataflow graph forward, propagating the
non-transient ANs' access classifications (CONSTANT vs non-CONSTANT) through
every tasklet, and assigns each intermediate transient one of:

* **Scalar / length-1** -- when its producer chain ultimately reads only
  CONSTANT-classified non-transient sources (loop-invariant access).
* **Full tile** ``(widths,)`` -- when any producer in the chain reads a
  non-transient with a non-CONSTANT classification (LINEAR / AFFINE /
  REPLICATE / MODULAR / GATHER), the per-lane value is different per lane
  and a tile-shape register is required.

After this pass runs:

* The walker (:class:`StageInsideBody`) sees correctly-shaped intermediate
  transients and the existing `_widen_output_transient_for_tile` reactive
  widening in :class:`ConvertTaskletsToTileOps` becomes a no-op (the
  transients are already the right shape).
* The lib-node ``validate()`` output-kind contract (design section 6.2)
  trivially matches the in-flight descriptors -- no post-hoc narrowing.

Conservative defaults: when in doubt (e.g. a transient with no detectable
producer in the body, or a producer whose input classification is unknown),
the transient is widened to full tile. Over-widening is benign (extra
register pressure); under-widening is a correctness bug.
"""
from typing import Any, Dict, Optional, Tuple

import dace
from dace import dtypes, properties
import dace.data as dd
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset
from dace.transformation.passes.vectorization.utils.tile_access import PerDimKind, classify_tile_access


@properties.make_properties
@transformation.explicit_cf_compatible
class InferBodyTransientShapes(ppl.Pass):
    """Forward-propagation pre-shape of body-NSDFG transients.

    :ivar widths: Per-tile-dim widths; same shape contract as the walker.
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
            raise ValueError(f"InferBodyTransientShapes: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG.

        Same predicate as :class:`StageInsideBody`.
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

    def _is_tile_kind_for_edge(self, edge, an: AccessNode, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> bool:
        """Classify ``edge``'s an-side subset; True iff the access requires a tile-shape transient.

        Per the design 3.1 staging rule: CONSTANT-only access -> Scalar bridge;
        anything else (LINEAR / AFFINE / REPLICATE / MODULAR / GATHER) -> tile.
        """
        try:
            subset = an_side_subset(edge, an, inner_sdfg)
        except Exception:  # noqa: BLE001 -- helper may refuse on edge shapes outside scope.
            return True  # conservative: assume tile when classification fails.
        record = classify_tile_access(subset, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
        if not record.per_dim_kind:
            return True
        kinds = set(record.per_dim_kind)
        return kinds != {PerDimKind.CONSTANT}

    def _classify_non_transient_ans(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> Dict[str, bool]:
        """Return ``{non_transient_name: is_tile_shape_required}`` over all states.

        A non-transient AN's read AND write edges are considered: if any access
        requires a tile shape, the corresponding lane transient downstream / upstream
        is also tile-shape.
        """
        labels: Dict[str, bool] = {}
        for inner_state in inner_sdfg.states():
            for an in inner_state.nodes():
                if not isinstance(an, AccessNode):
                    continue
                desc = inner_sdfg.arrays.get(an.data)
                if desc is None or desc.transient:
                    continue
                for edge in list(inner_state.out_edges(an)) + list(inner_state.in_edges(an)):
                    if self._is_tile_kind_for_edge(edge, an, inner_sdfg, iter_vars):
                        labels[an.data] = True
                        break
                if an.data not in labels:
                    labels.setdefault(an.data, False)
        return labels

    def _propagate_forward(self, inner_sdfg: SDFG, non_transient_kinds: Dict[str, bool]) -> Dict[str, bool]:
        """Forward-propagate "needs tile shape" through tasklets to every transient.

        Each tasklet output inherits the OR of its inputs: if any input feeds from a
        tile source (or another transient already marked tile), the output transient
        is tile. Iterate to fixed point.

        :returns: ``{transient_name: is_tile_shape_required}``.
        """
        transients: Dict[str, bool] = {}
        # Seed: every transient starts as Scalar (conservative bias toward narrow shapes).
        for name, desc in inner_sdfg.arrays.items():
            if desc.transient:
                transients[name] = False
        # Iterate to fixed point. Each iteration walks every tasklet; if any input is
        # tile-shape (non-transient with tile classification OR transient already marked),
        # the output's transient is tile.
        changed = True
        max_iters = 32
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            for inner_state in inner_sdfg.states():
                for node in inner_state.nodes():
                    if not isinstance(node, Tasklet):
                        continue
                    out_edges = list(inner_state.out_edges(node))
                    in_edges = list(inner_state.in_edges(node))
                    any_input_tile = False
                    for e in in_edges:
                        if e.data is None or e.data.data is None:
                            continue
                        src_name = e.data.data
                        if src_name in non_transient_kinds and non_transient_kinds[src_name]:
                            any_input_tile = True
                            break
                        if src_name in transients and transients[src_name]:
                            any_input_tile = True
                            break
                    if not any_input_tile:
                        continue
                    for e in out_edges:
                        if e.data is None or e.data.data is None:
                            continue
                        dst_name = e.data.data
                        if dst_name in transients and not transients[dst_name]:
                            transients[dst_name] = True
                            changed = True
        return transients

    def _apply_shape_to_transient(self, inner_sdfg: SDFG, name: str, target_tile: bool) -> bool:
        """Replace ``name``'s descriptor + its memlets to the inferred shape.

        Returns True on rewrite, False when no change is needed.
        """
        desc = inner_sdfg.arrays.get(name)
        if desc is None:
            return False
        widths = tuple(self.widths)
        if target_tile:
            # Target: Array(shape=widths). Skip if already correct.
            if isinstance(desc, dd.Array):
                shape = tuple(desc.shape)
                if shape == widths:
                    return False
                # Only widen length-1 Arrays (anything else is user-shaped; leave alone).
                if not all(bool(dace.symbolic.simplify(s - 1) == 0) for s in shape):
                    return False
            elif not isinstance(desc, dd.Scalar):
                return False
            new_desc = dd.Array(dtype=desc.dtype,
                                shape=widths,
                                transient=True,
                                storage=desc.storage if hasattr(desc, "storage") else dtypes.StorageType.Register)
            inner_sdfg.arrays[name] = new_desc
            # Rewrite every memlet referencing this name to span the full tile. We DO NOT
            # touch memlets that already span widths or larger -- only the length-1 ones
            # produced by the python frontend's default scalar wiring.
            subset_str = ", ".join(f"0:{w}" for w in widths)
            for inner_state in inner_sdfg.states():
                for edge in inner_state.edges():
                    if edge.data is None or edge.data.data != name:
                        continue
                    edge.data.subset = dace.subsets.Range.from_string(subset_str)
            return True
        # Target: Scalar / length-1. Already-Scalar / already-length-1 -> no-op.
        if isinstance(desc, dd.Scalar):
            return False
        if isinstance(desc, dd.Array) and all(bool(dace.symbolic.simplify(s - 1) == 0) for s in desc.shape):
            return False
        # We do NOT narrow tile-shape transients down -- that would risk masking a real
        # downstream tile use. Conservative bias toward keeping tiles.
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Pre-shape every body-NSDFG transient based on forward propagation.

        :returns: Total number of transient descriptors rewritten across the SDFG, or
            ``None`` if zero.
        """
        K = len(self.widths)
        rewrites = 0
        for _state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            iter_vars = tuple(map_entry.map.params[-K:])
            inner_sdfg = nsdfg_node.sdfg
            non_transient_kinds = self._classify_non_transient_ans(inner_sdfg, iter_vars)
            transient_kinds = self._propagate_forward(inner_sdfg, non_transient_kinds)
            for name, target_tile in transient_kinds.items():
                if self._apply_shape_to_transient(inner_sdfg, name, target_tile):
                    rewrites += 1
            # Widen non-transient AN edge memlets to span the full tile region (per user
            # direction 2026-06-09): ``A[ii]`` -> ``A[ii:ii+W]`` etc. ONLY for accesses
            # classified as non-CONSTANT (so Scalar / broadcast accesses stay single-element).
            for non_transient_name, needs_tile in non_transient_kinds.items():
                if not needs_tile:
                    continue
                if self._widen_non_transient_memlets(inner_sdfg, non_transient_name, iter_vars):
                    rewrites += 1
        return rewrites or None

    def _widen_non_transient_memlets(self, inner_sdfg: SDFG, name: str, iter_vars: Tuple[str, ...]) -> bool:
        """Widen single-element memlets on edges incident to a non-transient AN.

        For each edge whose memlet's data is ``name`` and whose subset is single-element
        on dims that reference ``iter_vars``, replace those dims with ``[beg : beg + W - 1]``.
        Leaves multi-element / non-iter-var dims untouched, and leaves Scalar / broadcast
        accesses (whose subset doesn't reference any iter_var) alone.
        """
        widths = tuple(self.widths)
        K = len(iter_vars)
        changed = False
        for inner_state in inner_sdfg.states():
            for edge in inner_state.edges():
                if edge.data is None or edge.data.data != name:
                    continue
                if edge.data.subset is None:
                    continue
                try:
                    ranges = list(edge.data.subset.ranges)
                except Exception:  # noqa: BLE001
                    continue
                modified = False
                # Walk every SUBSET dim and find which (if any) iter_var dominates the
                # begin. Widen using THAT iter_var's tile width. Handles partial-dim
                # accesses (subset.dims < K) correctly -- e.g. ``A[ii]`` in K=2 widens
                # to ``A[ii:ii+W_0]`` based on the iter_var ``ii``'s width.
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
                        # The subset begin doesn't reference any iter_var -- it's a
                        # CONSTANT / Symbol access on this dim. Leave it alone.
                        continue
                    w = widths[dominating_k]
                    new_end = dace.symbolic.pystr_to_symbolic(f"({beg}) + {w} - 1")
                    ranges[d] = (beg, new_end, step)
                    modified = True
                if modified:
                    new_subset = dace.subsets.Range(ranges)
                    edge.data.subset = new_subset
                    # Recompute volume from the new subset (Memlet.volume can't be None).
                    edge.data.volume = new_subset.num_elements()
                    changed = True
        return changed
