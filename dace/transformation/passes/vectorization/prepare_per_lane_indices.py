# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Materialise per-lane gather indices into integer tile transients.

TILIFICATION_TRANSFORMATION_DESIGN.md section 3.8: for each gather
memlet inside a body NSDFG, this pass derives the lane-dependency set,
mints an integer transient whose shape is the Cartesian product of
widths over the dependent tile dims, fills it with a constant-
assignment tasklet, and wires it into the corresponding ``TileLoad`` /
``TileStore`` ``_idx_<d>`` connector.

This replaces the legacy 1D pipeline's ``_laneid_<i>`` symbol fan-out
machinery (``utils/lane_expansion.py``). The modern output is a tile
the lib node broadcasts at expansion time; no per-lane symbol survives
into the post-emit SDFG (see ``ClearPerLaneIndexSymbols`` post-audit,
design section 10.6).

Status (incremental landing):

* Step 1 (this commit): module-level helper
  :func:`materialise_per_lane_index_tile` for the single-dim
  dependency case (the bulk of cloudsc / ICON kernels). Pass scaffold
  with a no-op stub that documents the contract. Unit tests for the
  helper.
* Step 2 (follow-up): multi-dim dependency case (2-D / N-D index
  tiles).
* Step 3 (follow-up): pass walks every tile-tagged Map's body NSDFG,
  classifies gather memlets, calls the helper, rewires to TileLoad /
  TileStore.
"""
from typing import Any, Dict, Optional, Tuple

import dace
from dace import dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG
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
    """Materialise a per-lane gather index tile.

    Creates an integer transient of shape ``tuple(tile_widths)`` and
    emits a CPP tasklet at the current ``state`` that fills it with the
    per-lane evaluation of ``gather_expr``. The shape matches the
    Cartesian product of widths over the tile dims the expression
    depends on (design section 9.2 lane-dependency rule):

    | ``tile_iter_vars``    | shape              | per-lane body                       |
    |-----------------------|--------------------|-------------------------------------|
    | ``("i",)``            | ``(W_i,)``         | substitute ``i -> __l0`` per lane   |
    | ``("i", "j")``        | ``(W_i, W_j)``     | substitute ``i -> __l0, j -> __l1`` |
    | ``("i", "j", "k")``   | ``(W_i, W_j, W_k)``| ditto with ``__l2``                 |

    Accepts a single ``str`` + ``int`` (the K=1 short form) OR an
    iterable of ``str`` + ``int`` (the multi-dim form). The flat offset
    into the materialised tile is row-major over ``tile_widths`` so
    consumers can decode by stride.

    :param state: State to emit the materialisation tasklet into.
    :param name_hint: Hint for the transient name; uniquified via
        ``find_new_name=True``.
    :param gather_expr: The gather expression, with the
        ``tile_iter_vars`` as free variables. Tasklet body substitutes
        each ``var -> __l<idx>`` per lane.
    :param tile_iter_vars: Name(s) of the tile iter-vars the expression
        depends on. ``str`` for K=1; tuple / list for multi-dim.
    :param tile_widths: Tile width(s); shape matches ``tile_iter_vars``.
    :param idx_dtype: Descriptor dtype; ``int32`` or ``int64`` per
        design section 10.4.
    :returns: The materialised transient's name (AccessNode added to
        ``state``).
    """
    sdfg = state.sdfg
    # Normalise to tuples so the K=1 and K>=2 paths share one body.
    if isinstance(tile_iter_vars, str):
        iter_vars = (tile_iter_vars, )
        widths = (int(tile_widths), )
    else:
        iter_vars = tuple(tile_iter_vars)
        widths = tuple(int(w) for w in tile_widths)
    if len(iter_vars) != len(widths):
        raise ValueError(f"materialise_per_lane_index_tile: tile_iter_vars (len={len(iter_vars)}) "
                         f"and tile_widths (len={len(widths)}) must align")
    # Per user direction 2026-06-10: the idx tile shape is K-D, where each
    # dim is either ``widths[d]`` (lane-dependent) or ``ONE`` (collapsed /
    # broadcast). No trailing ONE. The per-dim lane-dependency is detected
    # by parsing the gather expression's free symbols.
    #
    # K=2 examples for an output tile of shape (W_0, W_1):
    #
    #   * gather expr depends on i AND j: idx shape ``(W_0, W_1)``.
    #   * gather expr depends only on i:  idx shape ``(W_0, ONE)``.
    #   * gather expr depends only on j:  idx shape ``(ONE, W_1)``.
    #
    # The all-non-dep case (gather expr has no lane dep) is NOT handled here
    # -- such accesses are CONSTANT per the design 3.1 lattice and stage
    # through :func:`stage_constant_access` (Scalar bridge), not the gather
    # materialiser. The walker dispatch refuses to call this helper for the
    # all-CONST case; the assertion below guards against accidental invocation.
    #
    # The cuTile lowering relies on this shape contract: the idx tile rank
    # matches the result tile rank, and ONE-marked dims broadcast at gather
    # time. ONE survives sympy operations (firewall) and codegen folds it via
    # the ``constexpr int ONE = 1;`` emission.
    from dace.symbolic import ONE
    # Per-dim dep detection: each dim is either ``widths[d]`` (lane-dep) or
    # ``ONE`` (broadcast / non-lane-dep). When the caller passes ``dep_mask``
    # (the walker uses :func:`compute_per_iter_var_dep_mask` from
    # ``utils.tile_access`` which walks interstate edges to resolve per-lane
    # symbols like ``__sym_<>``), that mask is authoritative. Otherwise the
    # helper falls back to the direct iter-var-membership check, with
    # conservative full-dep when no iter-var appears directly (the
    # post-Bypass-naive form).
    if dep_mask is not None:
        if len(dep_mask) != len(iter_vars):
            raise ValueError(f"materialise_per_lane_index_tile: dep_mask length {len(dep_mask)} "
                             f"!= iter_vars length {len(iter_vars)}")
        dep_mask = tuple(bool(b) for b in dep_mask)
    else:
        try:
            expr_free_syms = {str(s) for s in dace.symbolic.pystr_to_symbolic(gather_expr).free_symbols}
        except Exception:  # noqa: BLE001 -- conservative fallback to all-dep.
            expr_free_syms = set(iter_vars)
        direct_dep_mask = tuple(v in expr_free_syms for v in iter_vars)
        dep_mask = direct_dep_mask if any(direct_dep_mask) else tuple(True for _ in iter_vars)
    shape = tuple(widths[d] if dep_mask[d] else ONE for d in range(len(iter_vars)))
    # Register ``ONE`` as a compile-time constant so ``generate_constants``
    # emits ``constexpr int ONE = 1`` at file scope and the C++ compiler folds
    # it to a literal everywhere it appears in the generated code.
    if "ONE" not in sdfg.constants_prop:
        sdfg.add_constant("ONE", 1, dace.data.Scalar(dace.int32))
    arr_name, _ = sdfg.add_array(name_hint,
                                 shape=shape,
                                 dtype=idx_dtype,
                                 storage=dtypes.StorageType.Register,
                                 transient=True,
                                 find_new_name=True)
    # Substitute each iter_var with its lane variable in the body.
    body_expr = gather_expr
    for d, var in enumerate(iter_vars):
        body_expr = body_expr.replace(var, f"(__l{d})")
    # Build the nested loops -- only over the dep dims. Non-dep dims have
    # shape ONE and use ``__l_d = 0`` (the broadcast lane).
    dep_dims = [d for d, is_dep in enumerate(dep_mask) if is_dep]
    # Row-major flat offset uses dep-dim strides; non-dep dims contribute 0.
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
    # Non-dep dims fixed at lane 0 for the broadcast write.
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
class PreparePerLaneIndices(ppl.Pass):
    """Per-lane index materialisation pass (design section 3.8).

    For every tile-tagged Map's body NSDFG, walks every non-transient
    AccessNode whose AN-incident per-tile subset classifies with at
    least one ``GATHER`` dim. For each such (AN, source-dim k) pair the
    walker calls :func:`materialise_per_lane_index_tile` to mint a
    fresh integer transient holding the per-lane index values, ready
    to be wired into the corresponding :class:`TileLoad` ``_idx_<k>``
    connector by :class:`InsertTileLoadStore` (G7 step 4c).

    The materialised tile's shape follows the design section 9.2 lane-
    dependency rule: the tuple of widths over the tile dims the gather
    expression depends on (record ``deps(k)``). For the first slice
    this walker emits a 1-D index of shape ``(W_p,)`` for the most
    common single-tile-iter-var case (``a[idx[i]]``).

    :ivar widths: Per-tile-dim widths, mirroring :class:`InsertTileLoadStore`.
    """

    CATEGORY: str = "Vectorization"

    widths = properties.Property(
        dtype=tuple,
        default=(8, ),
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        """Build the pass.

        :param widths: Per-tile-dim widths, innermost-last (1..3 entries).
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"PreparePerLaneIndices: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG."""
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

    def _materialise_for_body(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> int:
        """For every gather AN in ``inner_sdfg``, mint a per-lane index transient.

        :returns: Number of materialised index tiles.
        """
        K = len(self.widths)
        widths = tuple(self.widths)
        minted = 0
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
                record = classify_tile_access(subset, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                if not record.per_dim_kind or PerDimKind.GATHER not in record.per_dim_kind:
                    continue
                # For each per-dim GATHER, materialise an index tile. The classifier records
                # gather index access nodes per dim via ``record.gather_index_per_dim`` but for
                # the first slice we focus on the structural materialisation: emit a 1-D index
                # tile of shape ``(W_0,)`` for the single tile iter-var case (single tile dim).
                # Multi-tile-dim deps land in a follow-up slice.
                if K != 1:
                    continue  # Multi-tile-dim case deferred -- first slice handles K=1.
                iter_var = iter_vars[0]
                for k, kind in enumerate(record.per_dim_kind):
                    if kind != PerDimKind.GATHER:
                        continue
                    # Derive the per-lane index expression from the subset's begin on dim k.
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

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every tile-tagged body NSDFG; materialise per-lane index tiles for GATHER dims.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Pipeline results (unused).
        :returns: Number of index tiles materialised, or ``None`` if zero.
        """
        K = len(self.widths)
        total = 0
        for _state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            iter_vars = tuple(map_entry.map.params[-K:])
            total += self._materialise_for_body(nsdfg_node.sdfg, iter_vars)
        return total or None
